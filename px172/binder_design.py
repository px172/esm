# /// script
# requires-python = "<=3.13"
# dependencies = [
#     "abnumber",
#     "esm@git+https://github.com/Biohub/esm.git@main",
# ]
# ///
# NOTE - "modal" is intentionally omitted from the dependencies above so that
# local runs (`local=True`) do not require it. Install it separately (or use the
# Modal CLI) only when you want to run the design remotely on Modal.
"""
Code for binder design with ESMFold2 and ESMC.

As described in [Language Modeling Materializes a World Model of Protein Biology](https://www.biorxiv.org/content/10.64898/2026.06.03.729735).

px172 fork of cookbook/tutorials/binder_design.py, rebased onto upstream cf002f1.
Adds: --gpu selection, epitope-restricted interface loss, sequence loading from
file, batch design runs with PDB/CSV/TSV output, and a CLI.
"""

# ruff: noqa: E402
# The --gpu scan below must run before torch is imported, so the imports in this
# module cannot all sit at the top of the file.

import os
import sys

# CUDA_VISIBLE_DEVICES must be set before torch creates a CUDA context, so --gpu
# is resolved from argv here rather than after argparse runs.
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break
    if _arg.startswith("--gpu="):
        os.environ["CUDA_VISIBLE_DEVICES"] = _arg.split("=", 1)[1]
        break

import csv
import logging
import math
import random
import string
import time
from dataclasses import dataclass
from functools import cache, partial
from typing import Any

import biotite.structure
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
from transformers.models.esmc.modeling_esmc import (
    UnifiedTransformerBlock as TransformerBlock,
)
from transformers.models.esmc.tokenization_esmc import ESMCTokenizer
from transformers.models.esmfold2.modeling_esmfold2_common import (
    CUE_AVAILABLE,
    PairUpdateBlock,
)
from transformers.models.esmfold2.modeling_esmfold2_common import (
    _seed_context as seed_context,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    ESMFold2ExperimentalModel,
)
from transformers.models.esmfold2.modeling_esmfold2_experimental import (
    MSAEncoder as ESMFold2MSAEncoder,
)

from esm.models.esmfold2 import (
    ELEMENT_NUMBER_TO_SYMBOL,
    ProteinInput,
    StructurePredictionInput,
    load_ccd,
    prepare_esmfold2_input,
)
from esm.models.esmfold2.constants import (
    MOL_TYPE_NONPOLYMER,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    RES_TYPE_TO_CCD,
)
from esm.utils.structure.mmcif_parsing import PLDDT_B_FACTOR_SCALE
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

# Modal is only needed to run designs remotely; local runs work without it.
try:
    import modal
except ImportError:
    modal = None

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TrajectoryStep = dict[str, torch.Tensor | float]
Trajectory = dict[int, TrajectoryStep]


# ---- Constants ----


# General
TOKENS = ["<pad>", "-"] + [RES_TYPE_TO_CCD[i] for i in range(2, 33)]
ELEMENTS = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
ELEMENTS[0] = "<pad>"
for _atomic_num, _symbol in ELEMENT_NUMBER_TO_SYMBOL.items():
    ELEMENTS[_atomic_num] = _symbol[:1] + _symbol[1:].lower()
TOKEN_IDS = {token: idx for idx, token in enumerate(TOKENS)}
AA_DIMS = 20
# Cysteine index in the 20-dim AA space (TOKEN_IDS are offset by 2 for <pad> and -)
CYS_IDX = TOKEN_IDS[PROTEIN_1TO3["C"]] - 2
MUTABLE_TOKEN = "#"
# Contains AA chars at fixed positions and MUTABLE_TOKEN at mutable positions.
BinderPromptStr = str

# Design
# px172: which residues count as CDR for cdr_distogram_iptm_proxy. abnumber's
# SUPPORTED_CDR_DEFINITIONS; "chothia" is upstream's hard-coded behaviour. Measured
# on this fork's IL8 scFv, these pick out 47 / 51 / 60 / 64 residues respectively,
# and only "north" (contact-based) covers every designed position in GH2-H3_13.txt.
# This affects scoring only -- the designed positions are set by '#' in the prompt.
CDR_DEFINITIONS = ("imgt", "chothia", "kabat", "north")
CDR_DEFINITION = "chothia"

# px172: heavy-atom distance below which a target residue counts as contacted by the
# binder in the predicted structure. 5 A is the usual heavy-atom contact convention.
# NOTE - unrelated to the 22 A cutoff in compute_inter_contact_loss: that one is a
# soft "get into the neighbourhood" term on the distogram, which is why it cannot by
# itself pin a binder to an epitope. This is a real geometric contact check.
INTERFACE_CONTACT_CUTOFF = 5.0

LOSS_WEIGHTS = {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2}
STEPS = 150
LOG_INTERVAL = 5
LEARNING_RATE = 0.1
TEMPERATURE_MIN = 1e-2
ESMC_MASK_FRACTION = 0.15
LM_LOSS_BATCH_SIZE = 128
LM_MASK_PASSES = 4
BYTES_PER_GIB = 1024**3
# NOTE - measured on this fork's IL8 config (cxcl8 + a 243-residue scFv, 320 tokens,
# batch_size=1): 1.30 s/step compiled vs 2.11 s/step eager, for ~6s of one-time
# warmup and slightly less VRAM. Compilation is keyed on shape, so this pays off
# because the binder prompt has a fixed length; a preset binder that samples a new
# CDR length per seed would recompile on every new length instead.
COMPILE = True
# px172: activation-checkpoint the ESMC transformer blocks to cut VRAM.
# NOTE - upstream removed this in cf002f1 in favour of REUSE_ESMC + a larger
# LM_LOSS_BATCH_SIZE. It is kept here because it is a deliberate fork choice, but
# be aware of the interaction with REUSE_ESMC documented in ESMFold2Design.load.
CHECKPOINT_LM = True
# NOTE - This significantly reduces VRAM usage.
# On config (target_name="cd45", binder_name="trastuzumab_framework_vhvl", batch_size=1)
# this reduces VRAM from 51GB -> 27GB.  And enables increasing batch size up to 6.
# We are testing this setting in silico, and may change the default to True, in the future.
REUSE_ESMC = True


# ---- Prompts ----


@dataclass(frozen=True)
class PromptFactory:
    """A simple factory for making binder prompt strings."""

    name: str
    template: str  # string with format fields
    length_ranges: dict[str, tuple[int, int]]  # map from field name tp length range
    is_antibody: bool  # Used to set LM loss weight for antibodies.

    def sample(self, seed: int) -> BinderPromptStr:
        random.seed(seed)
        return self.template.format(
            **{
                key: MUTABLE_TOKEN * random.randint(low, high)
                for key, (low, high) in self.length_ranges.items()
            }
        )


# fmt: off
BINDER_PROMPT_FACTORIES = {
    "minibinder": PromptFactory(name="minibinder", template="{seq}", length_ranges={"seq": (60, 200)}, is_antibody=False),
    "trastuzumab_framework_vhvl": PromptFactory(
        name="trastuzumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}YIHWVRQAPGKGLEWVARI{hcdr2}TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "atezolizumab_framework_vhvl": PromptFactory(
        name="atezolizumab_framework_vhvl",
        template="EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}WIHWVRQAPGKGLEWVAWI{hcdr2}TYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    ),
    "ocankitug_framework_vhvl": PromptFactory(
        name="ocankitug_framework_vhvl",
        template="QVQLVQSGAEVKKPGSSVKVSCKAS{hcdr1}WMHWVRQAPGQGLEWMGII{hcdr2}TSLNQKFQGRVTITADTSTSTAYMELSSLRSEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        length_ranges = {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (8, 14), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
        is_antibody=True,
    )
}


TARGET_SEQUENCES = {
    # https://www.uniprot.org/uniprotkb/P08575  389-574
    "cd45": "GSPGEPQIIFCRSEAAHQGVITWNPPQRSFHNFTLCYIKETEKDCLNLDKNLIKYDLQNLKPYTKYVLSLHAYIIAKVQRNGSAAMCHFTTKSAPPSQVWNMTVSMTSDNSMHVKCRPPRDRNGPHERYHLEVEAGNTLVRNESHKNCDFRVKDLQYSTDYTFKAYFHNGDYPGEPFILHHSTSY",
    # https://www.uniprot.org/uniprotkb/P16410  37-155
    "ctla4": "MHVAQPAVVLASSRGIASFVCEYASPGKATEVRVTVLRQADSQVTEVCAATYMMGNELTFLDDSICTGTSSGNQVNLTIQGLRAMDTGLYICKVELMYPPPYYLGIGNGTQIYVIDPE",
    # https://www.uniprot.org/uniprotkb/P00533  333-524
    "egfr": "RKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCV",
    # https://www.uniprot.org/uniprotkb/Q9NZQ7  17-132
    "pd-l1": "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA",
    # https://www.uniprot.org/uniprotkb/P09619  125-312
    "pdgfr": "GFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYICKTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESGRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE",
    # px172: CXCL8 (IL-8) mature 1-77. https://www.uniprot.org/uniprotkb/P10145
    "cxcl8": "AVLPRSAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS",
}
# fmt: on


# ---- Helper functions ----


def build_initial_soft_sequence_logits(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Initialize logits with:
    - High confidence (10.0) for fixed positions
    - Random (~0) for mutable positions
    - -1e6 for cysteines
    """
    if all(aa == MUTABLE_TOKEN for aa in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, CYS_IDX] = -1e6  # remove cysteines
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for i, aa in enumerate(sequence):
            if aa == MUTABLE_TOKEN:  # mutable position - random
                logits[:, i, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                logits[:, i, CYS_IDX] = -1e6
            else:  # fixed position
                assert aa in PROTEIN_1TO3, aa
                token_id = TOKEN_IDS[PROTEIN_1TO3[aa]]
                logits[:, i, token_id - 2] = 10.0

    return logits.requires_grad_(True)


def build_gradient_mask(sequence: str, batch_size: int) -> torch.Tensor:
    """
    Build gradient mask [B, L, V]:
    - 0 for fixed (all amino acids)
    - 0 for cysteine at all positions
    - 1 for non-cysteine amino acids at mutable positions
    """
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed_positions = [i for i, aa in enumerate(sequence) if aa != MUTABLE_TOKEN]
    mask[:, fixed_positions, :] = 0.0
    mask[:, :, CYS_IDX] = 0.0
    return mask


def sequence_to_one_hot(sequence: str, device="cuda") -> torch.Tensor:
    """Convert target string to one-hot tensor [1, L_target, num_tokens]."""

    const_dict = {token: i for i, token in enumerate(TOKENS)}
    target_index = [const_dict[PROTEIN_1TO3[letter]] for letter in sequence]
    one_hot = F.one_hot(torch.tensor(target_index), num_classes=len(TOKENS))
    return one_hot.to(device).unsqueeze(0).float()


def get_mid_points() -> torch.Tensor:
    """128 distance bin midpoints (2p-52 Angstrom range)."""

    boundaries = torch.linspace(2, 52.0, 127)
    lower = torch.tensor([1.0])
    upper = torch.tensor([52.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    return (exp_boundaries[:-1] + exp_boundaries[1:]) / 2


def binned_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Entropy of distance distribution within cutoff (design losses only)."""

    bin_mask = ~(bin_distance < cutoff)
    masked_dgram = dgram - (1e7 * bin_mask)
    px = torch.softmax(masked_dgram, dim=-1)
    log_px = torch.log_softmax(dgram, dim=-1)
    return -(px * log_px).sum(-1)


def masked_min_k(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Mean of the smallest k values in x under mask along the last dimension."""

    mask = mask.bool()
    y = torch.sort(torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean along last axis."""

    mask = mask.bool()
    return torch.where(mask, x, 0).sum(-1) / (torch.where(mask, 1, 0).sum(-1) + 1e-8)


# ---- Loss functions ----


def compute_contact_loss(
    distogram_logits: torch.Tensor,
    bin_distance: torch.Tensor,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: torch.Tensor,
    binder_mask: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 12 Contact Losses.

    Entropy-based contact loss with sequence separation constraint."""

    con_loss = binned_entropy(distogram_logits, bin_distance, cutoff)
    position = torch.arange(distogram_logits.shape[1])
    p_dist = position[:, None] - position[None, :]
    if min_sep > 0:
        separation_mask = (torch.abs(p_dist) >= min_sep).to(distogram_logits.device)
        binder_mask = torch.logical_and(separation_mask, binder_mask)
    per_residue = masked_min_k(con_loss, mask=binder_mask, k=num_contacts).to(
        distogram_logits.device
    )
    return masked_average(per_residue, mask=chain_mask).to(distogram_logits.device)


def compute_intra_contact_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Binder internal contacts (k=2, min_sep=9, cutoff=14A)."""

    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0
    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=2,
        min_sep=9,
        cutoff=14.0,
        chain_mask=is_binder,
        binder_mask=is_binder,
    )


def validate_epitope_indices(epitope_indices: list[int], target_length: int) -> None:
    """Raise if epitope_indices cannot select any residue on the target.

    An out-of-range index would otherwise be silently dropped from the mask, and
    an all-empty mask makes masked_average return 0.0 -- i.e. the interface loss
    vanishes and the binder is optimized with no binding pressure at all, with
    nothing in the logs to indicate it.
    """
    if not epitope_indices:
        raise ValueError(
            "epitope_indices is empty; omit it entirely to target the whole target chain."
        )
    invalid = sorted({i for i in epitope_indices if not 0 <= i < target_length})
    if invalid:
        raise ValueError(
            f"epitope_indices {invalid} are out of range for a target of length "
            f"{target_length}. Indices are 0-based positions on the target "
            f"sequence, so the valid range is 0..{target_length - 1}."
        )


def compute_inter_contact_loss(
    distogram_logits: torch.Tensor,
    binder_length: int,
    bin_distance: torch.Tensor,
    epitope_indices: list[int] | None = None,
) -> torch.Tensor:
    """Binder-target interface (k=1, min_sep=0, cutoff=22A).

    The target occupies the prefix [0, target_length) of the complex, so
    epitope_indices index it directly. When given, the loss is averaged over only
    those target residues instead of the whole target chain, which focuses the
    interface on the epitope.
    """

    full_len = distogram_logits.shape[1]
    is_binder = torch.ones(full_len, device=distogram_logits.device)
    is_binder[:-binder_length] *= 0.0

    if epitope_indices is None:
        chain_mask = 1 - is_binder
    else:
        validate_epitope_indices(epitope_indices, full_len - binder_length)
        chain_mask = torch.zeros(full_len, device=distogram_logits.device)
        chain_mask[list(epitope_indices)] = 1.0

    return compute_contact_loss(
        distogram_logits,
        bin_distance,
        num_contacts=1,
        min_sep=0,
        cutoff=22.0,
        chain_mask=chain_mask,
        binder_mask=is_binder,
    )


def compute_globularity_loss(
    distogram_logits: torch.Tensor, binder_length: int, bin_distance: torch.Tensor
) -> torch.Tensor:
    """Algorithm 13 Globularity Loss.

    Radius of gyration vs theoretical packed protein."""

    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    n = binder_disto.shape[1]
    disto_probs = torch.softmax(binder_disto, dim=-1)
    bin_distance = bin_distance.clamp(max=27)
    e_sq_dist = torch.sum(disto_probs * torch.square(bin_distance), dim=-1)
    sum_sq_dist = torch.sum(torch.tril(e_sq_dist, diagonal=-1), dim=(1, 2))
    rg_term = torch.sqrt(sum_sq_dist / (n * n))
    rg_th = 2.38 * (n**0.365)
    return F.elu(rg_term - rg_th)


def compute_structure_losses(
    distogram_logits: torch.Tensor,
    binder_length: int,
    epitope_indices: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute structural losses and a weighted total."""

    bin_distance = get_mid_points().to(distogram_logits.device)
    losses: dict[str, torch.Tensor] = {}
    losses["intra_contact_loss"] = compute_intra_contact_loss(
        distogram_logits, binder_length, bin_distance
    )
    losses["inter_contact_loss"] = compute_inter_contact_loss(
        distogram_logits, binder_length, bin_distance, epitope_indices=epitope_indices
    )
    losses["glob_loss"] = compute_globularity_loss(
        distogram_logits, binder_length, bin_distance
    )
    B = distogram_logits.size(0)
    total = torch.tensor([0.0] * B, device=distogram_logits.device, requires_grad=True)
    total = total + LOSS_WEIGHTS["intra_contact"] * losses["intra_contact_loss"]
    total = total + LOSS_WEIGHTS["inter_contact"] * losses["inter_contact_loss"]
    total = total + LOSS_WEIGHTS["glob"] * losses["glob_loss"]
    losses["total_loss"] = total
    return losses


# ---- Distogram iptm proxy ----


def _binding_confidence_entropy(
    dgram: torch.Tensor, bin_distance: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Pair entropy within cutoff."""

    probs = torch.softmax(dgram, dim=-1)
    cutoff_mask = bin_distance < cutoff
    p_cut = probs[..., cutoff_mask]
    p_cut = p_cut / (p_cut.sum(-1, keepdim=True) + 1e-8)
    return -(p_cut * torch.log(p_cut + 1e-10)).sum(-1)


def _entropy_to_confidence(mean_entropy: float) -> float:
    """Map mean pair entropy to [0, 1]; lower entropy → higher score."""
    return float(max(0.0, min(1.0, 1.0 - mean_entropy / math.log(51))))


def _is_antibody_sequence(binder_sequence: str) -> bool:
    """Return True if ANARCI recognizes binder_sequence as antibody variable domain(s)."""
    from abnumber.common import _anarci_align

    sequence = binder_sequence.replace(MUTABLE_TOKEN, "A")
    result = _anarci_align(
        sequences=[sequence], scheme="chothia", allowed_species=None
    )[0]
    if not result:
        return False
    valid_chain_types = {"H", "K", "L"}
    return all(chain_type in valid_chain_types for _, chain_type, *_ in result)


def _cdr_indices(binder_sequence: str, cdr_definition: str = CDR_DEFINITION) -> list[int]:
    """0-based binder indices for all CDRs under cdr_definition.

    abnumber separates the numbering scheme from the CDR definition. Only the
    definition changes which residues come back: measured on this fork's IL8 scFv,
    all four schemes give byte-identical CDRs for a given definition, so the scheme
    stays fixed at chothia and only the definition is configurable.
    """
    from abnumber import Chain
    from abnumber.common import SUPPORTED_CDR_DEFINITIONS, _anarci_align

    if cdr_definition not in SUPPORTED_CDR_DEFINITIONS:
        raise ValueError(
            f"cdr_definition {cdr_definition!r} is not supported by abnumber; "
            f"choose one of {sorted(SUPPORTED_CDR_DEFINITIONS)}."
        )

    result = _anarci_align(
        sequences=[binder_sequence], scheme="chothia", allowed_species=None
    )[0]
    chains = [
        Chain(
            "".join(result[i][0].values()),
            scheme="chothia",
            cdr_definition=cdr_definition,
        )
        for i in range(len(result))
    ]
    if len(chains) == 2 and not chains[0].is_heavy_chain():
        chains.reverse()
    indices: list[int] = []
    for chain in chains:
        for cdr in (chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq):
            start = binder_sequence.find(cdr)
            assert start >= 0
            indices.extend(range(start, start + len(cdr)))
    return indices


def compute_distogram_iptm_proxy(
    distogram_logits: torch.Tensor,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool,
    cdr_definition: str = CDR_DEFINITION,
) -> dict[str, float]:
    """Algorithm 15 Distogram ipTM Proxy.

    Distogram iptm proxy for a target|binder complex (binder at suffix).

    Returns distogram_iptm_proxy for all designs and
    cdr_distogram_iptm_proxy when the binder can be annotated as an
    antibody; otherwise the CDR score is NaN. cdr_definition selects which
    residues count as CDR (see _cdr_indices); it does not affect any other score.
    """
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]

    binder_length = len(binder_sequence)
    assert distogram_logits.shape[0] == target_length + binder_length

    bin_distance = get_mid_points().to(distogram_logits.device)
    binder_start = target_length

    def _mean_lowest_k(entropies: torch.Tensor, k: int) -> float:
        sorted_entropies, _ = torch.sort(entropies.reshape(-1))
        k = min(k, sorted_entropies.numel())
        return float(sorted_entropies[:k].mean())

    binder_to_target_entropy = _binding_confidence_entropy(
        distogram_logits[binder_start:, :target_length, :], bin_distance, cutoff=22.0
    )
    distogram_iptm_proxy = _entropy_to_confidence(
        _mean_lowest_k(binder_to_target_entropy, k=binder_length)
    )

    if not is_antibody:
        cdr_distogram_iptm_proxy = float("nan")
    else:
        cdr_indices = _cdr_indices(binder_sequence, cdr_definition=cdr_definition)
        cdr_rows = [binder_start + i for i in cdr_indices]
        cdr_to_target_entropy = _binding_confidence_entropy(
            distogram_logits[cdr_rows, :target_length, :], bin_distance, cutoff=22.0
        )
        cdr_distogram_iptm_proxy = _entropy_to_confidence(
            _mean_lowest_k(cdr_to_target_entropy, k=len(cdr_indices))
        )

    return {
        "distogram_iptm_proxy": distogram_iptm_proxy,
        "cdr_distogram_iptm_proxy": cdr_distogram_iptm_proxy,
    }


def compute_interface_contacts(
    inputs: dict[str, torch.Tensor],
    output: dict[str, Any],
    target_length: int,
    epitope_indices: list[int] | None,
    cutoff: float = INTERFACE_CONTACT_CUTOFF,
) -> dict[str, Any]:
    """Which target residues the binder actually touches in the predicted structure.

    This is measured on real 3D coordinates, unlike distogram_iptm_proxy / iptm,
    which score how confidently the binder binds *anywhere* on the target and so
    cannot distinguish an on-epitope design from an off-epitope one.

    Returns the contacted target residues (0-based), and -- when epitope_indices is
    given -- how many of them were hit and how close the binder got. epitope_contacts
    is 0 for a design that binds a different site entirely; that is a real outcome to
    filter or group on, not necessarily a failure.
    """
    coords = output["sample_atom_coords"][0]
    atom_to_token = inputs["atom_to_token"][0]
    resolved = inputs["atom_attention_mask"][0].bool()
    # ELEMENT_NUMBER_TO_SYMBOL keys are atomic numbers; 1 is hydrogen.
    heavy = resolved & (inputs["ref_element"][0] != 1)

    is_target = heavy & (atom_to_token < target_length)
    is_binder = heavy & (atom_to_token >= target_length)
    if not bool(is_target.any()) or not bool(is_binder.any()):
        raise ValueError("Could not split the complex into target and binder atoms.")

    target_atoms = coords[is_target].float()
    binder_atoms = coords[is_binder].float()
    tokens = atom_to_token[is_target].long()

    # Closest binder atom for each target atom, reduced to a per-residue minimum.
    atom_min = torch.cdist(target_atoms, binder_atoms).min(dim=1).values
    per_residue = torch.full((target_length,), float("inf"), device=atom_min.device)
    per_residue.scatter_reduce_(0, tokens, atom_min, reduce="amin", include_self=True)

    interface = [i for i in range(target_length) if per_residue[i].item() < cutoff]
    result: dict[str, Any] = {
        "interface_residues": interface,
        "interface_size": len(interface),
    }
    if epitope_indices is None:
        result["epitope_contacts"] = float("nan")
        result["epitope_min_dist"] = float("nan")
    else:
        epitope = set(epitope_indices)
        result["epitope_contacts"] = len(epitope & set(interface))
        result["epitope_min_dist"] = min(per_residue[i].item() for i in epitope)
    return result


# ---- Folding ----


def _resize_tensor(tensor: torch.Tensor, *, dim: int, size: int) -> torch.Tensor:
    current = tensor.shape[dim]
    if current >= size:
        return tensor.narrow(dim, 0, size)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": 0,
    "atom_to_token": 0,
    "is_resolved": 0,
    "gt_coords": 1,
}


@cache
def _ensure_ccd_loaded() -> None:
    load_ccd()


def prepare_esmfold2_tensors(
    input: StructurePredictionInput,
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    max_seqs: int = 16384,
    pad_to_max_seqs: bool = False,
    seed: int | None = None,
    use_vectorized_msa_assembly: bool = True,
) -> dict[str, torch.Tensor]:
    del max_tokens, max_seqs, pad_to_max_seqs, use_vectorized_msa_assembly
    _ensure_ccd_loaded()
    features, _ = prepare_esmfold2_input(input, seed=seed)
    if max_atoms is not None:
        for key, dim in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dim, size=max_atoms)
    return features


def fold_and_get_distogram(
    model: ESMFold2ExperimentalModel,
    target_seq: str,
    target_one_hot: torch.Tensor,
    design: torch.Tensor,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict:
    """Prepare inputs, run model forward, return distogram_logits + raw output."""
    padding = (2, 11)
    padded_design = F.pad(design, padding, mode="constant", value=0)

    # Argmax to get the designed sequence string.
    token_lists = torch.argmax(padded_design, dim=-1)
    designed_seq = [
        [PROTEIN_3TO1[TOKENS[int(tkn.item())]] for tkn in token_list]
        for token_list in token_lists
    ]
    seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]
    max_atoms = None if len(seq_list) == 1 else ((len(seq_list[0]) - 1) * 14) // 32 * 32

    inputs_list = []
    for seq in seq_list:
        sequences = {
            sequence: [str(idx)] for idx, sequence in enumerate(seq.split("|"))
        }
        inputs_raw = StructurePredictionInput(
            sequences=[
                ProteinInput(id=chain_id, sequence=sequence, msa=None)
                for sequence, chain_id in sequences.items()
            ]
        )
        inputs_list.append(prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms))

    inputs = {
        key: torch.stack([inp[key] for inp in inputs_list], dim=0).cuda()
        for key in inputs_list[0]
    }
    inputs["res_type_soft"] = torch.cat(
        (target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1
    )

    with seed_context(seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )

    result: dict = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": inputs_list,
        "output": output,
        "seq_list": seq_list,
    }
    if calculate_confidence:
        result.update(
            {
                "ptm": output.get("ptm"),
                "iptm": output.get("iptm"),
                "plddt": output.get("plddt"),
            }
        )
    return result


_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _asym_id_to_chain_label(asym_id: int) -> str:
    if asym_id < 0:
        raise ValueError(f"asym_id must be >= 0, got {asym_id}")
    label = ""
    n = len(_CHAIN_ID_ALPHABET)
    while True:
        label = _CHAIN_ID_ALPHABET[asym_id % n] + label
        asym_id = asym_id // n - 1
        if asym_id < 0:
            return label


def to_atom_array(
    coords: np.ndarray,
    atom_to_token: np.ndarray,
    res_type: np.ndarray,
    residue_index: np.ndarray,
    asym_id: np.ndarray,
    mol_type: np.ndarray,
    ref_atom_name_chars: np.ndarray,
    ref_element: np.ndarray,
    atom_attention_mask: np.ndarray,
    plddt_per_atom: np.ndarray | None = None,
) -> biotite.structure.AtomArray:
    atoms = []
    for atom_i, (
        atom_coord,
        token_idx,
        atom_name_chars,
        element_idx,
        is_not_pad,
    ) in enumerate(
        zip(
            coords, atom_to_token, ref_atom_name_chars, ref_element, atom_attention_mask
        )
    ):
        if not is_not_pad:
            continue
        atoms.append(
            biotite.structure.Atom(
                coord=atom_coord,
                chain_id=_asym_id_to_chain_label(int(asym_id[token_idx])),
                res_id=residue_index[token_idx] + 1,
                res_name=TOKENS[res_type[token_idx]],
                atom_name="".join(chr(c + 32) for c in atom_name_chars if c != 0),
                element=ELEMENTS[element_idx],
                ins_code=" ",
                hetero=mol_type[token_idx] == MOL_TYPE_NONPOLYMER,
                b_factor=float(plddt_per_atom[atom_i])
                if plddt_per_atom is not None
                else 0.0,
            )
        )
    return biotite.structure.array(atoms)


def build_complex(
    inputs: dict[str, torch.Tensor], output: dict[str, Any]
) -> ProteinComplex:
    """Build ProteinComplex from model output."""
    plddt_per_atom = output.get("plddt_per_atom")
    if plddt_per_atom is not None:
        plddt_per_atom = plddt_per_atom[0].cpu().numpy() * PLDDT_B_FACTOR_SCALE
    atom_arr = to_atom_array(
        coords=output["sample_atom_coords"][0].cpu().numpy(),
        atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
        res_type=inputs["res_type"][0].cpu().numpy(),
        residue_index=inputs["token_index"][0].cpu().numpy(),
        asym_id=inputs["asym_id"][0].cpu().numpy(),
        mol_type=inputs["mol_type"][0].cpu().numpy(),
        ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
        ref_element=inputs["ref_element"][0].cpu().numpy(),
        atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
        plddt_per_atom=plddt_per_atom,
    )
    return ProteinComplex.from_chains(
        [
            ProteinChain.from_atomarray(a, is_predicted=True)
            for a in biotite.structure.chain_iter(atom_arr)
        ]
    )


# ---- LM loss ----


@cache
def _folding_trunk_to_lm_aa_vocab_matrix(device: torch.device) -> torch.Tensor:
    """Build a matrix of shape [ft_aas=20, lm_aas=20]."""
    three_to_one_map = {v: k for k, v in PROTEIN_1TO3.items()}
    ft_aas = [three_to_one_map[tok_3letter] for tok_3letter in TOKENS[2:22]]

    lm_vocab = sorted(ESMCTokenizer().vocab.items(), key=lambda x: x[1])
    lm_aas = [lm_vocab[i][0] for i in range(4, 24)]

    ft_to_lm_aa_matrix = torch.zeros(20, 20)
    for ft_idx, ft_aa in enumerate(ft_aas):
        lm_idx = lm_aas.index(ft_aa)
        ft_to_lm_aa_matrix[ft_idx, lm_idx] = 1

    return ft_to_lm_aa_matrix.to(device=device)


def _one_hot_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.size(-1)).to(
        probs.dtype
    )


def _straight_through(discrete: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
    return continuous + (discrete - continuous).detach()


def compute_esmc_pseudoperplexity_nll(
    esmc_model: ESMCForMaskedLM,
    binder_design: torch.Tensor,
    score_mask: torch.Tensor,
    batch_size: int = 4,
    n_passes: int = 4,
) -> torch.Tensor:
    """Algorithm 14 ESMC Pseudo-perplexity Sequence Regularization.

    Approximate pseudoperplexity NLL via multiple sampled masks."""
    device = binder_design.device
    lm_vocab_size = esmc_model.config.vocab_size
    model_dtype = esmc_model.esmc.embed.weight.dtype

    target_esm = binder_design @ _folding_trunk_to_lm_aa_vocab_matrix(device)
    input_esm = _straight_through(_one_hot_from_probs(target_esm), target_esm)
    input_ids = torch.zeros(
        (binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size),
        dtype=model_dtype,
        device=device,
    )
    tokenizer = ESMCTokenizer()
    input_ids[:, 0, tokenizer.cls_token_id] = 1  # pyright: ignore
    input_ids[:, -1, tokenizer.eos_token_id] = 1  # pyright: ignore
    input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    elif score_mask.shape != binder_design.shape[:2]:
        raise ValueError(
            f"Expected score_mask with shape {(binder_design.size(0), binder_design.size(1))}, "
            f"got {tuple(score_mask.shape)}"
        )
    score_mask = score_mask.to(device=device, dtype=torch.bool)

    mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
    mask_token[esmc_model.config.mask_token_id] = 1
    esmc = esmc_model.esmc

    all_masked_sequences = []
    all_pass_masks = []
    for batch_idx in range(binder_design.size(0)):
        position_indices = score_mask[batch_idx].nonzero(as_tuple=False).flatten()
        num_positions = int(position_indices.numel())
        if num_positions == 0:
            raise ValueError(
                "ESMC pseudoperplexity score mask selected zero positions."
            )

        num_masked = max(1, math.ceil(ESMC_MASK_FRACTION * num_positions))
        random_scores = torch.rand((n_passes, num_positions), device=device)
        masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
        pass_masks = torch.zeros(
            (n_passes, binder_design.size(1)), dtype=torch.bool, device=device
        )
        pass_masks[
            torch.arange(n_passes, device=device)[:, None],
            position_indices[masked_offsets],
        ] = True

        masked_sequences = input_ids[batch_idx : batch_idx + 1].repeat(n_passes, 1, 1)
        mask_rows, mask_cols = pass_masks.nonzero(as_tuple=True)
        masked_sequences[mask_rows, mask_cols + 1] = mask_token

        all_masked_sequences.append(masked_sequences)
        all_pass_masks.append(pass_masks)

    logit_chunks = []
    masked_sequence_rows = torch.cat(all_masked_sequences, dim=0)
    for start in range(0, masked_sequence_rows.size(0), batch_size):
        chunk = masked_sequence_rows[start : start + batch_size]
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden, *_ = esmc.transformer(
                chunk @ esmc.embed.weight.to(chunk.dtype),
                sequence_id=None,
                layers_to_collect=[],
                output_attentions=False,
            )
            logit_chunks.append(esmc_model.lm_head(hidden))
    logits = torch.cat(logit_chunks, dim=0)

    losses = []
    for batch_idx, pass_masks in enumerate(all_pass_masks):
        start = batch_idx * n_passes
        stop = start + n_passes
        target_weights = target_esm[batch_idx]
        log_probs = logits[start:stop].log_softmax(dim=-1)[:, 1:-1, 4:24]
        nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(
            dim=-1
        )
        losses.append(nlls[pass_masks].mean())

    return torch.stack(losses, dim=0)


# ---- Design ----


def normalized_gradient_tensor(
    grad: torch.Tensor, gradient_mask: torch.Tensor
) -> torch.Tensor:
    masked_grad = grad * gradient_mask
    index_has_nonzero_grad = torch.square(masked_grad).sum(-1) > 0  # (B, L)
    eff_L = index_has_nonzero_grad.sum(-1)  # (B,)
    grad_norm = torch.linalg.norm(masked_grad, axis=(-1, -2))  # (B,)
    normalized_grad = (masked_grad / (grad_norm[:, None, None] + 1e-7)) * torch.sqrt(
        eff_L[:, None, None]
    )
    return normalized_grad * gradient_mask


def design_binder(
    inversion_models: dict[str, ESMFold2ExperimentalModel],
    hf_critic_models: dict[str, ESMFold2ExperimentalModel],
    esmc_model: ESMCForMaskedLM,
    target_name: str | None,
    target_sequence: str | None,
    binder_name: str | None,
    binder_sequence: str | None,
    is_antibody: bool | None,
    seed: int,
    batch_size: int = 1,
    epitope_indices: list[int] | None = None,
    cdr_definition: str = CDR_DEFINITION,
) -> tuple[list[str], Trajectory, list[dict]]:
    """
    Algorithm 11 Gradient-Guided Binder Sequence Optimization.

    Run the full optimization loop.
    Returns dict with designed_sequence, complex, and trajectory.

    Every critic is folded once on the best designed sequence via HF ESMFold2.
    Hero critics expose iPTM; scaling critics contribute distogram scores only.
    ``distogram_binding_confidence`` / ``cdr_distogram_binding_confidence`` come
    from the distogram in all cases.
    """
    # Setup
    device = "cuda"
    if target_name in TARGET_SEQUENCES:
        if target_sequence is not None:
            raise ValueError(
                f"{target_name!r} is a preset target; omit target_sequence."
            )
        target_sequence = TARGET_SEQUENCES[target_name]
    elif target_sequence is None:
        raise ValueError(
            f"{target_name!r} is not a preset target; provide target_sequence."
        )
    target_one_hot = sequence_to_one_hot(target_sequence, device=device)

    if binder_name in BINDER_PROMPT_FACTORIES:
        if binder_sequence is not None:
            raise ValueError(
                f"{binder_name!r} is a preset binder; omit binder_sequence."
            )
        binder_prompt_factor = BINDER_PROMPT_FACTORIES[binder_name]
        if is_antibody is not None:
            assert (
                binder_prompt_factor.is_antibody == is_antibody
            ), "Conflict in is_antibody settings."
        is_antibody = binder_prompt_factor.is_antibody
        binder_sequence = binder_prompt_factor.sample(seed=seed)
    elif binder_sequence is None:
        raise ValueError(
            f"{binder_name!r} is not a preset binder; provide binder_sequence."
        )
    elif is_antibody is None:
        is_antibody = _is_antibody_sequence(binder_sequence)

    binder_length = len(binder_sequence)

    # By default, we only support single binder and target chains.
    # To support this case, remove the asserts below and check that losses
    # and selection metrics are appropriate for your multi-chain case.
    assert "|" not in target_sequence
    assert "|" not in binder_sequence

    if is_antibody:
        if cdr_definition not in CDR_DEFINITIONS:
            raise ValueError(
                f"cdr_definition {cdr_definition!r} is not one of {list(CDR_DEFINITIONS)}."
            )
        logger.info(f"Scoring CDRs with the {cdr_definition!r} definition.")

    # Fail before loading anything expensive rather than at the first step.
    if epitope_indices is not None:
        validate_epitope_indices(epitope_indices, len(target_sequence))
        epitope_residues = "".join(target_sequence[i] for i in sorted(epitope_indices))
        logger.info(
            f"Restricting the interface loss to {len(set(epitope_indices))} epitope "
            f"residues (0-based {sorted(set(epitope_indices))}): {epitope_residues}"
        )

    with seed_context(seed), torch.device(device):
        logits = build_initial_soft_sequence_logits(
            binder_sequence, batch_size=batch_size
        )
        gradient_mask = build_gradient_mask(binder_sequence, batch_size=batch_size)

    # step -> {loss_name: [B] tensor on CPU, metric_name: float}
    trajectory: Trajectory = {}
    global_step = 0

    def run_step(
        logits: torch.Tensor,
        optimizer: optim.Optimizer,
        temperature: float,
        calculate_confidence: bool,
    ) -> tuple[torch.Tensor, list[str], list[float] | None]:
        nonlocal global_step
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        optimizer.zero_grad()

        random.seed(seed + global_step)
        replicate_choice = random.randint(0, len(inversion_models) - 1)
        inversion_model = list(inversion_models.values())[replicate_choice]
        design = F.softmax(logits / temperature, dim=-1)

        fold_result = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            num_loops=1,
            num_sampling_steps=50 if calculate_confidence else 1,
            calculate_confidence=calculate_confidence,
            seed=seed + global_step,
        )
        sequences: list[str] = fold_result["seq_list"]
        losses = compute_structure_losses(
            fold_result["distogram_logits"],
            binder_length,
            epitope_indices=epitope_indices,
        )
        structure_loss = losses["total_loss"]
        structure_grad = torch.autograd.grad(structure_loss.mean(), logits)[0]

        # Recompute the logits -> design transform for a fresh graph.
        design = F.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with seed_context(seed + global_step):
            plm_loss = compute_esmc_pseudoperplexity_nll(
                esmc_model=esmc_model,
                binder_design=design,
                score_mask=score_mask,
                batch_size=LM_LOSS_BATCH_SIZE,
                n_passes=LM_MASK_PASSES,
            )
        plm_grad = torch.autograd.grad(plm_loss.mean(), logits)[0]

        logits.grad = normalized_gradient_tensor(structure_grad, gradient_mask) + (
            0.05 if is_antibody else 0.15
        ) * normalized_gradient_tensor(plm_grad, gradient_mask)

        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE * temperature

        optimizer.step()

        step = global_step
        step_losses: TrajectoryStep = {k: v.detach().cpu() for k, v in losses.items()}
        step_losses["plm_loss"] = plm_loss.detach().cpu()
        step_losses["total_loss"] = (structure_loss + plm_loss).detach().cpu()
        step_losses["time"] = time.time() - start
        step_losses["peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / BYTES_PER_GIB
        )
        step_losses["peak_reserved_gib"] = (
            torch.cuda.max_memory_reserved() / BYTES_PER_GIB
        )
        trajectory[step] = step_losses
        loss_str = "  ".join(
            f"{k}={v.mean().item() if torch.is_tensor(v) else v:.4f}"
            for k, v in step_losses.items()
        )
        if step % LOG_INTERVAL == 0:
            logger.info(f"  step {step:3d}  |  {loss_str}  T={temperature:.4f}")
        global_step += 1
        return logits, sequences, fold_result.get("iptm", None)

    # Optimize
    optimizer = optim.SGD([logits], lr=LEARNING_RATE)
    best_iptm: list[float] = [-1.0] * batch_size
    best_sequences: list[str] = [""] * batch_size
    for step in range(STEPS):
        # Cosine schedule
        t = (step + 1) / STEPS
        remaining = 0.5 * (1 + math.cos(math.pi * t))
        temperature = TEMPERATURE_MIN + (1 - TEMPERATURE_MIN) * remaining
        logits, sequences, iptm = run_step(
            logits,
            optimizer,
            temperature=temperature,
            calculate_confidence=temperature < 0.05,
        )
        if iptm is not None:
            for b in range(batch_size):
                if iptm[b] is not None and iptm[b] > best_iptm[b]:
                    best_iptm[b] = iptm[b]
                    best_sequences[b] = sequences[b]
    assert all(seq != "" for seq in best_sequences)

    # Score
    critic_results: list[dict] = []
    target_length = len(target_sequence.replace("|", ""))
    final_total_loss = trajectory[global_step - 1]["total_loss"]
    assert isinstance(final_total_loss, torch.Tensor)
    for batch_idx in range(batch_size):
        best_seq = best_sequences[batch_idx]
        binder_seq = best_seq.split("|")[-1]
        binder_design = sequence_to_one_hot(binder_seq)[..., 2:22]
        for critic_name, critic_model in hf_critic_models.items():
            is_scaling_critic = "ESMFold2-Experimental-Fast-base" in critic_name
            if is_scaling_critic:
                critic_model.cuda()
            final_fold = fold_and_get_distogram(
                critic_model,
                target_sequence,
                target_one_hot,
                binder_design,
                num_loops=3,
                num_sampling_steps=200,
                calculate_confidence=True,
                seed=seed,
            )
            if is_scaling_critic:
                critic_model.cpu()
            pred_complex = build_complex(final_fold["inputs"], final_fold["output"])
            contacts = compute_interface_contacts(
                final_fold["inputs"],
                final_fold["output"],
                target_length,
                epitope_indices,
            )
            iptm_proxy_scores = compute_distogram_iptm_proxy(
                final_fold["distogram_logits"],
                target_length,
                binder_seq,
                is_antibody,
                cdr_definition=cdr_definition,
            )
            iptm = final_fold["iptm"].item() if final_fold["iptm"] is not None else None
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "cdr_definition": cdr_definition if is_antibody else "",
                    "critic_name": critic_name,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_seq,
                    "complex": pred_complex,
                    "final_loss": final_total_loss[batch_idx].item(),
                    "iptm": iptm,
                    "logits": logits[batch_idx].detach().cpu(),
                    **iptm_proxy_scores,
                    **contacts,
                }
            )

    if not critic_results:
        for batch_idx in range(batch_size):
            critic_results.append(
                {
                    "is_antibody": is_antibody,
                    "batch_idx": batch_idx,
                    "designed_sequence": best_sequences[batch_idx],
                    "final_loss": final_total_loss[batch_idx].item(),
                    "logits": logits[batch_idx].detach().cpu(),
                }
            )

    return best_sequences, trajectory, critic_results


# ---- Model Loading ----

_ESMC_CACHE: dict[str, torch.nn.Module] = {}


def _load_hf_model(
    critic_name: str, lm_dropout: float, cache_esmc: bool, device: str
) -> Any:
    """Loads ESMFold2 from huggingface. Will cache ESMC by checkpoint ID among
    all non-scaling checkpoints, to save on VRAM and load time."""
    repo_id = f"biohub/{critic_name}"
    model = ESMFold2ExperimentalModel.from_pretrained(repo_id, load_esmc=not cache_esmc)
    if cache_esmc:
        esmc_id = model.config.esmc_id
        if esmc_id not in _ESMC_CACHE:
            model.load_esmc(esmc_id)
            assert model._esmc is not None
            _ESMC_CACHE[esmc_id] = model._esmc
        model._esmc = _ESMC_CACHE[esmc_id]
    model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
    model.set_kernel_backend("cuequivariance" if CUE_AVAILABLE else None)
    return model.to(device=device).eval().requires_grad_(False)


def _apply_torch_compile(model: torch.nn.Module) -> None:
    """A helper for torch compiling the model."""
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 512

    compile_targets = (ESMFold2MSAEncoder, PairUpdateBlock)

    def _maybe_compile_module(module: torch.nn.Module) -> None:
        if not isinstance(module, compile_targets):
            return
        module.forward = torch.compile(module.forward)  # ty:ignore[invalid-assignment]

    model.apply(_maybe_compile_module)


class ESMFold2Design:
    lm_name = "biohub/ESMC-6B"
    inversion_model_names: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
    ]
    hero_critic_hf_paths: list[str] = [
        "ESMFold2-Experimental-Fast",
        "ESMFold2-Experimental-Fast-Cutoff2025",
        "ESMFold2-Experimental",
        "ESMFold2-Experimental-Cutoff2025",
    ]
    scaling_critic_hf_paths: list[str] = []

    def load(self, use_scaling_critics: bool):
        if use_scaling_critics:
            self.scaling_critic_hf_paths = [
                f"ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            ]

        self.inversion_models = {
            model_name: _load_hf_model(
                model_name, lm_dropout=0.5, cache_esmc=True, device="cuda"
            )
            for model_name in self.inversion_model_names
        }
        if COMPILE:
            for model in self.inversion_models.values():
                _apply_torch_compile(model)

        self.hf_critic_models: dict[str, Any] = {}
        for name in self.hero_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=True, device="cuda"
            )
        for name in self.scaling_critic_hf_paths:
            self.hf_critic_models[name] = _load_hf_model(
                name, lm_dropout=0.25, cache_esmc=False, device="cpu"
            )

        # px172: `torch_dtype` still works but is deprecated in transformers >= 4.57.
        self.esmc_model = ESMCForMaskedLM.from_pretrained(
            self.lm_name, dtype=torch.float32
        )
        if REUSE_ESMC:
            reusable_esmc_model = self.inversion_models["ESMFold2-Experimental-Fast"]
            assert reusable_esmc_model.config.esmc_id == self.lm_name, (
                f"Cannot reuse ESMC trunk from {reusable_esmc_model.config.esmc_id!r} "
                f"with LM head from {self.lm_name!r}."
            )
            assert reusable_esmc_model._esmc is not None
            del self.esmc_model.esmc
            torch.cuda.empty_cache()
            self.esmc_model.esmc = reusable_esmc_model._esmc
        self.esmc_model = self.esmc_model.cuda().eval().requires_grad_(False)

        if CHECKPOINT_LM:
            # NOTE - apply_activation_checkpointing rewrites modules in place. With
            # REUSE_ESMC=True, self.esmc_model.esmc IS the shared trunk held by the
            # inversion models and hero critics, so this checkpoints their ESMC
            # forward too, not just the LM's. That is extra recompute for them, but
            # it also cuts their activation memory, and NO_REENTRANT preserves RNG
            # state so the forced LM dropout still reproduces exactly.
            apply_activation_checkpointing(
                self.esmc_model,
                checkpoint_wrapper_fn=partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                check_fn=lambda module: isinstance(module, TransformerBlock),
            )

    def design(
        self,
        target_name: str | None = None,
        binder_name: str | None = None,
        target_sequence: str | None = None,
        binder_sequence: str | None = None,
        is_antibody: bool | None = None,
        seed: int = 0,
        batch_size: int = 1,
        epitope_indices: list[int] | None = None,
        cdr_definition: str = CDR_DEFINITION,
    ) -> tuple[list[str], Trajectory, list[dict]]:
        return design_binder(
            self.inversion_models,
            self.hf_critic_models,
            self.esmc_model,
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=seed,
            batch_size=batch_size,
            epitope_indices=epitope_indices,
            cdr_definition=cdr_definition,
        )


# ---- Modal ----


def get_base_image():
    return (
        modal.Image.micromamba(python_version="3.12")
        .run_commands("apt update && apt install -y git build-essential")
        .micromamba_install(
            "anarci>=2020.04.03",
            "hmmer=3.4",
            "cuda-version=12.8",
            "cuda-libraries-dev=12.8",
            "cuda-nvcc=12.8",
            "cmake",
            "ninja",
            channels=["conda-forge", "bioconda"],
        )
        .pip_install(
            "torch==2.8.0",
            "triton==3.4.0",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "flash-attn==2.8.3",
            "transformer-engine[core-cu12,pytorch]==2.13.0",
            "xformers==0.0.32.post1",
            extra_options="--no-build-isolation",
            env={
                "CPATH": (
                    "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/include:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nccl/include:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nvtx/include"
                ),
                "LIBRARY_PATH": (
                    "/opt/conda/lib/python3.12/site-packages/nvidia/cudnn/lib:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nccl/lib:"
                    "/opt/conda/lib/python3.12/site-packages/nvidia/nvtx/lib"
                ),
                "MAX_JOBS": "8",
                "NVTE_FRAMEWORK": "pytorch",
            },
        )
        .pip_install(
            "abnumber", "esm@git+https://github.com/Biohub/esm.git@main", "modal"
        )
        .env(
            {
                "HF_HOME": "/models",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            }
        )
    )


if modal is not None:
    app = modal.App(
        name="esmfold2-design",
        image=get_base_image(),
        volumes={
            "/models": modal.Volume.from_name("esmfold2-models", create_if_missing=True)
        },
    )

    # NOTE - Currently the memory usage is quite high, inflating costs.
    # In an update coming soon, scaling critics will be loaded on demand
    # to avoid needing this large amount of RAM.
    @app.cls(gpu="H100", timeout=60 * 60, cpu=16, memory=80 * 1024)
    class ESMFold2DesignModal(ESMFold2Design):
        """Modal entrypoint. Hero critics are HF experimental exports with
        confidence heads. Set ``use_scaling_critics=True`` to also load the
        15-checkpoint scaling-experiment ensemble (distogram binding confidence only).
        """

        use_scaling_critics: bool = modal.parameter(default=True)

        @modal.enter()
        def load(self):
            return super().load(self.use_scaling_critics)

        @modal.method()
        def design(self, *args, **kws):
            return super().design(*args, **kws)


# ---- Outputs ----

SCFV_LINKER = "GGGSGGGSGGGSGGGS"
# epitope_contacts / epitope_min_dist come from the predicted 3D structure, so unlike
# iptm and the distogram proxies they tell you *where* the binder landed, not just how
# well it binds. A design with high iptm and epitope_contacts=0 bound a different site.
CSV_METRICS = (
    "iptm",
    "distogram_iptm_proxy",
    "cdr_distogram_iptm_proxy",
    "epitope_contacts",
    "epitope_min_dist",
    "interface_size",
)


def load_sequence(seq_input: str | None) -> str | None:
    """Return a sequence from a raw string, a plain text file, or a FASTA file.

    An existing path is read from disk; anything else is treated as a literal
    sequence. Binder prompts may contain MUTABLE_TOKEN ('#') at designed
    positions.
    """
    if not seq_input:
        return None
    if not os.path.isfile(seq_input):
        return seq_input

    with open(seq_input, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]

    num_records = sum(1 for line in lines if line.startswith(">"))
    if num_records > 1:
        raise ValueError(
            f"{seq_input!r} holds {num_records} FASTA records, but a design targets a "
            "single chain. Split the file and pass one record at a time."
        )
    sequence = "".join(
        line for line in lines if line and not line.startswith(">")
    ).replace(" ", "")
    if not sequence:
        raise ValueError(f"{seq_input!r} contains no sequence.")
    return sequence


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _as_float(value: Any) -> float | None:
    """Return value as a float, or None if it is missing or NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _atomic_write(path: str, write_fn) -> None:
    """Write via a temp file + rename so an interrupted run cannot truncate path."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        write_fn(handle)
    os.replace(tmp_path, path)


def _output_path(filename: str, output_dir: str | None) -> str:
    return os.path.join(output_dir, filename) if output_dir else filename


def save_complexes(results: list[dict], output_dir: str | None) -> None:
    """Write one PDB per (design, critic) pair, recording the path on each result."""
    for result in results:
        if "complex" not in result:
            continue
        pdb_name = f"{result['design_id']}_{_safe_name(result['critic_name'])}.pdb"
        pdb_path = _output_path(pdb_name, output_dir)
        result["complex"].to_pdb(pdb_path)
        result["pdb_path"] = pdb_path
        logger.info(f"Saved complex structure to {pdb_path}")


def write_results_csv(all_results: list[dict], output_dir: str | None) -> str:
    """Write one row per design, with per-critic metrics pivoted into columns."""
    path = _output_path("design_results.csv", output_dir)
    critic_names = list(
        dict.fromkeys(r.get("critic_name", "N/A") for r in all_results)
    )

    groups: dict[str, dict] = {}
    for result in all_results:
        group = groups.setdefault(
            result["design_id"],
            {
                "design_run": result["design_run"],
                "seed": result["run_seed"],
                "batch_idx": result.get("batch_idx", 0),
                # Recorded per row so runs that used different CDR definitions stay
                # comparable after the fact.
                "cdr_definition": result.get("cdr_definition", ""),
                "metrics": {},
            },
        )
        group["metrics"][result.get("critic_name", "N/A")] = result

    def write(handle) -> None:
        writer = csv.writer(handle)
        headers = ["design_id", "design_run", "seed", "batch_idx", "cdr_definition"]
        headers += [f"mean_{metric}" for metric in CSV_METRICS]
        for critic in critic_names:
            headers += [f"{_safe_name(critic)}_{metric}" for metric in CSV_METRICS]
        writer.writerow(headers)

        for design_id, group in groups.items():
            row: list[Any] = [
                design_id,
                group["design_run"],
                group["seed"],
                group["batch_idx"],
                group["cdr_definition"],
            ]
            for metric in CSV_METRICS:
                values = [
                    _as_float(group["metrics"].get(critic, {}).get(metric))
                    for critic in critic_names
                ]
                values = [v for v in values if v is not None]
                row.append(sum(values) / len(values) if values else "N/A")
            for critic in critic_names:
                result = group["metrics"].get(critic, {})
                for metric in CSV_METRICS:
                    number = _as_float(result.get(metric))
                    row.append("" if number is None else number)
            writer.writerow(row)

    _atomic_write(path, write)
    return path


def write_sequences_tsv(all_results: list[dict], output_dir: str | None) -> str:
    """Write one row per design: VH/VL split, plus where the binder actually landed.

    The interface is reported from whichever critic was most confident about the
    complex, since that is the structure worth reading. Designs that missed the
    requested epitope are still listed with the site they did hit, so they can be
    grouped rather than silently discarded.
    """
    path = _output_path("design_sequences.tsv", output_dir)

    best: dict[str, dict] = {}
    for result in all_results:
        current = best.get(result["design_id"])
        if current is None or (_as_float(result.get("iptm")) or -1.0) > (
            _as_float(current.get("iptm")) or -1.0
        ):
            best[result["design_id"]] = result

    def write(handle) -> None:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "design name",
                "VH sequence",
                "VL sequence",
                "antigen sequence",
                "epitope contacts",
                "interface residues (0-based)",
                "interface from critic",
            ]
        )
        for design_id, result in best.items():
            designed = result.get("designed_sequence", "")
            if "|" in designed:
                antigen, binder = designed.split("|", 1)
            else:
                antigen, binder = "", designed
            if SCFV_LINKER in binder:
                vh_seq, vl_seq = binder.split(SCFV_LINKER, 1)
            else:
                vh_seq, vl_seq = binder, "N/A"

            epitope_hits = result.get("epitope_contacts", "")
            if isinstance(epitope_hits, float) and math.isnan(epitope_hits):
                epitope_hits = "N/A"  # no epitope was requested
            interface = ",".join(str(i) for i in result.get("interface_residues", []))
            writer.writerow(
                [
                    design_id,
                    vh_seq,
                    vl_seq,
                    antigen,
                    epitope_hits,
                    interface,
                    result.get("critic_name", ""),
                ]
            )

    _atomic_write(path, write)
    return path


# ---- Entrypoints ----


def run_design(
    target_name: str | None = None,
    binder_name: str | None = None,
    target_sequence: str | None = None,
    binder_sequence: str | None = None,
    use_scaling_critics: bool = False,
    is_antibody: bool | None = None,
    local: bool = False,
    seed: int | None = None,
    batch_size: int = 1,
    epitope_indices: list[int] | None = None,
    cdr_definition: str = CDR_DEFINITION,
    output_dir: str | None = None,
    num_designs: int = 1,
    design_prefix: str | None = None,
) -> list[dict]:
    """Run num_designs design runs, writing PDB/CSV/TSV after each one.

    Outputs are rewritten after every run, so an interrupted campaign keeps
    whatever already finished. The local path never touches Modal, so it works in
    an environment where ``modal`` is not installed.

    target_sequence and binder_sequence may each be a literal sequence or a path
    to a text/FASTA file. seed=None draws a fresh seed per run (recorded in the
    CSV so any design can be reproduced).
    """
    target_sequence = load_sequence(target_sequence)
    binder_sequence = load_sequence(binder_sequence)

    if local:
        assert not use_scaling_critics, (
            "'abnumber' will fail if running this script with uv run. "
            "It requires conda packages.  To be addressed soon."
        )
        design_app = ESMFold2Design()
        design_app.load(use_scaling_critics)
        run_fn = design_app.design
    else:
        if modal is None:
            raise RuntimeError(
                "Remote execution needs the 'modal' package, which is not installed. "
                "Either `pip install modal`, or pass local=True to run in-process."
            )
        design_app = ESMFold2DesignModal(
            use_scaling_critics=use_scaling_critics  # ty:ignore[unknown-argument]
        )
        run_fn = design_app.design.remote

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    prefix = design_prefix or "design"

    all_results: list[dict] = []
    for design_idx in range(num_designs):
        if seed is None:
            current_seed = (int(time.time() * 1000) + design_idx * 13) % 4294967295
        else:
            current_seed = seed + design_idx

        logger.info(
            f"\n--- Starting design run {design_idx + 1}/{num_designs} "
            f"(seed: {current_seed}) ---"
        )
        designed_sequences, trajectory, results = run_fn(
            target_name=target_name,
            target_sequence=target_sequence,
            binder_name=binder_name,
            binder_sequence=binder_sequence,
            is_antibody=is_antibody,
            seed=current_seed,
            batch_size=batch_size,
            epitope_indices=epitope_indices,
            cdr_definition=cdr_definition,
        )

        for result in results:
            result["design_run"] = design_idx + 1
            result["run_seed"] = current_seed
            result["design_id"] = (
                f"{prefix}_b_{result.get('batch_idx', 0) + 1}_d_{design_idx + 1}"
            )
        all_results.extend(results)

        avg_final_loss = sum(r["final_loss"] for r in results) / len(results)
        logger.info(f"Design {design_idx + 1} sequence: {designed_sequences}")
        logger.info(f"Trajectory length: {len(trajectory)} steps")
        logger.info(f"Average final loss: {avg_final_loss:.4f}")

        save_complexes(results, output_dir)
        logger.info(f"Updated metrics: {write_results_csv(all_results, output_dir)}")
        logger.info(f"Updated sequences: {write_sequences_tsv(all_results, output_dir)}")

        if local:
            torch.cuda.empty_cache()

    return all_results


if modal is not None:

    @app.local_entrypoint()
    def main(
        target_name: str | None = None,
        binder_name: str | None = None,
        target_sequence: str | None = None,
        binder_sequence: str | None = None,
        use_scaling_critics: bool = False,
        is_antibody: bool | None = None,
        local: bool = False,
        seed: int | None = None,
        batch_size: int = 1,
        output_dir: str | None = None,
        num_designs: int = 1,
        design_prefix: str | None = None,
    ):
        """Entrypoint for `modal run binder_design.py`.

        epitope_indices is not exposed here because Modal's CLI cannot pass list
        arguments; use the local argparse CLI for epitope-restricted designs.
        """
        run_design(
            target_name=target_name,
            binder_name=binder_name,
            target_sequence=target_sequence,
            binder_sequence=binder_sequence,
            use_scaling_critics=use_scaling_critics,
            is_antibody=is_antibody,
            local=local,
            seed=seed,
            batch_size=batch_size,
            output_dir=output_dir,
            num_designs=num_designs,
            design_prefix=design_prefix,
        )


def _parse_args(argv: list[str] | None = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Binder design with ESMFold2 and ESMC."
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help=f"Preset target: {', '.join(TARGET_SEQUENCES)}. Omit when using --target-sequence.",
    )
    parser.add_argument(
        "--target-sequence",
        default=None,
        help="Target sequence, or a path to a text/FASTA file holding it.",
    )
    parser.add_argument(
        "--binder-name",
        default=None,
        help=f"Preset binder prompt: {', '.join(BINDER_PROMPT_FACTORIES)}. Omit when using --binder-sequence.",
    )
    parser.add_argument(
        "--binder-sequence",
        default=None,
        help=(
            f"Binder prompt using '{MUTABLE_TOKEN}' at designed positions and literal "
            "amino acids at fixed ones, or a path to a file holding it."
        ),
    )
    parser.add_argument(
        "--use-scaling-critics",
        action="store_true",
        help="Also evaluate with the 15-checkpoint scaling ensemble (not available locally).",
    )
    parser.add_argument(
        "--is-antibody",
        choices=["true", "false", "none"],
        default="none",
        help="Force the antibody setting. 'none' auto-detects with ANARCI.",
    )
    parser.add_argument(
        "--local", action="store_true", help="Run in this process instead of on Modal."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; run N uses seed+N. Omit for a fresh seed per run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Designs optimized in parallel on the GPU within one run.",
    )
    parser.add_argument(
        "--epitope-indices",
        type=int,
        nargs="+",
        default=None,
        help="0-based indices into the target sequence to focus the interface loss on.",
    )
    parser.add_argument(
        "--cdr-definition",
        choices=CDR_DEFINITIONS,
        default=CDR_DEFINITION,
        help=(
            "Which residues count as CDR for cdr_distogram_iptm_proxy (scoring only; "
            f"designed positions come from '{MUTABLE_TOKEN}' in the prompt). "
            f"Default {CDR_DEFINITION!r}."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory for PDB/CSV/TSV outputs."
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA device index. Applied at import time, before torch loads.",
    )
    parser.add_argument(
        "--num-designs",
        type=int,
        default=1,
        help="Number of sequential design runs.",
    )
    parser.add_argument(
        "--design-prefix", default=None, help="Prefix for output names (default: design)."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    # NOTE - args.gpu is intentionally not applied here: CUDA_VISIBLE_DEVICES is
    # already set from argv at the top of this module, before torch was imported.
    is_antibody_arg = {"true": True, "false": False, "none": None}[args.is_antibody]

    if not (args.target_name or args.target_sequence):
        raise SystemExit(
            "Provide a target: --target-name for a preset "
            f"({', '.join(TARGET_SEQUENCES)}), or --target-sequence with a "
            "sequence or FASTA path. See --help."
        )

    run_design(
        target_name=args.target_name,
        target_sequence=args.target_sequence,
        binder_name=args.binder_name,
        binder_sequence=args.binder_sequence,
        use_scaling_critics=args.use_scaling_critics,
        is_antibody=is_antibody_arg,
        local=args.local,
        seed=args.seed,
        batch_size=args.batch_size,
        epitope_indices=args.epitope_indices,
        cdr_definition=args.cdr_definition,
        output_dir=args.output_dir,
        num_designs=args.num_designs,
        design_prefix=args.design_prefix,
    )
