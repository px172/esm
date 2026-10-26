export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# NOTE - --is-antibody true makes the scoring pass compute cdr_distogram_iptm_proxy,
# which goes through abnumber -> ANARCI -> the `hmmscan` binary from HMMER. hmmscan
# is not a Python package: it lives in the conda env's bin/, so this script must run
# with the env on PATH:
#
#     conda activate biohub-esm
#
# `import abnumber` succeeds without it (that part IS a Python package), so nothing
# fails at startup -- the FileNotFoundError only surfaces in the scoring pass, i.e.
# after the full 150-step design loop has already burned. At 128 designs that is
# ~5.8h of wasted GPU time, so fail immediately instead.
if ! command -v hmmscan > /dev/null 2>&1; then
	echo "ERROR: hmmscan is not on PATH -- run 'conda activate biohub-esm' first." >&2
	echo "       ANARCI needs it for --is-antibody true. Without this check the run" >&2
	echo "       would only fail after the design loop finished (~5.8h at 128 designs)." >&2
	exit 1
fi

# NOTE - on this box `conda activate biohub-esm` leaves /usr/bin ahead of the env
# on PATH, so a bare `python` can resolve to /usr/bin/python (no biotite/torch/esm)
# even with the env active. hmmscan, however, exists ONLY in the env, so the check
# above guarantees it came from the env -- and the env's python sits right next to
# it. Deriving PYTHON from hmmscan's directory is therefore immune to PATH ordering.
PYTHON="$(dirname "$(command -v hmmscan)")/python"

# NOTE - bumped R002 -> R003. The outputs of this run differ from R002's (PDBs now
# carry per-residue pLDDT in the B-factor column), and the design_id scheme is
# unchanged, so reusing R002 would overwrite the existing IL8_ESM_P6K_R002_D128
# results in place. Set this back to R002 only if you mean to replace them.
BATCH_NAME="IL8_ESM_P6K_R005"

# NUM_RUNS sequential runs x BATCH_SIZE designs optimized together = TOTAL designs.
# BATCH_SIZE=2 is the throughput sweet spot on this config: measured 1.32 / 1.01 /
# 1.05 / 1.06 s per design per step at batch 1 / 2 / 4 / 8, i.e. all of the gain is
# in 1->2 (1.30x) and the GPU is saturated beyond that. Peak VRAM ~26 GiB (~31 GiB
# reserved) out of 95 GiB, which still leaves room to run ollama alongside.
# Batching is a throughput knob only: each design is optimized independently (the
# 1/B from loss.mean() cancels against the per-element gradient normalization).
# It is not bit-reproducible across batch sizes though -- batched-kernel float
# non-determinism makes trajectories diverge -- so reproducing a specific design
# needs the same seed AND the same BATCH_SIZE.
NUM_RUNS=64
BATCH_SIZE=2
TOTAL=$((NUM_RUNS * BATCH_SIZE))

# NOTE - COMPILE=True in binder_design.py is keyed on tensor shape. This config
# passes a fixed-length prompt (GH2-H3_13.txt, 243 residues), so the folding trunk
# compiles once and all 150 steps reuse it -- that is where the ~1.6x speedup comes
# from. Anything that makes the binder length vary re-triggers compilation for each
# new length and eats the win: e.g. switching to --binder-name minibinder (or any
# BINDER_PROMPT_FACTORY preset), which resamples CDR lengths per seed. Editing
# GH2-H3_13.txt to a different fixed length is fine -- it just compiles once at the
# new shape. If you move to variable-length prompts, re-measure before trusting the
# timings above; setting COMPILE=False may then be faster.

# NOTE - --cdr-definition selects which residues cdr_distogram_iptm_proxy scores. It
# is a SCORING knob only: the positions actually designed come from '#' in
# GH2-H3_13.txt and do not change with it. Measured on one R002 design (243-residue
# scFv, 32 '#' positions), the definitions pick out:
#     imgt     47 residues, covering 28/32 of the designed positions
#     chothia  51 residues, covering 28/32   <- upstream default
#     kabat    60 residues, covering 30/32
#     north    64 residues, covering 32/32   <- the only one that covers them all
# north is contact-based and the widest, so every '#' lands inside a CDR. Note the
# converse still holds: 32 of north's 64 residues are framework positions this
# prompt holds fixed, so the metric spans more than what is being designed.
# The chosen value is recorded per row in design_results.csv.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" binder_design.py \
	--target-name cxcl8 \
	--binder-sequence GH2-H3_13.txt \
	--is-antibody true \
	--cdr-definition north \
	--local \
	--batch-size ${BATCH_SIZE} \
	--num-designs ${NUM_RUNS} \
	--output-dir ${BATCH_NAME}_D${TOTAL} \
	--design-prefix ${BATCH_NAME} \
	--epitope-indices 16 17 18 19 20 21 22 23 24
