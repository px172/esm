# px172 binder design fork

`binder_design.py` is a fork of `cookbook/tutorials/binder_design.py`, rebased onto
upstream `cf002f1` and extended with: `--gpu` selection, epitope-restricted interface
loss, `--cdr-definition` (imgt/chothia/kabat/north), sequence loading from file,
interface-contact metrics, batch design runs, and PDB/CSV/TSV output. `run.sh` drives
the IL8 (cxcl8) N-loop campaign. `binder_design.py.bak` is the pre-rebase version.

## Running

```bash
conda activate biohub-esm     # REQUIRED: puts hmmscan on PATH for ANARCI CDR scoring
cd px172 && ./run.sh          # ~5.8 h for 128 designs on one RTX PRO 6000
```

Outputs land in `<BATCH_NAME>_D<TOTAL>/`: one PDB per (design, critic), plus
`design_results.csv` (metrics) and `design_sequences.tsv` (VH/VL + where the binder
landed). `candidates_on_epitope.csv` / `candidates_off_epitope.csv` split designs by
`epitope_contacts` (>=3 = on the requested epitope; 0 = an alternative site — a real,
often-convergent binding mode, not a failure).

## Performance / environment notes

Measured for this config (cxcl8 + 243-residue scFv = 320 tokens, batch_size 2, torch
2.12.0+cu130, single RTX PRO 6000 96 GB). Re-measure for a much larger complex.

- **Bottleneck is the ESMFold2 folding trunk = ~81% of each step** (forward 34% +
  backward 47%); ESMC LM-loss is only ~19%. So folding-side speedups matter; ESMC-side
  ones (xformers/flash-attn/transformer-engine) give no measurable design speedup.
- **`COMPILE=True` (set) = ~1.6x faster.** Keyed on tensor shape, so it pays off with a
  fixed-length prompt; a preset binder that resamples CDR length per seed recompiles per length.
- **`batch_size=2` is the throughput sweet spot.** 1→2 is +31% designs/hr; bs>2 gives
  nothing (GPU compute-saturated). Two concurrent jobs on one card also give no gain —
  only a second physical GPU does.

### cuequivariance: do NOT install (verified 2026-07-17)

The folding trunk can use NVIDIA cuequivariance kernels via
`set_kernel_backend("cuequivariance")`. It is tempting because folding is the 81%
bottleneck, and it **does install and run on cu130** (`cuequivariance-torch` +
`cuequivariance-ops-torch-cu13`, both 0.10.0; the triangle kernel runs correctly).

**But it is ~14% SLOWER for this config** — clean A/B: 1.98 s/step off vs 2.25 s/step on,
same loss trajectory. It clashes with `torch.compile` (its `torch.jit.script` kernels
cause graph breaks) and the 320-token complex is too small to amortize kernels designed
for large L. It was installed, benchmarked, and removed. Do not reinstall unless you move
to a much larger complex (many-hundred-residue target or multi-chain), where its
large-L kernels may finally win — and re-benchmark against `COMPILE=True` before trusting it.

The code now selects the fused Triton backend instead
(`BACKEND_FUSED if TRITON_KERNELS_AVAILABLE else None`), matching upstream after it
dropped the cuequivariance backend in 3.4.2. The previous `CUE_AVAILABLE` check resolved
to `None` on this box, so no kernel backend was active at all — the measurements above
predate that fix and are worth re-running.
