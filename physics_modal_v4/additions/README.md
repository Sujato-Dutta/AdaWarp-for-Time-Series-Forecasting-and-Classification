# LatentFlow Additional Evidence

This directory contains the extended evidence for the finalized no-exchange
LatentFlow implementation in the parent directory.

## Scope

| Block | Runs | Purpose |
|---|---:|---|
| TimePro seeds 43--46 | 112 | Matched five-seed comparison on 28 tasks |
| Synthetic mechanism | 40 | Periodic + Matern + OU recovery, independent/interacting |
| Central ablation | 700 reported cells | Five variants, all 28 tasks, five seeds |
| Missing-prefix stress | 8 tasks | 0/20/40% masks for Weather/Traffic |
| Process visualization | 2 | Weather/Traffic H=720 interventions |
| VPNet separator control | 28 | Tests the VPNet-plus-stochastic-frontend explanation |
| Channel scaling | 9 sizes | Latency/memory from 7 to 2048 channels |

All real-data experiments retain input length 96, training-only normalization,
the same 2048 training and 1024 validation windows, validation-MSE model
selection, and evaluation over every available test window. TimePro and
LatentFlow use the same five seeds, 42--46. Central-ablation artifacts are
reused only when their candidate metadata and validation protocol exactly
match the revised no-exchange model.

The missing-prefix study masks values on an otherwise regular grid and applies
prefix-only forward filling. The implementation does not consume arbitrary
timestamps, so this experiment must be described as sensor-dropout robustness,
not native irregular-time forecasting.

## Vista Order

After uploading `physics_modal_v4/additions/`, from the repository root:

```bash
find physics_modal_v4/additions -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec sed -i 's/\r$//' {} +
mkdir -p physics_modal_v4/additions/logs
bash physics_modal_v4/additions/submit_vista.sh smoke
```

Check the smoke job before launching arrays. Then submit the independent blocks
as allocation limits permit:

```bash
bash physics_modal_v4/additions/submit_vista.sh timepro
bash physics_modal_v4/additions/submit_vista.sh synthetic
bash physics_modal_v4/additions/submit_vista.sh central
bash physics_modal_v4/additions/submit_vista.sh missing
bash physics_modal_v4/additions/submit_vista.sh process
bash physics_modal_v4/additions/submit_vista.sh vpnet_control
bash physics_modal_v4/additions/submit_vista.sh scaling
```

Submit `central` only after all 140 revised LatentFlow headline runs exist. It
evaluates `final_no_exchange`, `x_to_z`, `early_fusion`, `no_multiscale`, and
`no_linear_bank` for five seeds on all 28 dataset--horizon tasks. The final
cell is reused from the headline run; the other cells are trained separately.
The `no_multiscale` cell performs validation-only selection over three
single-scale candidates. Consequently, the central matrix contains 700
reported cells but can require up to 840 new model fits after the 140 final
headline cells are reused.

Each array uses at most 16 concurrent GH nodes by default. Override this with,
for example, `LATENTFLOW_ADDITION_SHARDS=8` when other jobs occupy the allocation.

After all blocks finish:

```bash
bash physics_modal_v4/additions/submit_vista.sh aggregate
```

The aggregator writes audit-ready CSVs and PDF figures to
`physics_modal_v4/additions/results/summary/`. It refuses incomplete five-seed
evidence by default. During debugging only, add `--allow-incomplete` when
calling `python -m physics_modal_v4.additions.aggregate` directly.

## Monitoring

```bash
squeue -u "$USER" -o "%.18i %.24j %.2t %.10M %.30R"
tail -f physics_modal_v4/additions/logs/JOB_NAME.JOB_ID_ARRAY_INDEX.out
```

Every result line has a unique prefix such as `timepro-multiseed-result`,
`synthetic-result`, `central-ablation-result`, or `vpnet-separator-result`.
