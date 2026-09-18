# LatentFlow Final Revision Experiments

This directory owns the confirmatory experiments that are absent from the
release benchmark. It imports the locked LatentFlow implementation, but writes
all new checkpoints, histories, metrics, manifests, and summaries beneath
`physics_modal_v4/enhancements/results/`.

`protocol.json` is the immutable, machine-readable definition of the final
architecture, selected hyperparameters, training budget, seeds, and selection
rule used by this suite.

> **Selector-safe protocol.** The original v1 control exporter rebuilt a model
> from `state_dict` without restoring the validation-selected `reference_only`
> runtime flag. Because a new `LatentFlow` starts in stage A, extension-selected
> controls were evaluated as the frozen reference. Selector-safe v2 protocols
> restore `selection.best_stage` before evaluation, serialize the inference flag
> beside the weights, and reject v1 summaries as stale. Existing v1 checkpoints
> can be repaired by evaluation-only replay; v1 metric summaries must not be used
> for central, factorial, capacity, full-data, data-budget, or channel-order claims.

The experiment runner enforces three invariants:

1. LatentFlow contains no cross-process exchange modules or parameters.
2. Checkpoint selection uses validation MSE; the test split is evaluated only
   after model selection.
3. Exchange-era results and results produced by a different protocol hash are
   rejected rather than silently reused.

## Final Submission Checks

The last revision pass adds checkpoint-level audits without changing the
released architecture. Run them in this order:

```bash
bash physics_modal_v4/enhancements/submit_vista.sh submission_smoke
bash physics_modal_v4/enhancements/submit_vista.sh traffic_replay
bash physics_modal_v4/enhancements/submit_vista.sh submission_provenance
bash physics_modal_v4/enhancements/submit_vista.sh submission_numerical
bash physics_modal_v4/enhancements/submit_vista.sh submission_fusion_profile
bash physics_modal_v4/enhancements/submit_vista.sh submission_selector
```

The Traffic replay is a gate: batch-1 and batch-4 predictions from the same
canonical Traffic-720 seed-42 checkpoint must agree within the declared
tolerance before any conflicting result is used. Provenance covers all 140
final fits. Numerical and fusion profiles use seed 42 over all 28 tasks.
Selector analysis covers all 140 fits and reuses the cached raw-extension
checkpoints produced by the prior checkpoint audit.

Only after these evaluation checks pass, launch the one new training control:

```bash
bash physics_modal_v4/enhancements/submit_vista.sh matched_controls
```

This control replaces the GP separator with a generic deterministic temporal
filter bank whose branches sum exactly to the structural state. It retains the
same process count, process-preserving continuation, reference candidate,
training stages, and validation selector. In the smoke configuration it has 585,510
active parameters versus 585,435 for LatentFlow (0.013% difference). The
existing causal `X->Z` control
already supplies the equal-stage single-state deterministic comparison.

After every phase completes:

```bash
bash physics_modal_v4/enhancements/submit_vista.sh submission_aggregate
```

## Final Multiseed And Budget Checks

Two final controls resolve the remaining attribution and training-budget
questions without changing LatentFlow. The first adds seeds 43--46 to the
parameter-matched deterministic multibranch control (112 fits). The second
fits TimePro at 128, 512, and all available training origins (84 fits). Its
2,048-origin point is reused from the matched headline benchmark. All four
budget levels use the same fixed validation/test protocol and nested training
subsets.

```bash
bash physics_modal_v4/enhancements/submit_vista.sh matched_controls_multiseed
bash physics_modal_v4/enhancements/submit_vista.sh timepro_data_curve
```

The recommended queued workflow submits a two-cell `gh-dev` smoke test, both
16-worker arrays with `afterok` dependencies, and the final aggregator in one
command:

```bash
bash physics_modal_v4/enhancements/submit_vista.sh remaining_all
```

After both arrays finish:

```bash
bash physics_modal_v4/enhancements/submit_vista.sh remaining_aggregate
```

The compact outputs are written beneath
`enhancements/results/remaining_checks/statistics/`. Strict aggregation
requires 140 matched-control pairs and 112 budget points per model.

Strict aggregation expects 140 provenance rows, 140 selector rows, 28
numerical rows, 28 fusion-profile rows, one Traffic replay, and 28 matched
non-GP control rows under `enhancements/results/submission_checks/statistics/`.
## Selector-Safe Ablation Repair

If the original v1 checkpoints are still present under the enhancement results
root, repair the five-seed central and three-seed factorial metrics without
retraining:

```bash
bash physics_modal_v4/enhancements/submit_vista.sh repair_ablation_tables
```

This submits three dependent jobs: central checkpoint replay, factorial
checkpoint replay, and core-only aggregation. Corrected tables are written to
`physics_modal_v4/enhancements/results/selector_safe_statistics/`. The required
files are `central_dataset_summary.csv`, `central_paired_effects.csv`,
`central_task_summary.csv`, `factorial_cell_summary.csv`, and
`factorial_effects.csv`. If any v1 checkpoint was deleted, only that missing
cell must be retrained with the selector-safe protocol.

## Experimental Matrix

| Phase | Task groups | Reported cells | Seeds | Purpose |
|---|---:|---:|---|---|
| `central` | 140 | 700 | 42--46 | final, `X->Z`, early fusion, single scale, no bank |
| `factorial` | 84 | 672 | 42--44 | balanced scale x bank x continuation study |
| `strongest` | 112 | 112 new | 43--46 | locked strongest non-TimePro comparator per dataset |
| `capacity` | 28 | 112 | 42 | single-scale and no-bank capacity-matched controls |
| `full_data` | 28 | 56 | 42 | LatentFlow and TimePro using every train/validation window |
| `data_curve` | 112 | 112 | 42 | LatentFlow with nested 128/512/2048/all training budgets and fixed validation/test evaluation |
| `diagnostics` | 7 | 7 | 42 | bank/scale oracle gaps and gate-faithfulness interventions |
| `channel_order` | 4 | 4 | 42 | two fixed permutations on Electricity/Traffic-720 |
| `channel_order_expanded` | 6 | 6 | 42 | three additional fixed permutations after the pilot trigger |
| `synthetic` | 35 | 105 | 42--46 | seven controlled process settings and three continuation designs |
| `real_scaling` | 6 | 12 | 42 | nested Traffic channel sets for LatentFlow and TimePro |
| `prior_controls` | 252 | 252 | 42--44 | no group priors, permuted families, neutral process scales |
| `checkpoint_audits` | 140 | 280 audit rows | 42--46 | reference/extension accounting and fusion-identity audit |

## Final Reviewer Audits

These controls do not change the frozen no-exchange architecture. The prior
suite changes one initialization assumption at a time while retaining the
selected field, bank, continuation, optimizer, data, and validation protocol.

- `no_sensor_group_priors` preserves each domain's group count but replaces
  semantic and behavioral initialization by near-uniform learned per-channel
  group logits; process allocations also begin uniformly and receive no group
  prior penalty.
- `permuted_process_families` reverses the family-to-process assignment while
  retaining process counts and trainability.
- `neutral_process_scales` initializes all datasets from the same sample-index
  values: period 24, nonperiodic length 12, periodic phase width 1, decay 48.

The checkpoint audit compares the frozen reference, the best raw extension
candidate, and the validation-selected result. Existing extension-selected
checkpoints are evaluated directly. A reference-selected checkpoint does not
contain the discarded extension state, so only those cases perform a seeded
head-plus-extension replay from the frozen backbone, then cache the raw
candidate. The raw extension is the complete extension candidate,
including its learned reference gate, before the discrete reference fallback.

The fusion anomaly audit compares early fusion and causal `X->Z` before and
after reference fallback using bitwise hashes, maximum absolute differences,
and explicit reference-selection counts. It requires the five-seed central
checkpoints, not only their CSV summaries.

Run the two arrays after the reviewer smoke test. They are independent and can
be queued together if the account's submission limit permits 32 array workers:

```bash
bash physics_modal_v4/enhancements/submit_vista.sh reviewer_smoke
bash physics_modal_v4/enhancements/submit_vista.sh prior_controls
bash physics_modal_v4/enhancements/submit_vista.sh checkpoint_audits
```

After both arrays complete, produce strict summaries and the no-training
dataset-wise mechanism breakdown:

```bash
bash physics_modal_v4/enhancements/submit_vista.sh final_audit_analysis
```

Expected outputs under `enhancements/results/final_audits/` are:

- `prior_control_dataset_effects.csv`;
- `reference_extension_by_dataset.csv`;
- `fusion_anomaly_by_dataset.csv`;
- `mechanism_effects_by_dataset.csv`;
- `factorial_main_effects_by_dataset.csv`;
- the corresponding run-level audit tables and `final_audit_status.json`.

One task group is one dataset-horizon-seed combination. A worker evaluates all
variants belonging to that group sequentially, which keeps its data split,
seed, and validation rule identical.

The synthetic generator has three identifiable additive targets: periodic,
smooth, and innovation. Trend and Matérn-like variation form the smooth target;
OU dynamics, abrupt events, and observation noise form the innovation target.
The study therefore tests recovery at the kernel-family level and does not
claim unique recovery of every constituent waveform.

## Prerequisite

The revised five-seed no-exchange headline run must exist first:

```bash
bash physics_modal_v4/submit_vista.sh headline
```

The enhancement runner accepts only the candidate identifier defined by the
current release code. Historical exchange-enabled checkpoints cannot satisfy
this prerequisite.

## Vista Execution Order

Run from the repository root. Upload the files listed in
`physics_modal_v4/VISTA_UPLOAD_MANIFEST.txt`, normalize line endings once, and
then proceed in order.

```bash
find physics_modal_v4/enhancements -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec sed -i 's/\r$//' {} +
mkdir -p physics_modal_v4/enhancements/logs

bash physics_modal_v4/enhancements/submit_vista.sh smoke
bash physics_modal_v4/enhancements/submit_vista.sh central
bash physics_modal_v4/enhancements/submit_vista.sh factorial
bash physics_modal_v4/enhancements/submit_vista.sh strongest
bash physics_modal_v4/enhancements/submit_vista.sh capacity
bash physics_modal_v4/enhancements/submit_vista.sh full_data
bash physics_modal_v4/enhancements/submit_vista.sh data_curve
bash physics_modal_v4/enhancements/submit_vista.sh diagnostics
bash physics_modal_v4/enhancements/submit_vista.sh channel_order
bash physics_modal_v4/enhancements/submit_vista.sh channel_order_expanded
bash physics_modal_v4/enhancements/submit_vista.sh synthetic
bash physics_modal_v4/enhancements/submit_vista.sh real_scaling
bash physics_modal_v4/enhancements/submit_vista.sh aggregate
```

Do not launch `factorial` until `central` has completed because four factorial
cells are exact central-run reuses. The remaining independent phases may be
queued after the smoke test subject to Vista's job limits.

Set `LATENTFLOW_ENHANCEMENT_SHARDS` to change parallelism; the default is 16.
For example:

```bash
LATENTFLOW_ENHANCEMENT_SHARDS=8 bash physics_modal_v4/enhancements/submit_vista.sh strongest
```

## Monitoring

```bash
squeue -u "$USER" -o "%.18i %.18j %.2t %.10M %.30R"
tail -f physics_modal_v4/enhancements/logs/lf_cent5.<JOBID>_0.out
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

## Aggregation

The strict aggregate job expects 700 central rows, 672 factorial rows, 112 new
comparator rows, 112 capacity-control rows, 56 full-data rows, 112 data-budget
rows, seven diagnostic
rows, ten channel-order rows, 105 synthetic-mechanism rows, and 12 real
channel-scaling rows. It creates:

- `central_task_summary.csv` and `central_paired_effects.csv`;
- `factorial_cell_summary.csv` and `factorial_effects.csv`;
- `strongest_comparator_summary.csv`;
- `capacity_summary.csv`;
- `full_data_summary.csv` and `channel_order_runs.csv`;
- `data_budget_runs.csv`, `data_budget_dataset_summary.csv`, `data_budget_overall_summary.csv`, and `data_budget_task_deltas.csv`;
- `diagnostic_summary.csv` and `diagnostic_component_metrics.csv`;
- `synthetic_summary.csv` and process-recovery detail tables;
- `real_channel_scaling.csv`;
- `run_manifest.csv`, including configuration hashes and result SHA-256 sums.

For an intermediate audit, run the aggregator manually with
`--allow-incomplete`. Final paper artifacts must use strict aggregation.
