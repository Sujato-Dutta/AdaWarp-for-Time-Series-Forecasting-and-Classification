# LatentFlow

This directory is the clean research release for LatentFlow and its matched long-horizon forecasting evaluation.

## Locked architecture

LatentFlow uses:

- an explicit causal structural map X -> Z;
- variational multi-stochastic latent separation;
- learned inducing timestamps and a representation objective;
- process-preserving variate-patch continuation with no cross-process exchange;
- the selected multiscale variate-patch backbone and adaptive linear field bank;
- the final two field blocks, future projector, and decoder unfrozen;
- validation-MSE checkpoint selection.

The test split was not used to select architecture or hyperparameters. The
final point model does not use primal-dual optimization.

### Selected hyperparameters

| Dataset | Latent dim | Inducing points | Patch geometry | Representation weight |
|---|---:|---:|---|---:|
| ETTh1 | 16 | 32 | 16,32,48 | 0.05 |
| ETTh2 | 8 | 32 | 16,32,48 | 0.001 |
| ETTm1 | 8 | 32 | 8,24,48 | 0.01 |
| ETTm2 | 8 | 16 | 8,16,32 | 0.001 |
| Weather | 16 | 32 | 8,16,32 | 0.001 |
| Electricity | 16 | 32 | 8,16,32 | 0.001 |
| Traffic | 16 | 32 | 8,16,32 | 0.001 |

The machine-readable selections, search space, and validation scores are in
results/hyperparameters/.

## Matched protocol

All models use input length 96, horizons {96,192,336,720}, train-only
normalization, the same 2,048 training windows and 1,024 validation windows,
validation-MSE checkpointing, and every chronological test window.

The final comparison contains:

- LatentFlow: seeds 42,43,44,45,46, reported as mean +/- sample standard
  deviation;
- VPNet, xCPD, TimePro, iTransformer, TimeMixer, and DLinear: matched seed 42;
- ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, and Traffic.

This gives 140 LatentFlow runs and 168 comparison runs. Competitor cells are
single-seed results and are not assigned artificial standard deviations.
Efficiency is measured once at seed 42 for all seven models.

## Ablation program

The broad diagnostic ladder uses seed 42, identical data windows and
optimization settings, and validation-MSE checkpoint selection. The central
confirmatory removals described below use all five seeds on all 28 tasks.

The cumulative variate-patch backbone ladder is:

1. single_scale_field: one patch scale, fixed local field, no component gate,
   and no adaptive linear field bank.
2. multiscale_field: adds the selected dataset-specific patch scales.
3. adaptive_local_field: adds adaptive local variate-patch neighborhoods.
4. component_gated_field: adds input-conditioned component gating.
5. linear_field_bank: adds normalized/decomposition/slope/persistence
   candidates and the outer field-bank gate, yielding LatentFlow's finalized
   variate-patch backbone.

The cumulative latent stochastic ladder is:

1. x_to_z: explicit causal structural encoding.
2. single_stochastic: one stochastic latent process.
3. multi_stochastic: multiple process families with fixed inducing locations.
4. learned_inducing_elbo: learned inducing timestamps and representation loss.
5. process_preserving: preserves the process axis through every field block and
   future projection. Processes are fused only after decoding; no exchange
   module is allocated in the finalized architecture.

Direct controls are fixed_inducing, random_process_families,
no_sensor_priors, and no_nll. Adaptation depth is isolated by
continuation_frozen, continuation_last1, continuation_last2, and
continuation_full. The selected final depth adapts the last two field blocks.

There are 18 variants over 28 dataset-horizon tasks: 504 seed-42 runs.

The final revision matrix is implemented under `enhancements/` and evaluates
`final_no_exchange`, `x_to_z`, `early_fusion`, `no_multiscale`, and
`no_linear_bank` for seeds 42--46 on every dataset and horizon. Its 700
reported cells replace the earlier three-dataset, horizon-720 pilot as the
evidence for the central architectural claims. The final-model cells are
reused from the revised headline runs; `no_multiscale` selects among three
single-scale candidates using validation MSE. The same suite contains the
balanced scale x bank x continuation factorial, capacity-matched controls,
full-data confirmation, gate diagnostics, and channel-order pilot.

## Files

- model.py: finalized LatentFlow architecture.
- models/: the six retained comparison implementations.
- experiments/run.py: LatentFlow training, ablations, and efficiency.
- experiments/baselines.py: comparison training and efficiency.
- experiments/training.py: validation-checkpointed optimization.
- experiments/protocol.py: shared splits, windows, normalization, and metrics.
- experiments/statistics.py: aggregation, paired tests, ranks, and figures.
- vista.sbatch, submit_vista.sh: Vista execution.

## Vista execution

> **RUN THE FOLLOWING COMMANDS IN THIS EXACT ORDER.**
> Wait for each numbered stage to complete successfully before starting the next
> dependency-sensitive stage.

### 0. Prepare and smoke test

Run these commands immediately after uploading the files listed in
VISTA_UPLOAD_MANIFEST.txt:

~~~bash
cd "$WORK/motion_code-master"
find physics_modal_v4 -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec sed -i '1s/^\xEF\xBB\xBF//' {} +
find physics_modal_v4 -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec sed -i 's/\r$//' {} +
mkdir -p physics_modal_v4/logs
bash physics_modal_v4/submit_vista.sh smoke
~~~

> [!CAUTION]
> **Do not submit the main experiments until the smoke job reports
> LATENTFLOW-SMOKE PASSED and all six baseline smoke checks pass.**

### 1. Run LatentFlow

**140 runs: 7 datasets x 4 horizons x 5 seeds.**

~~~bash
bash physics_modal_v4/submit_vista.sh headline
~~~

### 2. Run the six comparison models

**168 runs: 6 models x 7 datasets x 4 horizons x seed 42.**

~~~bash
bash physics_modal_v4/submit_vista.sh baselines
~~~

Stages 1 and 2 are independent and may run concurrently if Vista has sufficient
submission capacity.

### 3. Run the five-seed central ablations

**700 reported cells: 5 variants x 7 datasets x 4 horizons x 5 seeds.**

> [!WARNING]
> Start this stage only after all revised LatentFlow headline runs from Stage 1
> have completed. Only candidate-matched no-exchange headline results are
> reused.

~~~bash
bash physics_modal_v4/enhancements/submit_vista.sh smoke
bash physics_modal_v4/enhancements/submit_vista.sh central
~~~

### 4. Run the controlled factorial and capacity controls

Start the factorial only after Stage 3 because it reuses exact central cells.

~~~bash
bash physics_modal_v4/enhancements/submit_vista.sh factorial
bash physics_modal_v4/enhancements/submit_vista.sh capacity
~~~

### 5. Run the broad seed-42 diagnostic ladder

**504 runs: 18 variants x 7 datasets x 4 horizons.**

> [!WARNING]
> Start this stage only after the seed-42 LatentFlow headline runs from Stage 1
> have completed. The finalized continuation_last2 ablation reads those
> validation-selected results directly.

~~~bash
bash physics_modal_v4/submit_vista.sh ablations
~~~

### 6. Run efficiency profiling

**196 profiles: 7 models x 7 datasets x 4 horizons, all at seed 42.**

> [!WARNING]
> Start this stage only after the seed-42 checkpoints for LatentFlow and all six
> comparison models exist.

~~~bash
bash physics_modal_v4/submit_vista.sh efficiency
~~~

### 7. Run the remaining revision checks

~~~bash
bash physics_modal_v4/enhancements/submit_vista.sh strongest
bash physics_modal_v4/enhancements/submit_vista.sh full_data
bash physics_modal_v4/enhancements/submit_vista.sh diagnostics
bash physics_modal_v4/enhancements/submit_vista.sh channel_order
~~~

### 8. Generate final statistics and figures

Run this only after Stages 1--5 have finished:

~~~bash
bash physics_modal_v4/submit_vista.sh statistics
bash physics_modal_v4/enhancements/submit_vista.sh aggregate
~~~

The final tables and paper-ready plots will be written under
results/final/statistics/.

### Optional: rerun one comparison model

Replace TimePro with VPNet, xCPD, iTransformer, TimeMixer, or DLinear as needed:

~~~bash
bash physics_modal_v4/submit_vista.sh baseline TimePro
~~~

Each GPU experiment uses 16 array shards by default.

## Paper artifacts

experiments/statistics.py writes task means and standard deviations, dataset
and overall summaries, win/tie/loss counts, average ranks, paired bootstrap
confidence intervals, one-sided Wilcoxon tests with Holm correction, Friedman
omnibus tests, complete ablation and efficiency tables, and paper-ready PDF/PNG
figures.

Efficiency reports total/trainable parameters, profiler FLOPs/GFLOPs, peak
inference/training GPU memory, training milliseconds per optimizer step, and
inference milliseconds per batch. Primary outputs are under
results/final/statistics/.

