# MVPF evidence summary

## Matched 24-task evidence

- MVPF average rank is **2.458 MSE** and **2.333 MAE** (lower is better).
- It obtains **10/24 outright MSE wins** and **10/24 outright MAE wins**.
- Friedman: MSE chi-square=60.556, p=1.17e-10; MAE chi-square=91.833, p=5.2e-17.
- Versus PatchTST, MVPF wins/ties/loses **14/0/10** on MSE and **11/0/13** on MAE.
- PatchTST comparison is not significant after Holm correction: MSE adjusted p=0.317; MAE adjusted p=0.345.
- Significant Holm-corrected MSE comparisons: DLinear, TimesNet, iTransformer, FEDformer, VPNet.
- Significant Holm-corrected MAE comparisons: DLinear, TimesNet, iTransformer, FEDformer, VPNet.

The defensible claim is: **MVPF achieves the lowest mean MSE and MAE and the most task wins; it has the best MSE average rank and ties the best MAE average rank in the eight-method table, while remaining statistically tied with PatchTST and TimeMixer under task-level Holm-Wilcoxon testing.**

## Horizon behaviour

- MSE h=96: median PatchTST-relative improvement +1.91% (4/0/2 W/T/L).
- MSE h=192: median PatchTST-relative improvement +2.20% (4/0/2 W/T/L).
- MSE h=336: median PatchTST-relative improvement +0.25% (3/0/3 W/T/L).
- MSE h=720: median PatchTST-relative improvement -0.03% (3/0/3 W/T/L).
- MAE h=96: median PatchTST-relative improvement +0.24% (3/0/3 W/T/L).
- MAE h=192: median PatchTST-relative improvement +0.87% (4/0/2 W/T/L).
- MAE h=336: median PatchTST-relative improvement -1.00% (2/0/4 W/T/L).
- MAE h=720: median PatchTST-relative improvement -0.15% (2/0/4 W/T/L).

## Ablations

- no_linear_field MSE: median degradation +0.196%, effect=+0.314, core-Holm p=0.248.
- no_linear_field MAE: median degradation +0.295%, effect=+0.438, core-Holm p=0.248.
- single_scale_16 MSE: median degradation +0.869%, effect=+0.495, core-Holm p=0.213.
- single_scale_16 MAE: median degradation +0.464%, effect=+0.448, core-Holm p=0.248.

The ablations support positive *effect direction* for the multiscale representation and linear field bank, but do not establish corrected task-level significance. Adaptive shifts are operationally inactive in many final-model tasks because prototype memory is disabled, and removing trend/residual decomposition slightly improves the 20-task average; neither should be presented as a demonstrated source of gains.

## Artifact limitation

- Locally available MVPF raw task predictions: 0/24.
- Locally available PatchTST raw task predictions: 0/24.
- Locally available validation-best MVPF checkpoints: 24.
- The checkpoints support model re-evaluation and gate interventions. Matched raw forecast arrays for MVPF and PatchTST are still required for paired per-window, per-channel, lead-time, and qualitative forecast comparisons.

## Scope statement

All significance tests above measure consistency across 24 dataset-horizon tasks from seed 42. They do not estimate training-seed variability and must not be described as a substitute for multi-seed experiments.
