# LatentFlow Final ICLR 2027 Revision Plan

## Scope and claim boundary

This sprint is limited to work that can credibly be completed in two to three
days. The architecture is frozen. The submission should describe LatentFlow as
a componentwise nonlinear continuation method built from sparse additive GP
coordinates, local variate-patch operators, a gated linear bank, and
validation-based model selection.

The revised paper must not claim:

- unavoidable information loss from summing the fitted GP component means;
- unique recovery of physical processes;
- a universal full-data state of the art;
- permutation invariance or irregular-time forecasting;
- calibrated multivariate uncertainty or a test-risk guarantee from fallback.

The strongest defensible claim is performance under the matched 2,048-window
protocol, together with evidence that componentwise nonlinear computation has
different empirical behavior from the precisely matched early-fusion control.

## Sprint status (2026-09-14)

Completed locally:

- [x] Replaced the incorrect information-loss argument with a scoped recovery lemma, a Bayes-risk statement, and nonlinear non-commutation analysis.
- [x] Distinguished `f_lat`, `f_mix`, and `f_sel`; documented candidate epochs, validation selection, the tie rule, and the selected density path.
- [x] Made five-seed LatentFlow versus TimePro the primary comparison and added source-family/source-seed intervals plus leave-one-source-family-out sensitivity.
- [x] Added selector confusion, regret, prior-control intervals, exact early-fusion/X-to-Z output-difference diagnostics, and selector-safe central results.
- [x] Added MAE factorial contrasts with explicit factor coding and corrected the single-resolution table and selection counts.
- [x] Narrowed the novelty claim, differentiated related decomposition methods, defined forecast-origin causality, and clarified that ``Flow'' is not a normalizing-flow claim.
- [x] Specified observation noise versus jitter, memberships, field depth, process-shared parameters, and batch-normalization axes.
- [x] Regenerated the manuscript package; citation, reference, source-dependency, and TeX-structure checks pass (34 cited entries). Local LaTeX compilation was intentionally not run.

Implementation and local analysis complete; Vista evidence still pending:

- [x] Implemented canonical Traffic H=720 batch-invariance replay and full artifact hashing; requires the Vista checkpoint and dataset.
- [x] Implemented checkpoint-derived early-fusion active parameters/FLOPs, extension-only selector diagnostics, and validation-versus-test selector data; requires Vista artifacts.
- [x] Added and passed exact-Gaussian component-recovery and dense-versus-sparse posterior tests locally (maximum mean error $1.67\times10^{-8}$; maximum variance error $4.66\times10^{-10}$).
- [x] Implemented the matched process-preserving deterministic multibranch control without changing the released model state; training remains to run on Vista.
- [x] Audited comparator provenance and result coverage in `enhancements/COMPARATOR_FIDELITY_AUDIT.md`; optional upstream one-task fidelity checks remain deferred.
## P0: Submission blockers

Complete these in order. Do not finalize the paper until items 1--5 pass.

### 1. Reconcile the Traffic H=720 numerical conflict

**Estimate:** 1--3 hours; evaluation only.

- [ ] Identify the exact seed-42, Traffic H=720 checkpoint used for the main
      result (`0.4914688`) and the channel-scaling endpoint (`0.4814112`).
- [ ] Evaluate one canonical selected checkpoint on the same 862 channels,
      test origins, preprocessing state, and selector with batch sizes 1 and 4.
- [ ] Record maximum and mean absolute prediction differences, MSE, MAE,
      checkpoint hash, configuration hash, data-window hash, and selected path.
- [ ] Verify `eval()` behavior and frozen normalization statistics so batch
      composition cannot change the forecast.
- [ ] Regenerate both the headline and scaling rows from the canonical manifest,
      or explicitly label the scaling endpoint as a separately retrained model.

**Pass condition:** batch-1 and batch-4 predictions agree within a declared
absolute/relative tolerance, and Tables 12/26 no longer imply that two different
scores came from the same checkpoint and configuration.

### 2. Audit the early-fusion intervention before using its 5.12% result

**Estimate:** 3--5 hours for code/provenance inspection; 4--8 GPU hours only if
the existing control is not matched.

- [ ] Write the exact forward equations for the process-preserving and
      early-fusion variants.
- [ ] Confirm whether every control was retrained or altered after training.
- [ ] Confirm matched sum/mean scaling, process count, branch width, modulation,
      router inputs, normalization, trainable parameters, optimizer, epochs,
      checkpoint selection, and reference candidate.
- [ ] Report parameters, FLOPs, and fallback rates for both variants.
- [ ] Report selected-pipeline and no-fallback extension-only MSE/MAE separately.
- [ ] Explain ties caused by shared reference selection.

**Conditional rerun:** if any material item is unmatched, rerun the smallest
valid design first: seed 42 over all 28 tasks for early versus preserved fusion,
crossed with modulation on/off. Expand seeds only after this matched run passes.

**Pass condition:** the paper can state exactly what changed and can no longer
attribute differences caused by amplitude, modulation, capacity, or selector
reuse to fusion location.

### 3. Replace the incorrect model-specific information-loss argument

**Estimate:** 3--4 hours; no training.

- [ ] Remove statements that the fitted early sum necessarily destroys the GP
      component tuple.
- [ ] Add the fixed-group recovery lemma:
      `m_q = A_q S^dagger sum_r m_r` under the stated additive Gaussian model.
- [ ] Retain the abstract kernel-inclusion proposition only with an attainable-
      state qualification and all side information conditioned upon.
- [ ] Add the Bayes-risk identity comparing conditioning on a full state and a
      fused statistic; state the condition required for strict inequality.
- [ ] Motivate componentwise continuation with nonlinear non-commutation:
      averaging before and after a nonlinear operator generally differs, with a
      curvature/dispersion bound rather than an accuracy guarantee.
- [ ] Remove unsupported claims of contraction, stability, physical
      identification, or selector safety.

**Pass condition:** every theorem is true for its declared variables and none is
presented as proof that the trained LatentFlow system must outperform fusion.

### 4. Separate the three predictors in method, figures, and tables

**Estimate:** 2--3 hours; no training.

- [ ] Define the process forecast `f_lat`, learned mixture `f_mix`, and hard
      validation-selected predictor `f_sel` separately.
- [ ] State exactly which object is called the "raw extension" in every table.
- [ ] Define the candidate set, eligible epochs, tie rule, and selector state.
- [ ] State the density used when the reference path is selected.
- [ ] Replace "safeguard" or "guarantee" with "validation fallback" or
      "empirical model selection."
- [ ] Update the architecture figure to show training, validation selection,
      and test inference as distinct operations.

### 5. Freeze a full-precision provenance manifest

**Estimate:** 3--5 hours; mostly analysis.

- [ ] Generate every result table from one full-precision manifest.
- [ ] Include dataset version/checksum, split indices, eligible origins, sampled
      origins and seed, unique time-point coverage, checkpoint hash, config hash,
      selected epoch/path, validation MSE, and test MSE/MAE.
- [ ] State that replay agrees for 115/140 fits and identify why 25 source metrics
      are unavailable; do not say all 140 were independently matched.
- [ ] Declare absolute and relative replay tolerances.
- [ ] Package anonymized code, locked requirements, configs, and exact commands.

## P1: High-return analysis from existing results

These require little or no new GPU work and should be completed on Day 2.

### 6. Correct the primary statistics

**Estimate:** 3--4 hours.

- [ ] Make five-seed LatentFlow versus TimePro the primary comparison; move the
      complete seed-42 leaderboard to the supplement.
- [ ] Predeclare mean paired relative MSE as primary and define its denominator
      and averaging order.
- [ ] Add source-aware bootstrap intervals that group ETTh1/ETTm1 and
      ETTh2/ETTm2 by station, while retaining all horizons and seeds.
- [ ] Add leave-one-source-family-out sensitivity and state that only a small
      number of independent source clusters exists.
- [ ] Keep dataset-seed intervals as conditional training-seed variability, not
      as the sole generalization uncertainty.
- [ ] Add selected-versus-reference paired confidence intervals, selector 2x2
      confusion counts, and the regret distribution.
- [ ] Add seed-level points and label post-hoc tests as exploratory.

### 7. Supply already-promised supplementary evidence

**Estimate:** 2--3 hours.

- [ ] Add dataset effects, reference-use rates, and intervals for all three
      prior controls.
- [ ] Add the missing single-resolution interval and identify the exact selected
      patch length for each dataset.
- [ ] Add MAE factorial contrasts and define factorial interaction coding.
- [ ] Qualify the linear-bank MSE contribution because its factorial confidence
      interval crosses zero.
- [ ] Report exact early-fusion and causal-X-to-Z output-difference diagnostics.

### 8. Narrow and differentiate the novelty claim

**Estimate:** 2--3 hours.

- [ ] Replace the broad claim that decomposition methods usually fuse before
      forecasting.
- [ ] Compare explicitly with TimeMixer, Koopa, ETSformer, N-BEATS, and N-HiTS.
- [ ] Define the contribution as the GP-coordinate/local-variate-field interface,
      process-specific modulation, late forecast combination, and its controlled
      evaluation.
- [ ] State which local-field components are inherited from VPNet and which
      routing/modulation/continuation components are new.
- [ ] Use "prefix-only" or "forecast-origin causal," not causality at every
      intermediate prefix time.
- [ ] Define "Flow" operationally so it is not confused with normalizing flows,
      flow matching, or continuous-time dynamics.

## P1: Only targeted new experiments

Run these only after the P0 audits pass. They are ordered by reviewer value per
unit of time.

### 9. Equal-stage deterministic and non-GP branch controls

**Estimate:** 6--10 GPU hours with parallel workers.

- [ ] Train a deterministic second-stage extension with the same reference,
      parameter/update budget, losses where applicable, and selector budget.
- [ ] Train a matched multi-branch non-GP control using deterministic learned
      filters or replicated inputs with the same branch count and width.
- [ ] Report both extension-only and validation-selected results.

This is the most useful new control for separating the value of GP coordinates
from extra stages, extra branches, and additional model selection.

### 10. Small comparator-fidelity audit

**Estimate:** 4--8 hours of setup plus queued GPU time.

- [ ] Verify official TimePro and VPNet forward passes, parameter counts, input
      conventions, and one-task numerical behavior against their repositories.
- [ ] Report actual tuning trials and GPU-hours for LatentFlow and comparators.
- [ ] If official implementations can be run without protocol distortion, tune
      only TimePro and VPNet with an equal validation-search budget.
- [ ] Otherwise label current entries as portable/paper-derived implementations
      and narrow the comparison claim. Do not present an unverified rushed port
      as an official reproduction.

### 11. Minimal order-robustness comparator

**Estimate:** 2--5 GPU hours.

- [ ] On Electricity and Traffic H=720, apply the same original/fixed-permutation
      protocol to LatentFlow, TimePro, and VPNet at seed 42.
- [ ] Preserve all channel metadata and permute inputs/targets consistently.
- [ ] Report retraining results and permutation-equivariance error separately.
- [ ] Put the 9.84% Electricity sensitivity in the main limitations regardless
      of whether comparators are run.

This does not fix order sensitivity, but prevents the paper from implying that
the limitation was hidden or unique without evidence.

## P1: Method and numerical specification

Complete alongside the paper rewrite.

- [ ] Separate modeled observation noise from numerical jitter in the ELBO and
      covariance equations.
- [ ] Give soft-membership simplex constraints and batch/channel/group indices.
- [ ] State tensor shapes, pooling axes, padding, stride, BN axes, FFN widths,
      activations, dropout order, and whether processes share BN statistics.
- [ ] Use `J=2` for field depth so `K` is reserved for covariance.
- [ ] State that the input-conditioned linear-bank gate makes the bank nonlinear
      as a whole, although its experts are affine.
- [ ] Specify Student-t heads, scale floors, degrees-of-freedom margin, units, and
      loss reduction; describe NLL only as an auxiliary point-training loss.
- [ ] Report latent scale, decoder/adapter norms, kernel amplitudes/noise,
      covariance condition numbers, negative-eigenvalue checks, and clamp rates.
- [ ] Validate posterior mean/variance formulas on a tiny joint Gaussian case.

## P2: Presentation and final QA

**Estimate:** 4--6 hours on Day 3.

- [ ] Shorten the title toward "Componentwise Nonlinear Continuation" unless the
      current title is rewritten to avoid an information-preservation theorem.
- [ ] Replace the duplicate aggregate figure with a 28-task gain heatmap.
- [ ] Add a selector validation-difference versus test-difference plot and regret
      histogram from existing audit data.
- [ ] Redraw the early/late figure with matched computation and the recovery
      identity/nonlinear example; remove overlapping labels.
- [ ] Use vector, colorblind-safe figures legible in grayscale.
- [ ] Repair appendix float order, orphaned headings, tiny tables, and visible
      hyperlink borders.
- [ ] Check every citation key, author/year/venue, protected acronym, equation
      reference, table number, and numerical claim.
- [ ] Compile with the official ICLR 2027 style on Overleaf and inspect every page
      at 100% zoom.
- [ ] Update the AI-assistance statement and submission form consistently.

## Work that does not fit this sprint

Do not rush these into the submission. Narrow the claims instead.

- [ ] Five-seed full-window training for LatentFlow and all competitors.
- [ ] A new-source benchmark from GIFT-Eval or Monash.
- [ ] A new permutation-equivariant architecture or ordering augmentation study.
- [ ] Full PatchTST/ModernTCN/Koopa/N-HiTS/ETSformer leaderboard expansion unless
      an already verified official pipeline is immediately available.
- [ ] Context-length sweeps, foundation-model panels, or a new synthetic suite.
- [ ] Probabilistic calibration, conformal prediction, or physical-process
      identification claims.

## Two-to-three-day schedule

### Day 1: Validity first

- [ ] Morning: freeze artifacts and complete the Traffic canonical replay.
- [ ] Midday: inspect and document early-fusion matching; launch only a required
      corrective rerun.
- [ ] Afternoon: rewrite theory, predictor definitions, title, abstract, and
      novelty boundary.
- [ ] Evening: generate the provenance manifest and table-consistency checks.

### Day 2: Evidence and targeted controls

- [ ] Morning: source-aware statistics, leave-one-source-out analysis, selector
      confusion/regret, prior-control intervals, MAE factorial table.
- [ ] In parallel: deterministic equal-stage and non-GP multibranch controls.
- [ ] Afternoon: comparator-fidelity audit and minimal order comparator if time
      remains.
- [ ] Evening: regenerate all tables and plots from the locked manifest.

### Day 3: Paper integration and release

- [ ] Integrate only passed evidence; prominently disclose full-window and
      channel-order limitations.
- [ ] Finish figures, supplement details, references, and reproducibility package.
- [ ] Overleaf compile, page-by-page visual review, anonymous artifact smoke test,
      and final numerical cross-check.

## Final go/no-go checklist

- [ ] Traffic canonical replay is internally consistent.
- [ ] Early-fusion control is precisely specified and materially matched.
- [ ] Information-loss claims have been replaced by correct scoped theory.
- [ ] `f_lat`, `f_mix`, and `f_sel` are distinct everywhere.
- [ ] Primary five-seed statistics use source-aware sensitivity analysis.
- [ ] Full-data and channel-order limitations are visible, not buried.
- [ ] Every table is generated from the same full-precision manifest.
- [ ] Anonymous code reproduces one smoke task and locates every reported result.

If either of the first two checks fails, the corresponding numerical claim must
be removed or weakened before submission rather than defended with an ambiguous
experiment.
