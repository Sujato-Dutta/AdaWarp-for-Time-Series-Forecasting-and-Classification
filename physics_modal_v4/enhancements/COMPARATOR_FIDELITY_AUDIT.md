# Comparator Fidelity Audit

## Submission-safe interpretation

The benchmark contains 28 seed-42 results for every retained comparator. These are matched reruns under the shared data and validation protocol; they are not a table of published scores and should not collectively be described as official-source reproductions.

| Comparator | Implementation used | Defensible provenance statement | Result coverage |
|---|---|---|---:|
| DLinear | `TSLibrary/models/DLinear.py` through a four-argument adapter | Repository TSLibrary implementation | 28/28 |
| iTransformer | `TSLibrary/models/iTransformer.py` through a four-argument adapter | Repository TSLibrary implementation | 28/28 |
| TimeMixer | `TSLibrary/models/TimeMixer.py` through a four-argument adapter | Repository TSLibrary implementation | 28/28 |
| VPNet | Established repository-native matched `VPNetForecaster` used by the prior MVPF evaluation | Repository-native matched implementation | 28/28 |
| TimePro | Portable PyTorch implementation of the released hyper-state design; custom selective-scan/DCNv4 CUDA operations are represented with ordinary PyTorch operations | Portable reproduction associated with official commit `70a20e5`, not bitwise execution of the released CUDA stack | 28/28 |
| xCPD | DLinear host plus an implementation of the paper's graph-spectral plug-in equations | Paper-derived implementation; the named repository did not expose the plug-in source when the benchmark was frozen | 28/28 |

## Required manuscript wording

Use “matched reruns under a common data and selection protocol.” State the TimePro and xCPD qualifications explicitly. Do not use “official implementation” for the comparison suite as a whole. Shared data and validation checkpointing do not imply identical optimizers, epoch counts, or update budgets; those model-specific settings must remain in the appendix.

## Remaining fidelity work

A one-task numerical comparison against an upstream TimePro/VPNet environment remains optional and requires the corresponding external repositories and dependencies. It is not a prerequisite for reporting the current benchmark if the provenance language above is retained.