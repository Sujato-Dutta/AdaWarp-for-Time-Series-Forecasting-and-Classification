# Dataset audit

Generate the complete variable-level audit from the repository's local benchmark CSV files:

```powershell
.\.venv-awp\Scripts\python.exe dataset_audit\audit.py
```

The command writes:

- `dataset_audit.tex`: standalone Overleaf entry point;
- `figures/`: full-series, representative-window, correlation, and channel-profile plots;
- `statistics/`: machine-readable variable, lead/lag, and cluster statistics;
- `tables/`: complete Electricity and Traffic per-channel LaTeX appendices.

The report excludes Weather's `-9999` missing-value sentinel and documents every derived threshold and lag convention. Upload the complete `dataset_audit/` directory to Overleaf so that the relative figure and table paths remain valid.
