"""Run every zero-training MVPF paper analysis available from local artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = (
    "statistical_analysis.py",
    "ablation_analysis.py",
    "artifact_audit.py",
    "svg_figures.py",
    "paper_summary.py",
)


def main() -> None:
    for script in SCRIPTS:
        print(f"\n===== {script} =====", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True)
    print("\n[mvpf-tests] complete", flush=True)


if __name__ == "__main__":
    main()

