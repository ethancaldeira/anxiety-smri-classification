"""
abide/run_abide.py — orchestrator for the ABIDE sMRI arm.

Mirrors the GSP `run_experiments.py` pattern: every cell is a fresh
`python -m abide.train` subprocess so TF state is not shared between runs.
Reuses the same EXPERIMENT_TYPE × VIEW_MODE × BASE_MODEL knobs the GSP
orchestrator uses; results land at `results/abide_<EXPERIMENT_NAME>/`
to keep them visually separated from GSP outputs in the same `results/`
directory.

Subcommand:

  exp1                     Three-cell axis grid (ResNet50) on raw ANTs cortical
                           thickness: raw × {axial, coronal, sagittal}.
                           Output dirs: abide_exp1_raw_<view>_ResNet50

Example:
    python -m abide.run_abide exp1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# subprocess helper
# ---------------------------------------------------------------------------

def run_one(name: str, env_overrides: dict[str, str]) -> tuple[str, int, float]:
    """Launch one abide.train cell as a subprocess. Returns (name, rc, seconds).

    Name is prefixed with `abide_` so result dirs don't collide with GSP.
    """
    full_name = name if name.startswith("abide_") else f"abide_{name}"
    env = {**os.environ, **env_overrides, "EXPERIMENT_NAME": full_name}
    print("=" * 70, flush=True)
    print(f"[run] starting {full_name}")
    print(f"      env: {env_overrides}")
    print("=" * 70, flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "abide.train"],
        env=env,
        cwd=str(REPO_ROOT),
    )
    dt = time.time() - t0
    print(f"[run] {full_name} finished rc={proc.returncode} in {dt/60:.1f} min",
          flush=True)
    return full_name, proc.returncode, dt


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def cmd_exp1() -> list[tuple[str, int, float]]:
    """Three-cell axis grid (ResNet50) at raw ANTs cortical thickness.

    Unlike the GSP orchestrator, there is no raw × preprocessed sweep: the
    ABIDE-PCP ANTs cortical-thickness maps are already spatially normalised
    to a common template, so the GSP "preprocessed" branch (CAT12
    segmentation and modulation) does not apply. They are read from the
    "raw" slice directory. See abide/README.md.
    """
    print(f"\n{'#' * 70}\n# EXP1 — axis grid at raw ANTs thickness "
          f"(3 runs, ResNet50)\n{'#' * 70}\n", flush=True)
    grid = [("raw", view, "ResNet50") for view in ("axial", "coronal", "sagittal")]
    results = []
    for etype, view, model in grid:
        name = f"exp1_{etype}_{view}_{model}"
        results.append(run_one(name, {
            "EXPERIMENT_TYPE": etype, "VIEW_MODE": view, "BASE_MODEL": model,
        }))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("exp1", help="3-cell raw × {axial, coronal, sagittal} grid")

    args = ap.parse_args()
    if args.cmd == "exp1":
        results = cmd_exp1()
    else:
        ap.error(f"unknown command: {args.cmd}")

    failed = [r for r in results if r[1] != 0]
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
