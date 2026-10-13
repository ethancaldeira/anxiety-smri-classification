"""
run_experiments.py — single orchestrator for every kind of run in this repo.

Five subcommands:

  validate                 Two pipeline-validation runs at the hardcoded
                           raw-sagittal-ResNet50 cell:
                             validate_sex     — positive control (predict sex
                                                from T1; literature AUC > 0.9)
                             validate_permute — negative control (subject-level
                                                anxiety-label shuffle; AUC ≈ 0.5)

  exp1                     Six-cell {raw, preprocessed} × {axial, coronal,
                           sagittal} ResNet50 grid. Output dirs:
                             exp1_<type>_<view>_ResNet50

  exp2 --type T --view V   Weights × backbone grid at a given cell: 2 sources
                           (ImageNet, RadImageNet) × 3 backbones (ResNet50,
                           DenseNet121, InceptionV3) = 6 runs. Output dirs:
                             exp2_<T>_<V>_<weights>_<backbone>

  exp3 --type T --view V   5-fold CV at a given cell (USE_KFOLD=1). Defaults
       [--model M]         to ImageNet ResNet50; override with --model/--weights.
       [--weights W]       Output: exp3_kfold_<T>_<V>_<weights>_<model>

  sex_strat [--type T]     Sex-specific anxiety models at a given cell
            [--view V]     (default raw sagittal ResNet50): male-only and
                           female-only, single split and 5-fold CV. Output:
                             strat_{male,female}_single, strat_{male,female}

  all                      validate → exp1 → exp2 at the exp1 cell with the
                           highest subject-level test AUC → exp3 at the
                           highest-AUC configuration across exp1 + exp2.
                           Automatic selection; the thesis runs use a fixed
                           configuration instead (scripts/reproduce_ch4.sh).

Every cell is a fresh `python src/train.py` subprocess so TF state doesn't
bleed between runs. train.py auto-runs the full evaluation plot set at the
end of each fit.

Examples:
    python run_experiments.py validate
    python run_experiments.py exp1
    python run_experiments.py exp2 --type preprocessed --view axial
    python run_experiments.py exp3 --type preprocessed --view axial --model ResNet50 --weights ImageNet
    python run_experiments.py sex_strat
    python run_experiments.py all --skip-validate  # if validation already done
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "src" / "train.py"
RESULTS_ROOT = REPO_ROOT / "results"

# Fixed cell for the validation runs, so the positive and negative controls
# are directly comparable with each other.
VALIDATE_CELL = {"EXPERIMENT_TYPE": "raw", "VIEW_MODE": "sagittal", "BASE_MODEL": "ResNet50"}


# ---------------------------------------------------------------------------
# subprocess helper
# ---------------------------------------------------------------------------

def run_one(name: str, env_overrides: dict[str, str]) -> tuple[str, int, float]:
    """Launch one train.py cell as a subprocess. Returns (name, rc, seconds)."""
    env = {**os.environ, **env_overrides, "EXPERIMENT_NAME": name}
    print("=" * 70, flush=True)
    print(f"[run] starting {name}")
    print(f"      env: {env_overrides}")
    print("=" * 70, flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        env=env,
        cwd=str(REPO_ROOT),
    )
    dt = time.time() - t0
    print(f"[run] {name} finished rc={proc.returncode} in {dt/60:.1f} min", flush=True)
    return name, proc.returncode, dt


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def cmd_validate() -> list[tuple[str, int, float]]:
    """Positive + negative pipeline-validation controls."""
    print(f"\n{'#' * 70}\n# VALIDATE — positive (sex) + negative (permute) controls "
          f"at {VALIDATE_CELL['EXPERIMENT_TYPE']} {VALIDATE_CELL['VIEW_MODE']} "
          f"{VALIDATE_CELL['BASE_MODEL']} (2 runs)\n{'#' * 70}\n", flush=True)
    results = []
    results.append(run_one("validate_sex",     {**VALIDATE_CELL, "TARGET_OVERRIDE": "Sex"}))
    results.append(run_one("validate_permute", {**VALIDATE_CELL, "LABEL_PERMUTE": "1"}))
    return results


def cmd_exp1() -> list[tuple[str, int, float]]:
    """Six-cell orientation × preprocessing grid."""
    print(f"\n{'#' * 70}\n# EXP1 — orientation × preprocessing grid (6 runs, ResNet50)\n{'#' * 70}\n", flush=True)
    grid = [
        (etype, view, "ResNet50")
        for etype, view in product(["raw", "preprocessed"], ["axial", "coronal", "sagittal"])
    ]
    results = []
    for etype, view, model in grid:
        name = f"exp1_{etype}_{view}_{model}"
        results.append(run_one(name, {
            "EXPERIMENT_TYPE": etype, "VIEW_MODE": view, "BASE_MODEL": model,
        }))
    return results


def pick_winning_cell() -> tuple[str, str, float]:
    """Read metrics_subject.json from each exp1 dir, return (type, view, auc)."""
    best: tuple[str | None, str | None, float] = (None, None, -1.0)
    for d in sorted(RESULTS_ROOT.glob("exp1_*_ResNet50")):
        m = d / "metrics_subject.json"
        if not m.is_file():
            continue
        auc = json.loads(m.read_text()).get("roc_auc")
        if auc is None or auc != auc:        # NaN check
            continue
        parts = d.name.split("_")             # exp1_<type>_<view>_ResNet50
        etype, view = parts[1], parts[2]
        if auc > best[2]:
            best = (etype, view, auc)
    if best[0] is None:
        sys.exit("[run] could not pick a winning cell — no exp1 metrics found")
    print(f"\n[run] winning exp1 cell: {best[0]} {best[1]}  (subject AUC = {best[2]:.3f})\n", flush=True)
    return best  # type: ignore[return-value]


EXP2_MODELS = ("ResNet50", "DenseNet121", "InceptionV3")
EXP2_WEIGHTS = (("ImageNet", "0"), ("RadImageNet", "1"))  # (label, USE_RADIMAGENET)


def cmd_exp2(etype: str, view: str) -> list[tuple[str, int, float]]:
    """Pretraining-source × backbone grid at a given cell.

    2 weight sources (ImageNet, RadImageNet) × 3 backbones (ResNet50,
    DenseNet121, InceptionV3) = 6 runs. RadImageNet weights exist only for
    these three backbones (config.RADIMAGENET_WEIGHTS_MAP).
    """
    print(f"\n{'#' * 70}\n# EXP2 — weights × backbone grid at {etype} {view} (6 runs)\n{'#' * 70}\n", flush=True)
    results = []
    for wlabel, use_rad in EXP2_WEIGHTS:
        for bb in EXP2_MODELS:
            name = f"exp2_{etype}_{view}_{wlabel}_{bb}"
            results.append(run_one(name, {
                "EXPERIMENT_TYPE": etype, "VIEW_MODE": view, "BASE_MODEL": bb,
                "USE_RADIMAGENET": use_rad,
            }))
    return results


def pick_overall_winner() -> dict[str, str]:
    """Highest subject-AUC config across exp1 + exp2 (backbone AND weights).

    exp1 dirs:  exp1_<type>_<view>_<model>            (always ImageNet ResNet50)
    exp2 dirs:  exp2_<type>_<view>_<weights>_<model>  (weights in {ImageNet,RadImageNet})
    Returns an env-override dict: EXPERIMENT_TYPE, VIEW_MODE, BASE_MODEL, USE_RADIMAGENET.
    """
    best_auc = -1.0
    best: dict[str, str] | None = None
    best_name = ""
    for d in sorted(list(RESULTS_ROOT.glob("exp1_*")) + list(RESULTS_ROOT.glob("exp2_*"))):
        m = d / "metrics_subject.json"
        if not m.is_file():
            continue
        auc = json.loads(m.read_text()).get("roc_auc")
        if auc is None or auc != auc:          # None / NaN
            continue
        parts = d.name.split("_")
        if parts[0] == "exp1":                 # exp1_<type>_<view>_<model>
            etype, view, model, use_rad = parts[1], parts[2], parts[3], "0"
        else:                                  # exp2_<type>_<view>_<weights>_<model>
            etype, view, weights, model = parts[1], parts[2], parts[3], parts[4]
            use_rad = "1" if weights == "RadImageNet" else "0"
        if auc > best_auc:
            best_auc = auc
            best_name = d.name
            best = {"EXPERIMENT_TYPE": etype, "VIEW_MODE": view,
                    "BASE_MODEL": model, "USE_RADIMAGENET": use_rad}
    if best is None:
        sys.exit("[run] could not pick an overall winner — no exp1/exp2 metrics found")
    print(f"\n[run] overall winner: {best_name}  (subject AUC = {best_auc:.3f})\n", flush=True)
    return best


def cmd_exp3(cell: dict[str, str]) -> list[tuple[str, int, float]]:
    """5-fold CV at a fully-specified cell (type, view, model, weights)."""
    etype, view = cell["EXPERIMENT_TYPE"], cell["VIEW_MODE"]
    model = cell["BASE_MODEL"]
    wlabel = "RadImageNet" if cell.get("USE_RADIMAGENET") == "1" else "ImageNet"
    print(f"\n{'#' * 70}\n# EXP3 — 5-fold CV at {etype} {view} {wlabel} {model} (1 run, 5 folds)\n{'#' * 70}\n", flush=True)
    name = f"exp3_kfold_{etype}_{view}_{wlabel}_{model}"
    return [run_one(name, {**cell, "USE_KFOLD": "1", "N_FOLDS": "5"})]


def cmd_sex_strat(etype: str, view: str) -> list[tuple[str, int, float]]:
    """Male-only and female-only anxiety models, single split and 5-fold CV."""
    print(f"\n{'#' * 70}\n# SEX-STRATIFIED — {etype} {view} ResNet50 (4 runs)\n{'#' * 70}\n", flush=True)
    cell = {"EXPERIMENT_TYPE": etype, "VIEW_MODE": view, "BASE_MODEL": "ResNet50"}
    results = []
    for sex, label in (("M", "male"), ("F", "female")):
        results.append(run_one(f"strat_{label}_single", {**cell, "SEX_FILTER": sex}))
        results.append(run_one(f"strat_{label}", {**cell, "SEX_FILTER": sex,
                                                  "USE_KFOLD": "1", "N_FOLDS": "5"}))
    return results


def cmd_all(args) -> int:
    """validate → exp1 → pick winner → exp2 → exp3."""
    t_total = time.time()
    all_results: list[tuple[str, int, float]] = []

    if not args.skip_validate:
        all_results.extend(cmd_validate())
    if not args.skip_exp1:
        all_results.extend(cmd_exp1())

    etype, view, auc = pick_winning_cell()
    all_results.extend(cmd_exp2(etype, view))
    winner = pick_overall_winner()             # best of exp1 + exp2 (model + weights)
    all_results.extend(cmd_exp3(winner))

    total_min = (time.time() - t_total) / 60
    print("\n" + "=" * 70, flush=True)
    print(f"[run] ALL DONE. {len(all_results)} runs in {total_min:.1f} min total")
    print(f"      exp1 winning cell: {etype} {view}  (subj AUC = {auc:.3f})")
    print(f"      exp3 ran on overall winner: {winner}")
    print("=" * 70)
    for name, rc, dt in all_results:
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"  [{status}] {name}  ({dt/60:.1f} min)")
    return 0 if all(r[1] == 0 for r in all_results) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="run positive (sex) + negative (permute) controls")

    sub.add_parser("exp1", help="six-cell {raw,preprocessed} × {axial,coronal,sagittal} grid")

    ap2 = sub.add_parser("exp2", help="weights × backbone grid (2x3=6 runs) at a given cell")
    ap2.add_argument("--type", required=True, choices=["raw", "preprocessed"])
    ap2.add_argument("--view", required=True, choices=["axial", "coronal", "sagittal"])

    ap3 = sub.add_parser("exp3", help="5-fold CV at a given cell")
    ap3.add_argument("--type", required=True, choices=["raw", "preprocessed"])
    ap3.add_argument("--view", required=True, choices=["axial", "coronal", "sagittal"])
    ap3.add_argument("--model", default="ResNet50",
                     choices=["ResNet50", "DenseNet121", "InceptionV3", "VGG16", "VGG19"])
    ap3.add_argument("--weights", default="ImageNet", choices=["ImageNet", "RadImageNet"])

    aps = sub.add_parser("sex_strat", help="male-only / female-only models (single split + 5-fold CV)")
    aps.add_argument("--type", default="raw", choices=["raw", "preprocessed"])
    aps.add_argument("--view", default="sagittal", choices=["axial", "coronal", "sagittal"])

    apa = sub.add_parser("all", help="validate → exp1 → pick winner → exp2 → exp3")
    apa.add_argument("--skip-validate", action="store_true")
    apa.add_argument("--skip-exp1",     action="store_true")

    args = ap.parse_args()

    if args.cmd == "validate":
        results = cmd_validate()
    elif args.cmd == "exp1":
        results = cmd_exp1()
    elif args.cmd == "exp2":
        results = cmd_exp2(args.type, args.view)
    elif args.cmd == "exp3":
        results = cmd_exp3({
            "EXPERIMENT_TYPE": args.type, "VIEW_MODE": args.view,
            "BASE_MODEL": args.model,
            "USE_RADIMAGENET": "1" if args.weights == "RadImageNet" else "0",
        })
    elif args.cmd == "sex_strat":
        results = cmd_sex_strat(args.type, args.view)
    elif args.cmd == "all":
        sys.exit(cmd_all(args))
    else:
        ap.error(f"unknown command: {args.cmd}")

    failed = [r for r in results if r[1] != 0]
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
