#!/usr/bin/env bash
# Reproduce the Chapter 4 runs, tables and figures, and the Chapter 7
# structural (sMRI) arms. Run from anywhere; paths are relative to the repo.
#
# Usage:
#   scripts/reproduce_ch4.sh [STEP ...]
#   DRY_RUN=1 scripts/reproduce_ch4.sh all      # print the commands only
#
# Steps, in pipeline order:
#   preprocess  CAT12 on every GSP subject       (needs CAT12_DIR, GSP_RAW_DIR, GSP_PREPROC_DIR)
#   slices      GSP PNG slices, raw + CAT12, axial / coronal / sagittal
#   split       200-subject GSP subset and its 130/29/41 split (splits/make_gsp_split.py)
#   validate    sex-classification pipeline check            -> results/validate_sex
#   exp1        raw vs preprocessed x 3 planes, ImageNet ResNet50 -> results/exp1_*
#   exp2        3 backbones x {ImageNet, RadImageNet}, preprocessed axial -> results/exp2_*
#   exp3        5-fold CV, ImageNet ResNet50, preprocessed axial -> results/exp3_kfold_*
#   sex_strat   male-only / female-only models, raw sagittal -> results/strat_*
#   tables      Tables 4.1-4.7 and Figures 4.4-4.5           -> $TABLES_DIR
#   ch7_gsp     Ch7 GSP sMRI arm: all subjects on the anxiety-fmri-classification split
#   abide_data  ABIDE ANTs cortical-thickness maps + axial slices
#   ch7_abide   Ch7 ABIDE sMRI arm (ANTs cortical thickness, axial)
#   all         every step above
# With no STEP: split validate exp1 exp2 exp3 sex_strat tables
#
# Environment:
#   PYTHON           Python interpreter                   (default: python)
#   CAT12_DIR, MCR_DIR, CAT12_TPM, GSP_RAW_DIR, GSP_PREPROC_DIR
#                    CAT12 install, MATLAB Runtime and GSP volumes
#                    (see preprocessing/scripts/config_datasets.py);
#                    GSP_RAW_DIR and GSP_PREPROC_DIR are also read by 'slices'
#   RADIMAGENET_DIR  RadImageNet no-top .h5 weights       (default: pretrained_weights/radimagenet)
#   FMRI_REPO        anxiety-fmri-classification checkout (default: ../anxiety-fmri-classification)
#   GSP_SPLIT_FILE   GSP split for ch7_gsp                (default: $FMRI_REPO/data/gsp_split.json)
#   ABIDE_SUBJECTS   ABIDE I subject list for abide_data  (default: $FMRI_REPO/data/abide/subjects_ABIDEI.txt)
#   TABLES_DIR       output of the tables step            (default: results/ch4_tables)
#   DRY_RUN=1        print the commands without running them
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PYTHON:-python}"
FMRI_REPO="${FMRI_REPO:-$REPO/../anxiety-fmri-classification}"
GSP_SPLIT_FILE="${GSP_SPLIT_FILE:-$FMRI_REPO/data/gsp_split.json}"
ABIDE_SUBJECTS="${ABIDE_SUBJECTS:-$FMRI_REPO/data/abide/subjects_ABIDEI.txt}"
TABLES_DIR="${TABLES_DIR:-results/ch4_tables}"
VIEWS=(axial coronal sagittal)

# Start every run from the src/config.py defaults; each step sets what it varies.
unset EXPERIMENT_NAME EXPERIMENT_TYPE VIEW_MODE BASE_MODEL USE_RADIMAGENET \
      USE_GSP_SUBSET SPLIT_FILE SEX_FILTER TARGET_OVERRIDE LABEL_PERMUTE \
      USE_KFOLD N_FOLDS STAI_CUTOFF STAGE1_LR USE_AUGMENTATION TRAIN_SEED DETERMINISTIC_OPS

run() {
    echo "+ $*"
    if [[ -z "${DRY_RUN:-}" ]]; then "$@"; fi
}

need_file() {
    if [[ -z "${DRY_RUN:-}" && ! -f "$1" ]]; then
        echo "missing $1 ($2)" >&2; exit 1
    fi
}

step_preprocess() {
    run "$PY" preprocessing/scripts/pipeline.py --dataset gsp --all --parallel 8
}

step_slices() {
    local raw_dir="${GSP_RAW_DIR:-}" preproc_dir="${GSP_PREPROC_DIR:-}"
    if [[ -n "${DRY_RUN:-}" ]]; then
        raw_dir="${raw_dir:-<GSP_RAW_DIR>}"
        preproc_dir="${preproc_dir:-<GSP_PREPROC_DIR>}"
    fi
    if [[ -z "$raw_dir" ]]; then
        echo "set GSP_RAW_DIR to the raw GSP T1w volumes" >&2; exit 1
    fi
    if [[ -z "$preproc_dir" ]]; then
        echo "set GSP_PREPROC_DIR to the CAT12 output directory" >&2; exit 1
    fi
    for view in "${VIEWS[@]}"; do
        run "$PY" preprocessing/scripts/extract_slices.py "$raw_dir" data/GSP/slices/raw \
            --axis "$view" --pattern '*_T1w.nii.gz'
        run "$PY" preprocessing/scripts/extract_slices.py "$preproc_dir" data/GSP/slices/preprocessed \
            --axis "$view" --pattern '*/mri/mwp1*.nii'
    done
}

step_split() {
    run "$PY" splits/make_gsp_split.py
}

step_validate() {
    run env EXPERIMENT_NAME=validate_sex EXPERIMENT_TYPE=raw VIEW_MODE=sagittal BASE_MODEL=ResNet50 \
        TARGET_OVERRIDE=Sex "$PY" src/train.py
}

step_exp1() {
    run "$PY" run_experiments.py exp1
}

step_exp2() {
    run "$PY" run_experiments.py exp2 --type preprocessed --view axial
}

step_exp3() {
    run "$PY" run_experiments.py exp3 --type preprocessed --view axial --model ResNet50 --weights ImageNet
}

step_sex_strat() {
    run "$PY" run_experiments.py sex_strat --type raw --view sagittal
}

step_tables() {
    run "$PY" scripts/make_ch4_tables.py --results results --out "$TABLES_DIR"
    run "$PY" scripts/make_ch4_figures.py --run results/exp2_preprocessed_axial_ImageNet_ResNet50 \
        --out "$TABLES_DIR"
}

step_ch7_gsp() {
    need_file "$GSP_SPLIT_FILE" "generated by \`python scripts/make_gsp_inputs.py --participants /path/to/GSP_participants.tsv\` in anxiety-fmri-classification; or set GSP_SPLIT_FILE"
    run env USE_GSP_SUBSET=0 SPLIT_FILE="$GSP_SPLIT_FILE" \
        EXPERIMENT_TYPE=preprocessed VIEW_MODE=axial BASE_MODEL=ResNet50 \
        EXPERIMENT_NAME=ch7_gsp_full_preprocessed_axial_ResNet50 "$PY" src/train.py
}

step_abide_data() {
    need_file "$ABIDE_SUBJECTS" "set ABIDE_SUBJECTS to the ABIDE I subject list"
    run "$PY" abide/download_ants.py --subjects "$ABIDE_SUBJECTS"
    run "$PY" preprocessing/scripts/extract_slices.py data/ABIDE/raw data/ABIDE/slices/raw \
        --axis axial --pattern 'sub-*.nii.gz'
}

step_ch7_abide() {
    run env EXPERIMENT_TYPE=raw VIEW_MODE=axial BASE_MODEL=ResNet50 \
        EXPERIMENT_NAME=ch7_abide_ants_axial_ResNet50 "$PY" -m abide.train
}

ALL_STEPS=(preprocess slices split validate exp1 exp2 exp3 sex_strat tables ch7_gsp abide_data ch7_abide)
DEFAULT_STEPS=(split validate exp1 exp2 exp3 sex_strat tables)

if [[ $# -eq 0 ]]; then
    steps=("${DEFAULT_STEPS[@]}")
elif [[ "$1" == "all" ]]; then
    steps=("${ALL_STEPS[@]}")
else
    steps=("$@")
fi

for s in "${steps[@]}"; do
    if ! declare -F "step_$s" > /dev/null; then
        echo "unknown step: $s (steps: ${ALL_STEPS[*]} all)" >&2
        exit 2
    fi
done

for s in "${steps[@]}"; do
    echo "=== $s ($(date '+%F %T'))"
    "step_$s"
done
echo "=== done ($(date '+%F %T'))"
