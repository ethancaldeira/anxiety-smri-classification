# ABIDE structural arm (Chapter 7)

This folder trains the Chapter 4 slice classifier (ImageNet ResNet50,
two-stage fine-tuning, 40 axial slices per subject, mean-probability
subject aggregation) on **ABIDE I** for autism (ASD) vs. control
classification. Its per-slice test predictions and trained model are the
ABIDE structural arm of the late-fusion pipeline in
[anxiety-multimodal-fusion](https://github.com/ethancaldeira/anxiety-multimodal-fusion),
alongside the ABIDE rs-fMRI arm from
[anxiety-fmri-classification](https://github.com/ethancaldeira/anxiety-fmri-classification).

The arm reuses `src/` unchanged: `abide/config.py` overrides the
dataset-specific attributes of `src.Config` at import time (participants
file, target column, slice directory, split file).

---

## Input: ANTs cortical thickness

The input is the ANTs cortical-thickness map of each subject
([Tustison et al., 2014][tustison2014]) from the
[Preprocessed Connectomes Project][pcp] release of ABIDE I
([Craddock et al., 2013][craddock2013]):

```
s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/ants/anat_thickness/<FILE_ID>_anat_thickness.nii.gz
```

The maps are spatially normalised to a common template by the PCP ANTs
pipeline: every map shares one 1 mm voxel grid (216 × 291 × 256 after RAS
reorientation), so a slice index corresponds to approximately the same
anatomy across subjects and no further structural preprocessing is applied.
The slices are therefore stored and read as the `raw` input type
(`EXPERIMENT_TYPE=raw`).

The PCP derivatives are used rather than a CAT12 run because ABIDE I
combines T1-weighted acquisitions from 17 sites with different scanners and
sequences ([Di Martino et al., 2014][dimartino2014]), the ANTs thickness
maps are already computed for the whole release, and ANTs cortical
thickness has good reproducibility across scanners and sites
([Tustison et al., 2014][tustison2014]). Cortical thickness and volume
measures give comparable discriminative performance in classification
([Schwarz et al., 2016][schwarz2016]), and other multimodal ABIDE studies
likewise combine structural features with functional connectivity
([Saponaro et al., 2024][saponaro2024]; [Rahman et al., 2026][rahman2026]).
The GSP structural arm uses CAT12 grey-matter density, so the two structural
arms use different feature types and their results are not compared
directly across datasets.

---

## Subjects and split

| | |
|---|---|
| Subjects | 774 ABIDE I subjects (379 ASD / 395 control): the anxiety-fmri-classification subject list (`data/abide/subjects_ABIDEI.txt`), all of which have an ANTs map |
| Target | `DX_Binary`: 1 = ASD (`DX_GROUP` 1), 0 = control (`DX_GROUP` 2) |
| Split | [`data_split_ABIDE_seed42.json`](data_split_ABIDE_seed42.json): stratified 65 / 15 / 20 subject split, seed 42 — train 503 (246 / 257), val 116 (57 / 59), test 155 (76 / 79) |

anxiety-fmri-classification trains its Chapter 7 ABIDE fMRI models on the same
split (`data/abide/abide_split_seed42.json`, the same subject IDs without
the `sub-` prefix), so both arms are evaluated on the same 155 test
subjects. `abide/config.py` uses this split file unless `SPLIT_FILE` is set.

---

## Layout

```
abide/
├── README.md
├── config.py                    # overrides src.Config for ABIDE
├── data_split_ABIDE_seed42.json # fixed train/val/test split (774 subjects)
├── download_ants.py             # ANTs maps + abide_participants.tsv
├── train.py                     # one run: python -m abide.train
└── run_abide.py                 # three-plane grid: python -m abide.run_abide exp1

data/ABIDE/                      # not tracked; populated by download_ants.py
├── Phenotypic_V1_0b_preprocessed1.csv   # PCP phenotype table
├── subjects_ABIDEI.txt                  # optional local copy of the subject list
├── abide_participants.tsv               # participant_id, DX_Binary, ...
├── raw/sub-<SUB_ID>.nii.gz              # ANTs cortical-thickness maps
└── slices/raw/{axial,coronal,sagittal}/sub-<SUB_ID>/*.png
```

---

## Reproduce

From the repository root:

```bash
# 1. ANTs maps and the participants table (public bucket, no AWS credentials;
#    the PCP phenotype CSV is downloaded too if it is not already present)
python abide/download_ants.py \
    --subjects ../anxiety-fmri-classification/data/abide/subjects_ABIDEI.txt

# 2. Axial slices (add --axis coronal / sagittal for run_abide exp1)
python preprocessing/scripts/extract_slices.py data/ABIDE/raw data/ABIDE/slices/raw \
    --axis axial --pattern 'sub-*.nii.gz'

# 3. Chapter 7 structural arm: ImageNet ResNet50, axial slices
EXPERIMENT_TYPE=raw VIEW_MODE=axial BASE_MODEL=ResNet50 \
    EXPERIMENT_NAME=ch7_abide_ants_axial_ResNet50 python -m abide.train
```

Step 3 writes `results/ch7_abide_ants_axial_ResNet50/` with the same files
as any other run (`predictions_slice.csv`, `best.keras`, metrics and
plots; see the main [README](../README.md#outputs)).
anxiety-multimodal-fusion reads `predictions_slice.csv` (fusion) and
`best.keras` (Grad-CAM) from this directory.

`python -m abide.run_abide exp1` trains the same model on each of the three
planes, writing `results/abide_exp1_raw_<view>_ResNet50/`.

---

## References

- Craddock, C. et al. (2013). The Neuro Bureau Preprocessing Initiative:
  open sharing of preprocessed neuroimaging data and derivatives.
  *Frontiers in Neuroinformatics*. [craddock2013][]
- Di Martino, A. et al. (2014). The autism brain imaging data exchange:
  towards a large-scale evaluation of the intrinsic brain architecture in
  autism. *Molecular Psychiatry*, 19(6), 659–667. [dimartino2014][]
- Tustison, N. J. et al. (2014). Large-scale evaluation of ANTs and
  FreeSurfer cortical thickness measurements. *NeuroImage*, 99, 166–179.
  [tustison2014][]
- Schwarz, C. G. et al. (2016). A large-scale comparison of cortical
  thickness and volume methods for measuring Alzheimer's disease severity.
  *NeuroImage: Clinical*, 11, 802–812. [schwarz2016][]
- Saponaro, S. et al. (2024). Deep learning based joint fusion approach to
  exploit anatomical and functional brain information in autism spectrum
  disorders. *Brain Informatics*, 11, 2. [saponaro2024][]
- Rahman, A., Shojaee-Mend, H. and Hatamikia, S. (2026). Multimodal
  Connectome Fusion via Cross-Attention for Autism Spectrum Disorder
  Classification Using Graph Learning. arXiv:2603.15168. [rahman2026][]

ABIDE I data are shared by the ABIDE consortium and the Preprocessed
Connectomes Project under their data use terms.

[pcp]: https://preprocessed-connectomes-project.org/abide/
[craddock2013]: https://preprocessed-connectomes-project.org/abide/
[dimartino2014]: https://doi.org/10.1038/mp.2013.78
[tustison2014]: https://doi.org/10.1016/j.neuroimage.2014.05.044
[schwarz2016]: https://doi.org/10.1016/j.nicl.2016.05.017
[saponaro2024]: https://doi.org/10.1186/s40708-023-00217-4
[rahman2026]: https://arxiv.org/abs/2603.15168
