"""Dataset configurations for preprocessing pipeline.

Machine-specific locations come from environment variables:
  CAT12_DIR        CAT12 standalone (MCR) install (default: <repo>/CAT12.9_R2023b_MCR_Linux)
  MCR_DIR          MATLAB Runtime R2023b          (default: CAT12_DIR/v232/R2023b)
  CAT12_TPM        tissue probability map         (default: SPM's TPM.nii inside CAT12_DIR)
  GSP_RAW_DIR      raw GSP T1w volumes,
                   <subject>_T1w.nii.gz           (default: data/GSP/raw)
  GSP_PREPROC_DIR  CAT12 output directory         (default: data/GSP/preprocessed)
"""

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    raw_dir: Path
    preprocessed_dir: Path
    participants_file: Path
    tpm_path: Path
    age_group: str
    file_suffix: str

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CAT12_DIR = Path(os.environ.get("CAT12_DIR", BASE_DIR / "CAT12.9_R2023b_MCR_Linux"))
# MATLAB Runtime R2023b passed to CAT12's run_spm25.sh; the default is a
# Runtime installed inside the CAT12 directory.
MCR_DIR = Path(os.environ.get("MCR_DIR", CAT12_DIR / "v232" / "R2023b"))

# GSP dataset (Brain Genomics Superstruct Project)
GSP = DatasetConfig(
    name="GSP",
    raw_dir=Path(os.environ.get("GSP_RAW_DIR", DATA_DIR / "GSP" / "raw")),
    preprocessed_dir=Path(os.environ.get("GSP_PREPROC_DIR", DATA_DIR / "GSP" / "preprocessed")),
    participants_file=DATA_DIR / "GSP" / "GSP_participants.tsv",
    # SPM's default adult TPM (tpm/TPM.nii). The default is where the
    # CAT12.9_R2023b_MCR_Linux standalone release stores it (under spm25_mcr/,
    # at the path the release was built from); CAT12_TPM overrides it.
    tpm_path=Path(os.environ.get(
        "CAT12_TPM",
        CAT12_DIR / "spm25_mcr" / "home" / "ubuntu" / "Desktop" / "spm12" / "tpm" / "TPM.nii")),
    age_group="adult",
    file_suffix="_T1w.nii.gz"
)

# Registry
DATASETS = {
    "gsp": GSP
}
