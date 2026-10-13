"""
download_ants.py — data setup for the ABIDE sMRI arm.

Downloads the ANTs cortical-thickness maps of the ABIDE I subjects in the
subject list from the Preprocessed Connectomes Project (PCP) release on the
public fcp-indi S3 bucket (anonymous access, no AWS credentials needed),
saves them as data/ABIDE/raw/sub-<SUB_ID>.nii.gz, and builds
data/ABIDE/abide_participants.tsv with a `DX_Binary` column (1 = ASD,
0 = control) derived from the PCP `DX_GROUP`, which src/data_loader.py
reads as the target.

Inputs:
  --subjects   ABIDE I subject IDs, one per line (default:
               data/ABIDE/subjects_ABIDEI.txt; the 774-subject list is
               anxiety-fmri-classification/data/abide/subjects_ABIDEI.txt)
  --phenotype  PCP phenotype table (default:
               data/ABIDE/Phenotypic_V1_0b_preprocessed1.csv; downloaded
               from s3://fcp-indi/data/Projects/ABIDE_Initiative/ if absent)

Idempotent: files already on disk are skipped.

Usage:
    python abide/download_ants.py
    python abide/download_ants.py --subjects ../anxiety-fmri-classification/data/abide/subjects_ABIDEI.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
ABIDE_DATA_DIR = os.path.join(REPO_ROOT, 'data', 'ABIDE')
RAW_DIR = os.path.join(ABIDE_DATA_DIR, 'raw')

# Default inputs
PHENO_CSV       = os.path.join(ABIDE_DATA_DIR, 'Phenotypic_V1_0b_preprocessed1.csv')
SUBJECTS_TXT    = os.path.join(ABIDE_DATA_DIR, 'subjects_ABIDEI.txt')
# Output phenotype TSV read by src/data_loader.py (abide/config.py)
OUT_PHENO_TSV   = os.path.join(ABIDE_DATA_DIR, 'abide_participants.tsv')

# fcp-indi public bucket; no AWS creds required.
PCP_BUCKET = 'fcp-indi'
PCP_ROOT = 'data/Projects/ABIDE_Initiative'
PCP_PREFIX = f'{PCP_ROOT}/Outputs/ants/anat_thickness'
PCP_PHENO_KEY = f'{PCP_ROOT}/Phenotypic_V1_0b_preprocessed1.csv'


def _client():
    """Anonymous S3 client (PCP is a public Open Data bucket)."""
    return boto3.client('s3', config=BotoConfig(signature_version=UNSIGNED))


def _list_pcp_ants() -> dict[int, str]:
    """{7-digit subject id: filename} for every file in the ANTs prefix."""
    s3 = _client()
    paginator = s3.get_paginator('list_objects_v2')
    pattern = re.compile(r'(\d{7})_anat_thickness\.nii\.gz$')
    out: dict[int, str] = {}
    for page in paginator.paginate(Bucket=PCP_BUCKET, Prefix=f'{PCP_PREFIX}/'):
        for obj in page.get('Contents', []):
            fname = obj['Key'].rsplit('/', 1)[-1]
            m = pattern.search(fname)
            if m:
                out[int(m.group(1))] = fname
    return out


def _read_subjects(subjects_txt: str) -> list[int]:
    if not os.path.isfile(subjects_txt):
        sys.exit(
            f"ERROR: {subjects_txt} not found. Pass --subjects, e.g. "
            f"anxiety-fmri-classification/data/abide/subjects_ABIDEI.txt."
        )
    return [int(line.strip()) for line in open(subjects_txt) if line.strip()]


def _build_phenotype_tsv(have_ids: set[str], pheno_csv: str) -> None:
    if not os.path.isfile(pheno_csv):
        print(f"   downloading s3://{PCP_BUCKET}/{PCP_PHENO_KEY} -> {pheno_csv}")
        os.makedirs(os.path.dirname(pheno_csv) or '.', exist_ok=True)
        _client().download_file(PCP_BUCKET, PCP_PHENO_KEY, pheno_csv)
    df = pd.read_csv(pheno_csv)
    if 'SUB_ID' not in df.columns:
        sys.exit(f"ERROR: 'SUB_ID' column missing from {pheno_csv}; got {list(df.columns)[:10]}")
    if 'DX_GROUP' not in df.columns:
        sys.exit(f"ERROR: 'DX_GROUP' column missing from {pheno_csv}")

    # `sub-NNNNN` so it matches the ABIDE slice-dir glob (`sub-*`) in
    # src/data_loader.py. The fMRI repositories use the bare integer ID.
    df['participant_id'] = 'sub-' + df['SUB_ID'].astype(int).astype(str)
    # ABIDE convention: DX_GROUP 1 = ASD, 2 = Control. Map to binary.
    df['DX_Binary'] = (df['DX_GROUP'] == 1).astype(int)

    # Filter to subjects we actually have thickness for + that are in our
    # 774-subject list.
    df = df[df['participant_id'].isin(have_ids)].copy()
    df = df[['participant_id', 'SUB_ID', 'DX_GROUP', 'DX_Binary',
             'AGE_AT_SCAN', 'SEX', 'SITE_ID']].sort_values('participant_id')

    df.to_csv(OUT_PHENO_TSV, sep='\t', index=False)
    n_pos = int((df.DX_Binary == 1).sum())
    n_neg = int((df.DX_Binary == 0).sum())
    print(f"  wrote {OUT_PHENO_TSV}: {len(df)} subjects "
          f"({n_pos} ASD / {n_neg} Control, {100*n_pos/max(1,len(df)):.1f}% ASD)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', default=SUBJECTS_TXT,
                    help='ABIDE I subject IDs, one per line (default: %(default)s)')
    ap.add_argument('--phenotype', default=PHENO_CSV,
                    help='PCP phenotype CSV; downloaded if absent (default: %(default)s)')
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"== ABIDE ANTs cortical thickness download ==")
    print(f"   target dir: {RAW_DIR}")

    want = _read_subjects(args.subjects)
    print(f"   subject list: {len(want)} entries (from {args.subjects})")

    print(f"   listing s3://{PCP_BUCKET}/{PCP_PREFIX}/ ...")
    pcp_files = _list_pcp_ants()
    print(f"   PCP ANTs files available: {len(pcp_files)}")

    matched = [(sid, pcp_files[sid]) for sid in want if sid in pcp_files]
    missing = [sid for sid in want if sid not in pcp_files]
    print(f"   matched: {len(matched)}/{len(want)}; missing: {len(missing)}")
    if missing:
        print(f"   first 10 missing: {missing[:10]}")

    s3 = _client()
    n_dl = 0
    n_skip = 0
    for sid, fname in matched:
        # Save as `sub-NNNNN.nii.gz` so the downstream slice extraction
        # produces `slices/raw/<axis>/sub-NNNNN/` subject dirs that match
        # the `sub-*` glob in src/data_loader.py without renaming.
        local = os.path.join(RAW_DIR, f'sub-{sid}.nii.gz')
        if os.path.isfile(local) and os.path.getsize(local) > 0:
            n_skip += 1
            continue
        s3.download_file(PCP_BUCKET, f'{PCP_PREFIX}/{fname}', local)
        n_dl += 1
        if n_dl % 50 == 0:
            print(f"   downloaded {n_dl} ...")
    print(f"   downloaded: {n_dl}  skipped (already on disk): {n_skip}")

    _build_phenotype_tsv({f'sub-{sid}' for sid, _ in matched}, args.phenotype)
    print("== done ==")


if __name__ == '__main__':
    main()
