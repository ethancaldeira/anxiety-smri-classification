"""
Build the GSP 200-subject subset and its train/val/test split.

Both steps are stratified on the binary label STAI_tAnxiety >= STAI_CUTOFF,
so the subset and every partition keep the cohort's class balance.

  1. Subset: stratified random sample of N_SUBJECTS from all participants
     with a STAI-T score.
  2. Split:  stratified 65/15/20 train/val/test (same proportions and seed
     as src/config.py).

Subject IDs are sorted before sampling, so the output depends only on the
participants file, the cutoff and the seed. With the GSP release phenotype
table (879 subjects with a STAI-T score) this gives 104 high / 96 low
subjects, split 130 / 29 / 41 (test: 21 high / 20 low).

Outputs, at the locations src/config.py reads them from:
    <data-dir>/GSP/GSP_subject_list_200.txt     one subject ID per line
    <data-dir>/data_split_GSP_seed42.json       train / val / test subject IDs

Usage (from the repo root):
    python splits/make_gsp_split.py
    python splits/make_gsp_split.py --participants /path/to/GSP_participants.tsv
    python splits/make_gsp_split.py --data-dir /tmp/check   # write elsewhere
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

N_SUBJECTS = 200
STAI_CUTOFF = 36
SEED = 42
TRAIN, VAL, TEST = 0.65, 0.15, 0.20


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', default=os.path.join(REPO, 'data'),
                    help='Root the two output files are written under (default: data/)')
    ap.add_argument('--participants', default=None,
                    help='GSP participants TSV (default: <data-dir>/GSP/GSP_participants.tsv)')
    args = ap.parse_args()
    participants = args.participants or os.path.join(args.data_dir, 'GSP', 'GSP_participants.tsv')

    df = pd.read_csv(participants, sep='\t')
    df['STAI_tAnxiety'] = pd.to_numeric(df['STAI_tAnxiety'], errors='coerce')
    df = df.dropna(subset=['STAI_tAnxiety']).sort_values('participant_id')
    ids = df['participant_id'].to_numpy()
    labels = (df['STAI_tAnxiety'] >= STAI_CUTOFF).astype(int).to_numpy()

    subset, _, subset_labels, _ = train_test_split(
        ids, labels, train_size=N_SUBJECTS, stratify=labels, random_state=SEED)
    order = subset.argsort()
    subset, subset_labels = subset[order], subset_labels[order]

    train, temp, _, temp_labels = train_test_split(
        subset, subset_labels, test_size=VAL + TEST,
        stratify=subset_labels, random_state=SEED)
    val, test = train_test_split(
        temp, test_size=TEST / (VAL + TEST), stratify=temp_labels, random_state=SEED)

    list_path = os.path.join(args.data_dir, 'GSP', 'GSP_subject_list_200.txt')
    split_path = os.path.join(args.data_dir, 'data_split_GSP_seed42.json')
    os.makedirs(os.path.dirname(list_path), exist_ok=True)

    with open(list_path, 'w') as f:
        f.write('\n'.join(subset) + '\n')
    split = {
        'train': sorted(train.tolist()),
        'val': sorted(val.tolist()),
        'test': sorted(test.tolist()),
        'random_seed': SEED,
        'dataset': 'GSP',
        'stai_cutoff': STAI_CUTOFF,
    }
    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)

    label = dict(zip(ids, labels))
    print(f"cohort: {len(ids)} subjects, {labels.sum()} with STAI-T >= {STAI_CUTOFF}")
    for name, part in (('subset', subset), ('train', train), ('val', val), ('test', test)):
        n_pos = sum(label[s] for s in part)
        print(f"{name:>6}: {len(part):3d} subjects, {n_pos} high / {len(part) - n_pos} low")
    print(f"wrote {list_path}\nwrote {split_path}")


if __name__ == '__main__':
    main()
