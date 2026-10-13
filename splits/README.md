# GSP subject subset and split

The Chapter 4 experiments use a fixed 200-subject subset of the GSP and a
fixed subject-level train/val/test split of that subset. Neither list is
distributed here; `make_gsp_split.py` regenerates both from the GSP
participants table.

```bash
python splits/make_gsp_split.py                      # reads data/GSP/GSP_participants.tsv
python splits/make_gsp_split.py --participants /path/to/GSP_participants.tsv
```

| Output | Contents |
|---|---|
| `data/GSP/GSP_subject_list_200.txt` | the 200 subject IDs, one per line |
| `data/data_split_GSP_seed42.json` | train / val / test subject IDs of those 200 (130 / 29 / 41) |

Both steps are stratified on the binary label STAI-T ≥ 36 with seed 42, and
subject IDs are sorted before sampling, so the output depends only on the
participants table:

| | Subjects | STAI-T ≥ 36 | STAI-T < 36 |
|---|---|---|---|
| Cohort (STAI-T available) | 879 | 456 | 423 |
| Subset | 200 | 104 | 96 |
| Train | 130 | 68 | 62 |
| Val | 29 | 15 | 14 |
| Test | 41 | 21 | 20 |

The script prints these counts. `--data-dir DIR` writes the two files under
`DIR` instead of `data/`.

The data loader reads both files from `data/`. The 200-subject list is
required while `USE_GSP_SUBSET=1` (the default), and every single-split run
on the subset reuses `data/data_split_GSP_seed42.json`. The sex-stratified
splits (`data/data_split_GSP_{M,F}_seed42.json`) and the 5-fold
cross-validation folds are derived from the same subset by the data loader
at run time, with the same seed.
