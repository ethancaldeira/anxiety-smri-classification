"""
Build the Chapter 4 tables (4.1-4.7 and the sex-stratified results) from the
run directories in results/.

Every table is printed with the files it was computed from and written to
<out>/ch4_tables.txt; the per-slice metrics behind Table 4.6 (all slice
positions) go to <out>/table_4_6_all_slices.csv.

Runs read (names as written by run_experiments.py / scripts/reproduce_ch4.sh):
  4.1        validate_sex
  4.2        exp1_{raw,preprocessed}_{axial,sagittal,coronal}_ResNet50
  4.3, 4.4   exp2_preprocessed_axial_{ImageNet,RadImageNet}_{ResNet50,DenseNet121,InceptionV3}
  4.5        exp3_kfold_preprocessed_axial_ImageNet_ResNet50
  sex strat  strat_{male,female}_single, strat_{male,female}
  4.6, 4.7   exp2_preprocessed_axial_ImageNet_ResNet50 (41-subject test set);
             Table 4.7 also reads STAI_tAnxiety from the participants TSV

Usage (from the repo root):
    python scripts/make_ch4_tables.py
    python scripts/make_ch4_tables.py --results results --out results/ch4_tables
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K = ('accuracy', 'precision', 'recall', 'roc_auc')
SELECTED_SLICES = (33, 35, 38, 39, 50, 64, 65, 68, 70, 71)   # rows shown in Table 4.6


def f2(x):
    return f'{x:.2f}'


def js(p):
    with open(p) as f:
        return json.load(f)


def row(m):
    return [f2(m[k]) for k in K]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.path.join(REPO, 'results'),
                    help='Directory holding the run directories (default: results/)')
    ap.add_argument('--participants',
                    default=os.path.join(REPO, 'data', 'GSP', 'GSP_participants.tsv'),
                    help='GSP participants TSV with STAI_tAnxiety (Table 4.7)')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'ch4_tables'),
                    help='Output directory (default: results/ch4_tables)')
    ap.add_argument('--cutoff', type=float, default=36,
                    help='STAI-T label cutoff; borderline = within +/-5 of it (default: 36)')
    args = ap.parse_args()

    R, OUT, PART = args.results, args.out, args.participants
    os.makedirs(OUT, exist_ok=True)
    out = []

    def emit(s=''):
        print(s)
        out.append(s)

    # 4.1
    d = f'{R}/validate_sex'
    emit('TABLE 4.1  Sex classification (validation)'); emit('level | acc | prec | rec | auc')
    emit('subject | ' + ' | '.join(row(js(f'{d}/metrics_subject.json'))))
    emit('slice | ' + ' | '.join(row(js(f'{d}/metrics_slice.json'))))
    emit(f'SOURCE: {d}/metrics_subject.json, {d}/metrics_slice.json\n')

    # 4.2
    emit('TABLE 4.2  exp1 raw vs preprocessed (ResNet50, ImageNet), subject level')
    emit('orientation | type | acc | prec | rec | auc')
    src = []
    for v in ('axial', 'sagittal', 'coronal'):
        for t in ('raw', 'preprocessed'):
            p = f'{R}/exp1_{t}_{v}_ResNet50/metrics_subject.json'; src.append(p)
            emit(f'{v} | {t} | ' + ' | '.join(row(js(p))))
    for t in ('raw', 'preprocessed'):
        ms = [js(f'{R}/exp1_{t}_{v}_ResNet50/metrics_subject.json') for v in ('axial', 'sagittal', 'coronal')]
        emit(f'Average | {t} | ' + ' | '.join(f2(np.mean([m[k] for m in ms])) for k in K))
    emit('SOURCE: ' + ', '.join(src) + '\n')

    # 4.3 + 4.4
    emit('TABLE 4.3  exp2 backbone x pretraining (preprocessed axial), subject level')
    emit('weights | backbone | acc | prec | rec | auc')
    src = []; avg = {}
    for w in ('ImageNet', 'RadImageNet'):
        for b in ('ResNet50', 'DenseNet121', 'InceptionV3'):
            p = f'{R}/exp2_preprocessed_axial_{w}_{b}/metrics_subject.json'; src.append(p); m = js(p)
            avg.setdefault(w, []).append(m); emit(f'{w} | {b} | ' + ' | '.join(row(m)))
    emit('SOURCE: ' + ', '.join(src) + '\n')
    emit('TABLE 4.4  exp2 averages by pretraining source'); emit('weights | acc | prec | rec | auc')
    for w, ms in avg.items():
        emit(f'{w} | ' + ' | '.join(f2(np.mean([m[k] for m in ms])) for k in K))
    emit('SOURCE: mean of the three Table 4.3 rows per source (same files)\n')

    # 4.5
    d = f'{R}/exp3_kfold_preprocessed_axial_ImageNet_ResNet50'; cv = js(f'{d}/cv_summary.json')
    emit('TABLE 4.5  exp3 5-fold CV (ImageNet ResNet50, preprocessed axial), mean ± SD across folds')
    emit('level | acc | prec | rec | auc')
    for lvl in ('subject', 'slice'):
        emit(f'{lvl} | ' + ' | '.join(f"{cv[lvl][k]['mean']:.2f} ± {cv[lvl][k]['std']:.2f}" for k in K))
    oof = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{d}/fold_*/predictions_slice.csv'))])
    subj = oof.groupby('subject_id').agg(true=('true', 'first'), prob=('prob', 'mean'))
    emit(f'pooled out-of-fold subject AUC (n={len(subj)}): {roc_auc_score(subj.true, subj.prob):.2f}')
    emit(f'SOURCE: {d}/cv_summary.json; pooled from {d}/fold_[1-5]/predictions_slice.csv\n')

    # sex-stratified paragraph (Section 4.3.4)
    emit('SEX-STRATIFIED (4.3.4 paragraph, raw sagittal ResNet50), subject level')
    emit('model | eval | n | acc | prec | rec | auc')
    for sx in ('male', 'female'):
        m = js(f'{R}/strat_{sx}_single/metrics_subject.json')
        emit(f'{sx} | single | {m["n"]} | ' + ' | '.join(row(m)))
        c = js(f'{R}/strat_{sx}/cv_summary.json')
        emit(f'{sx} | 5-fold | - | ' + ' | '.join(
            f"{c['subject'][k]['mean']:.2f} ± {c['subject'][k]['std']:.2f}" for k in K))
    emit(f'SOURCE: {R}/strat_{{male,female}}_single/metrics_subject.json, '
         f'{R}/strat_{{male,female}}/cv_summary.json\n')

    # 4.6 / 4.7: ImageNet ResNet50, preprocessed axial (the exp3 configuration), 41-subject test set
    d = f'{R}/exp2_preprocessed_axial_ImageNet_ResNet50'
    sl = pd.read_csv(f'{d}/predictions_slice.csv'); sl['pred'] = (sl.prob >= 0.5).astype(int)
    emit('TABLE 4.6  per-slice, ImageNet ResNet50 test set'); emit('slice | acc | prec | rec | avg prob | slice auc')
    rows = []
    for idx, g in sl.groupby('slice_idx'):
        rows.append(dict(slice=idx, n=len(g), acc=accuracy_score(g.true, g.pred),
                         prec=precision_score(g.true, g.pred, zero_division=0),
                         rec=recall_score(g.true, g.pred, zero_division=0), avg_prob=g.prob.mean(),
                         auc=roc_auc_score(g.true, g.prob) if g.true.nunique() == 2 else np.nan))
    per = pd.DataFrame(rows); per.to_csv(f'{OUT}/table_4_6_all_slices.csv', index=False)
    for s in SELECTED_SLICES:
        r = per[per.slice == s]
        if len(r):
            r = r.iloc[0]
            emit(f'{s} | {r.acc:.2f} | {r.prec:.2f} | {r.rec:.2f} | {r.avg_prob:.2f} | {r.auc:.2f}')
    emit(f'Mean (all slices) | {accuracy_score(sl.true, sl.pred):.2f} | '
         f'{precision_score(sl.true, sl.pred, zero_division=0):.2f} | '
         f'{recall_score(sl.true, sl.pred):.2f} | {sl.prob.mean():.2f} | {roc_auc_score(sl.true, sl.prob):.2f}')
    emit(f'avg prob by slice ranges {per.avg_prob.min():.2f}–{per.avg_prob.max():.2f}; '
         f'highest per-slice AUC {per.auc.max():.2f} at slice {int(per.loc[per.auc.idxmax(), "slice"])}')
    emit(f'SOURCE: {d}/predictions_slice.csv (all slices: {OUT}/table_4_6_all_slices.csv)\n')

    subj = sl.groupby('subject_id').agg(true=('true', 'first'), prob=('prob', 'mean'))
    subj['stai'] = (pd.read_csv(PART, sep='\t').set_index('participant_id')
                      .STAI_tAnxiety.reindex(subj.index).values)
    subj['pred'] = (subj.prob >= 0.5).astype(int)
    subj['group'] = np.where((subj.stai - args.cutoff).abs() <= 5, 'Borderline', 'Non-Borderline')
    emit(f'TABLE 4.7  borderline (STAI-T within ±5 of {args.cutoff:g}), ImageNet ResNet50 test set')
    emit('group | n | high | low | acc | auc')
    for gname, g in list(subj.groupby('group')) + [('All', subj)]:
        emit(f'{gname} | {len(g)} | {int(g.true.sum())} | {int((1 - g.true).sum())} | '
             f'{accuracy_score(g.true, g.pred):.2f} | {roc_auc_score(g.true, g.prob):.2f}')
    emit(f'subject prob range {subj.prob.min():.2f}–{subj.prob.max():.2f}; predicted high {subj.pred.mean():.2f}')
    emit(f'SOURCE: {d}/predictions_slice.csv + STAI_tAnxiety from {PART}')
    with open(f'{OUT}/ch4_tables.txt', 'w') as f:
        f.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
