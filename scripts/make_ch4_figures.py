"""
Figures 4.4 and 4.5 from the 41-subject test set of the ImageNet ResNet50,
preprocessed-axial run (the configuration used for the slice- and
subject-level analyses).

  fig_4_4_confidence_by_slice.png  mean predicted probability at each slice position
  fig_4_5_prob_vs_stai.png         each test subject's mean probability vs. STAI-T,
                                   coloured by whether the subject is classified correctly

Usage (from the repo root):
    python scripts/make_ch4_figures.py
    python scripts/make_ch4_figures.py --run results/exp2_preprocessed_axial_ImageNet_ResNet50 \\
        --out results/ch4_tables
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run',
                    default=os.path.join(REPO, 'results', 'exp2_preprocessed_axial_ImageNet_ResNet50'),
                    help='Run directory with predictions_slice.csv')
    ap.add_argument('--participants',
                    default=os.path.join(REPO, 'data', 'GSP', 'GSP_participants.tsv'),
                    help='GSP participants TSV with STAI_tAnxiety (Figure 4.5)')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'ch4_tables'),
                    help='Output directory (default: results/ch4_tables)')
    ap.add_argument('--cutoff', type=float, default=36, help='STAI-T label cutoff (default: 36)')
    args = ap.parse_args()

    RUN, OUT = args.run, args.out
    os.makedirs(OUT, exist_ok=True)
    sl = pd.read_csv(f'{RUN}/predictions_slice.csv')

    # Figure 4.4: average predicted probability by slice position
    by_slice = sl.groupby('slice_idx')['prob'].mean()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(by_slice.index, by_slice.values, color='purple', lw=1.8, label='Avg predicted probability')
    ax.axhline(0.5, color='red', ls='--', lw=1, label='Decision boundary (0.5)')
    ax.set_xlim(by_slice.index.min(), by_slice.index.max()); ax.set_ylim(0, 1)
    ax.set_xlabel('Slice Number'); ax.set_ylabel('Average Predicted Probability (Anxiety)')
    ax.set_title('Model Confidence by Slice Position'); ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_4_4_confidence_by_slice.png', dpi=150); plt.close(fig)

    # Figure 4.5: subject probability vs STAI-T, coloured by correctness
    subj = sl.groupby('subject_id').agg(true=('true', 'first'), prob=('prob', 'mean'))
    stai = pd.read_csv(args.participants, sep='\t').set_index('participant_id').STAI_tAnxiety
    subj['stai'] = stai.reindex(subj.index).values
    subj['correct'] = (subj.prob >= 0.5).astype(int) == subj.true
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for ok, col, lab in ((True, 'tab:green', 'correct'), (False, 'tab:red', 'misclassified')):
        s = subj[subj.correct == ok]
        ax.scatter(s.stai, s.prob, c=col, s=60, edgecolor='k', linewidth=0.5, alpha=0.85, label=lab)
    ax.axvline(args.cutoff, color='k', ls='--', lw=2, label=f'STAI Threshold ({args.cutoff:g})')
    ax.axhline(0.5, color='blue', ls='--', lw=1, label='Decision boundary (0.5)')
    ax.set_ylim(0, 1); ax.set_xlabel('STAI-T Score'); ax.set_ylabel('Model Predicted Probability (Anxiety)')
    ax.set_title('Predicted Probability vs STAI Score\n(green=correct, red=misclassified)')
    ax.grid(alpha=0.3); ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_4_5_prob_vs_stai.png', dpi=150); plt.close(fig)

    print('slices', by_slice.index.min(), '-', by_slice.index.max(),
          'avg prob range', round(by_slice.min(), 4), round(by_slice.max(), 4))
    print('subjects', len(subj), 'correct', int(subj.correct.sum()),
          'prob range', round(subj.prob.min(), 4), round(subj.prob.max(), 4))


if __name__ == '__main__':
    main()
