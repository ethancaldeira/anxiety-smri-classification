"""
evaluate.py — post-hoc analysis of training-run outputs.

Consumes the canonical layout written by train.py:

    results/<EXPERIMENT_NAME>/[fold_<k>/]
        ├── predictions_slice.csv     ← source of truth
        ├── metrics_slice.json
        └── metrics_subject.json

Four subcommands:

  run <dir>                   per-run plot set (CM, ROC, PR at slice + subject;
                              slice_importance, slice_misclassification,
                              borderline_analysis, prob_by_slice_position,
                              prob_vs_stai if participants TSV is available)
                              Auto-invoked by train.py at the end of every fit.

  compare <glob>              cross-run overlay — mean P(anxiety) by slice
                              position, one line per run, written to
                              results/_compare/prob_by_slice_position_all_runs.png

  table <glob>                cross-run comparison TABLE (markdown + optional CSV):
                              one row per run with hyperparams + slice/subject
                              metrics; k-fold cells show mean±std.

  validation-summary          three-bar headline figure comparing the winning
                              exp1 cell against the two validation runs
                              (positive control = sex, negative control = permute).
                              Written to results/_compare/validation_summary_auc.png.

Examples:
    python src/evaluate.py run results/exp1_raw_sagittal_ResNet50
    python src/evaluate.py run results/exp3_kfold_raw_sagittal_ResNet50 --all-folds
    python src/evaluate.py compare 'results/exp1_*'
    python src/evaluate.py table 'results/exp*' --out results/_compare/table.csv
    python src/evaluate.py validation-summary
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc as sk_auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

import results_io


CLASS_NAMES = ['Control', 'Anxiety']
DEFAULT_PARTICIPANTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'GSP', 'GSP_participants.tsv',
)


# -------------------------------------------------------------------------
# plots
# -------------------------------------------------------------------------

def _plot_confusion(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _plot_roc(y_true, y_prob, out_path, title):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _plot_pr(y_true, y_prob, out_path, title):
    if len(np.unique(y_true)) < 2:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def _plot_prob_by_slice_position(slice_df, out_path, title, test_prevalence=None):
    """
    Mean predicted P(anxiety) as a function of slice index, split by true class.
    Shaded band is ±1 SEM across subjects within each class at each slice.
    """
    df = slice_df[slice_df['slice_idx'] >= 0].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {0: 'tab:blue', 1: 'tab:red'}
    labels = {0: 'True Control', 1: 'True Anxiety'}
    for cls, sub in df.groupby('true'):
        g = sub.groupby('slice_idx')['prob']
        mean = g.mean()
        sem = g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))
        ax.plot(mean.index, mean.values, color=colours[cls],
                label=f'{labels[cls]} (n={sub.subject_id.nunique()})')
        ax.fill_between(mean.index, mean - sem, mean + sem,
                        color=colours[cls], alpha=0.2)

    ax.axhline(0.5, color='k', linestyle='--', alpha=0.4, label='threshold 0.5')
    if test_prevalence is not None:
        ax.axhline(test_prevalence, color='grey', linestyle=':', alpha=0.6,
                   label=f'test prevalence {test_prevalence:.2f}')
    ax.set_xlabel('Slice index (lower = more inferior/anterior/right)')
    ax.set_ylabel('Mean P(anxiety)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_prob_vs_stai(subj_df, participants_df, out_path, title, cutoff=None):
    """
    Per-subject scatter of mean P(anxiety) vs continuous STAI-T score.
    Coloured by true binary label. Annotates Pearson r and Spearman ρ.
    """
    merged = subj_df.merge(
        participants_df[['participant_id', 'STAI_tAnxiety']],
        left_on='subject_id', right_on='participant_id', how='inner',
    ).dropna(subset=['STAI_tAnxiety'])
    if merged.empty:
        print(f"  [skip] prob_vs_stai — no STAI matches for subjects in {out_path}")
        return

    x = merged['STAI_tAnxiety'].values
    y = merged['mean_prob'].values
    r, p_r = pearsonr(x, y) if len(x) >= 3 else (float('nan'), float('nan'))
    rho, p_s = spearmanr(x, y) if len(x) >= 3 else (float('nan'), float('nan'))

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, colour, lab in [(0, 'tab:blue', 'Control'),
                             (1, 'tab:red', 'Anxiety')]:
        mask = merged['true'] == cls
        ax.scatter(x[mask], y[mask], color=colour, label=f'{lab} (n={mask.sum()})',
                   s=55, alpha=0.75, edgecolor='k', linewidth=0.5)

    if cutoff is not None:
        ax.axvline(cutoff, color='k', linestyle='--', alpha=0.4,
                   label=f'STAI-T = {cutoff} (label cutoff)')
    ax.axhline(0.5, color='grey', linestyle=':', alpha=0.5, label='prob = 0.5 (decision)')
    ax.set_xlabel('STAI-T score (continuous)')
    ax.set_ylabel('Mean predicted P(anxiety) across slices')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    ax.text(0.02, 0.98,
            f"Pearson r = {r:+.3f} (p={p_r:.3g})\nSpearman ρ = {rho:+.3f} (p={p_s:.3g})\nn = {len(merged)}",
            transform=ax.transAxes, va='top', ha='left',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='grey'))
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_slice_importance(slice_df, out_path, title, min_slices_per_idx=10):
    """Per-slice AUC across test subjects: which slice positions are
    most discriminative? Bootstraps a ±1.96 SE band where possible."""
    df = slice_df[slice_df['slice_idx'] >= 0].copy()
    if df.empty:
        return
    grouped = df.groupby('slice_idx')
    rows = []
    rng = np.random.default_rng(0)
    for idx, sub in grouped:
        if len(sub) < min_slices_per_idx or sub['true'].nunique() < 2:
            continue
        y_true = sub['true'].values
        y_prob = sub['prob'].values
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            continue
        # Bootstrap SE (100 resamples — cheap)
        boots = []
        for _ in range(100):
            sel = rng.integers(0, len(sub), size=len(sub))
            if len(np.unique(y_true[sel])) < 2:
                continue
            boots.append(roc_auc_score(y_true[sel], y_prob[sel]))
        se = float(np.std(boots)) if boots else 0.0
        rows.append((idx, auc, se, len(sub)))
    if not rows:
        return
    rows.sort()
    xs   = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    ses  = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, aucs, 'o-', color='tab:purple', label='per-slice AUC')
    ax.fill_between(xs,
                    [a - 1.96 * s for a, s in zip(aucs, ses)],
                    [a + 1.96 * s for a, s in zip(aucs, ses)],
                    color='tab:purple', alpha=0.2, label='±1.96·SE (bootstrap)')
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='chance (0.5)')
    ax.set_xlabel('Slice index')
    ax.set_ylabel('AUC at this slice position')
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_slice_misclassification(slice_df, out_path, title):
    """Per-slice misclassification rate, split by true class."""
    df = slice_df[slice_df['slice_idx'] >= 0].copy()
    if df.empty:
        return
    df['err'] = (df['true'] != df['pred']).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {0: 'tab:blue', 1: 'tab:red'}
    labels  = {0: 'True Control', 1: 'True Anxiety'}
    for cls, sub in df.groupby('true'):
        g = sub.groupby('slice_idx')['err']
        rate = g.mean()
        n    = g.count()
        sem  = np.sqrt(rate * (1 - rate) / n.clip(lower=1))
        ax.plot(rate.index, rate.values, color=colours[cls],
                label=f'{labels[cls]} (n={sub.subject_id.nunique()} subj)')
        ax.fill_between(rate.index, rate - sem, rate + sem,
                        color=colours[cls], alpha=0.2)

    overall = df.groupby('slice_idx')['err'].mean()
    ax.axhline(overall.mean(), color='grey', linestyle=':', alpha=0.6,
               label=f'overall mean error = {overall.mean():.2f}')
    ax.set_xlabel('Slice index')
    ax.set_ylabel('Misclassification rate')
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_borderline_analysis(subj_df, slice_df, out_path, title, threshold=0.5):
    """Per-subject mean prob ± std, sorted; reveals borderline subjects."""
    if subj_df.empty:
        return
    std_per_subj = (slice_df.groupby('subject_id')['prob']
                            .std(ddof=1)
                            .rename('std_prob'))
    merged = subj_df.merge(std_per_subj, on='subject_id', how='left')
    merged['std_prob'] = merged['std_prob'].fillna(0.0)
    merged = merged.sort_values('mean_prob').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.2 * len(merged))))
    colours = merged['true'].map({0: 'tab:blue', 1: 'tab:red'})
    ax.errorbar(merged['mean_prob'], merged.index,
                xerr=merged['std_prob'],
                fmt='o', ecolor='lightgrey', elinewidth=1,
                markeredgecolor='k', markersize=6,
                color='none')
    for col, c in zip(['tab:blue', 'tab:red'], [0, 1]):
        mask = merged['true'] == c
        ax.scatter(merged.loc[mask, 'mean_prob'], merged.index[mask],
                   color=col, s=50, edgecolor='k', linewidth=0.5,
                   label=f"{'Control' if c == 0 else 'Anxiety'} (n={int(mask.sum())})")
    ax.axvline(threshold, color='k', linestyle='--', alpha=0.5,
               label=f'decision threshold = {threshold}')

    band_lo, band_hi = threshold - 0.05, threshold + 0.05
    ax.axvspan(band_lo, band_hi, color='yellow', alpha=0.18,
               label=f'borderline band (±0.05)')
    n_borderline = int(((merged['mean_prob'] >= band_lo) &
                        (merged['mean_prob'] <= band_hi)).sum())

    ax.set_xlabel('Mean predicted P(anxiety) across the subject\'s slices')
    ax.set_ylabel('Test subject (sorted by mean prob)')
    ax.set_yticks(merged.index)
    ax.set_yticklabels(merged['subject_id'], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_title(f'{title}  —  {n_borderline}/{len(merged)} subjects in borderline band')
    ax.grid(axis='x', alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_compare_slice_position(run_dirs, out_path, title='Mean P(anxiety) by slice position — all runs'):
    """
    Cross-run overlay: one line per run, mean prob across all test subjects at each slice index.
    No class split — the goal is to see whether different configs land on the same spatial pattern.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for rd in sorted(run_dirs):
        csv_path = os.path.join(rd, 'predictions_slice.csv')
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df = df[df['slice_idx'] >= 0]
        if df.empty:
            continue
        mean = df.groupby('slice_idx')['prob'].mean()
        ax.plot(mean.index, mean.values, label=os.path.basename(rd), alpha=0.85)
        plotted += 1

    if plotted == 0:
        print(f"  [skip] compare — no runs with predictions_slice.csv")
        plt.close(fig); return

    ax.axhline(0.5, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel('Slice index')
    ax.set_ylabel('Mean P(anxiety) across all test subjects')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}  ({plotted} runs overlaid)")


# -------------------------------------------------------------------------
# main per-run evaluation
# -------------------------------------------------------------------------

def _load_participants(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path, sep='\t')
    except Exception as e:
        print(f"  [warn] could not read participants TSV {path}: {e}")
        return None


def evaluate_run(run_dir, participants_df=None):
    """Produce plots + (re-)write metrics for one run dir."""
    csv_path = os.path.join(run_dir, 'predictions_slice.csv')
    if not os.path.isfile(csv_path):
        print(f"  [skip] {run_dir} — no predictions_slice.csv")
        return None

    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    slice_df = pd.read_csv(csv_path)
    subj_df = results_io.aggregate_to_subject(slice_df)

    slice_m, subj_m = results_io.write_metrics(run_dir, slice_df)
    run_name = os.path.basename(run_dir.rstrip('/'))

    # --- slice-level diagnostics ---------------------------------------
    _plot_confusion(slice_df['true'], slice_df['pred'],
                    os.path.join(plots_dir, 'confusion_matrix_slice.png'),
                    title=f'Slice-level (n={len(slice_df)})')
    _plot_roc(slice_df['true'], slice_df['prob'],
              os.path.join(plots_dir, 'roc_curve_slice.png'),
              title='Slice-level ROC')
    _plot_pr(slice_df['true'], slice_df['prob'],
             os.path.join(plots_dir, 'pr_curve_slice.png'),
             title='Slice-level PR')
    _plot_slice_importance(
        slice_df,
        os.path.join(plots_dir, 'slice_importance.png'),
        title=f'{run_name} — per-slice AUC (slice importance)',
    )
    _plot_slice_misclassification(
        slice_df,
        os.path.join(plots_dir, 'slice_misclassification.png'),
        title=f'{run_name} — misclassification rate by slice position',
    )

    # --- subject-level diagnostics -------------------------------------
    _plot_confusion(subj_df['true'], subj_df['pred'],
                    os.path.join(plots_dir, 'confusion_matrix_subject.png'),
                    title=f'Subject-level (n={len(subj_df)})')
    _plot_roc(subj_df['true'], subj_df['mean_prob'],
              os.path.join(plots_dir, 'roc_curve_subject.png'),
              title='Subject-level ROC')
    _plot_pr(subj_df['true'], subj_df['mean_prob'],
             os.path.join(plots_dir, 'pr_curve_subject.png'),
             title='Subject-level PR')
    _plot_borderline_analysis(
        subj_df, slice_df,
        os.path.join(plots_dir, 'borderline_analysis.png'),
        title=f'{run_name} — borderline analysis',
    )

    # --- probability by slice position and vs. continuous STAI-T --------
    test_prevalence = float(subj_df['true'].mean()) if len(subj_df) else None
    _plot_prob_by_slice_position(
        slice_df,
        os.path.join(plots_dir, 'prob_by_slice_position.png'),
        title=f'{run_name} — P(anxiety) by slice position',
        test_prevalence=test_prevalence,
    )
    if participants_df is not None:
        cfg = _load_json(os.path.join(run_dir, 'config.json')) or {}
        _plot_prob_vs_stai(
            subj_df, participants_df,
            os.path.join(plots_dir, 'prob_vs_stai.png'),
            title=f'{run_name} — predicted prob vs STAI-T',
            cutoff=cfg.get('STAI_CUTOFF'),
        )

    print(f"  {run_name:<40}  "
          f"slice acc={slice_m['accuracy']:.3f} auc={slice_m['roc_auc']:.3f}  "
          f"subject acc={subj_m['accuracy']:.3f} auc={subj_m['roc_auc']:.3f}")
    return {'slice': slice_m, 'subject': subj_m}


def evaluate_kfold(run_dir, participants_df=None):
    """Iterate every fold_<k>/ subdir; write/refresh cv_summary.json."""
    fold_dirs = sorted(glob.glob(os.path.join(run_dir, 'fold_*')),
                       key=lambda d: int(d.rsplit('_', 1)[-1]))
    if not fold_dirs:
        print(f"  [skip] {run_dir} — no fold_* subdirs")
        return

    fold_results = []
    for fd in fold_dirs:
        m = evaluate_run(fd, participants_df=participants_df)
        if m is not None:
            m['fold'] = int(fd.rsplit('_', 1)[-1])
            fold_results.append(m)
    if not fold_results:
        return

    def agg(level, key):
        vals = [r[level][key] for r in fold_results
                if not np.isnan(r[level][key])]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    summary = {
        'n_folds': len(fold_results),
        'slice':   {k: agg('slice',   k) for k in ('accuracy','precision','recall','roc_auc')},
        'subject': {k: agg('subject', k) for k in ('accuracy','precision','recall','roc_auc')},
        'fold_results': fold_results,
    }
    with open(os.path.join(run_dir, 'cv_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nK-fold summary ({len(fold_results)} folds):")
    for level in ('slice', 'subject'):
        print(f"  {level}-level:")
        for k, v in summary[level].items():
            print(f"    {k:9s}  {v['mean']:.3f} ± {v['std']:.3f}")


# -------------------------------------------------------------------------
# cross-run comparison table
# -------------------------------------------------------------------------

TABLE_DEFAULT_HYPERPARAMS = [
    'EXPERIMENT_TYPE', 'VIEW_MODE', 'BASE_MODEL',
    'USE_RADIMAGENET', 'USE_AUGMENTATION', 'N_CENTRAL_SLICES',
]
TABLE_METRICS = ['slice_acc', 'slice_auc', 'subj_acc', 'subj_auc',
                 'subj_prec', 'subj_recall']


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt_metric(level_metrics, key):
    """'mean±std' if dict-of-stats, else raw scalar formatted, else dash."""
    if level_metrics is None:
        return '—'
    v = level_metrics.get(key)
    if v is None:
        return '—'
    if isinstance(v, dict) and 'mean' in v:
        return f"{v['mean']:.3f}±{v['std']:.3f}"
    return f"{float(v):.3f}"


def _row_for_run(run_dir, hyperparams):
    cfg = _load_json(os.path.join(run_dir, 'config.json')) or {}
    cv = _load_json(os.path.join(run_dir, 'cv_summary.json'))
    if cv is not None:
        slice_m, subj_m = cv['slice'], cv['subject']
        kind = f"{cv['n_folds']}-fold"
    else:
        slice_m = _load_json(os.path.join(run_dir, 'metrics_slice.json'))
        subj_m = _load_json(os.path.join(run_dir, 'metrics_subject.json'))
        kind = 'single'
    row = {'run': os.path.basename(os.path.normpath(run_dir)), 'kind': kind}
    for hp in hyperparams:
        row[hp] = cfg.get(hp, '—')
    row['slice_acc']   = _fmt_metric(slice_m, 'accuracy')
    row['slice_auc']   = _fmt_metric(slice_m, 'roc_auc')
    row['subj_acc']    = _fmt_metric(subj_m, 'accuracy')
    row['subj_auc']    = _fmt_metric(subj_m, 'roc_auc')
    row['subj_prec']   = _fmt_metric(subj_m, 'precision')
    row['subj_recall'] = _fmt_metric(subj_m, 'recall')
    return row


def _sort_key(value):
    if value == '—':
        return float('-inf')
    return float(str(value).split('±')[0])


def build_comparison_table(pattern, hyperparams=None, sort_by='subj_auc', out_csv=None):
    """Markdown table of one row per run; optional CSV. Returns the DataFrame."""
    hyperparams = hyperparams or TABLE_DEFAULT_HYPERPARAMS
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    rows = []
    for d in dirs:
        if not (os.path.isfile(os.path.join(d, 'config.json')) or
                os.path.isfile(os.path.join(d, 'cv_summary.json'))):
            continue
        rows.append(_row_for_run(d, hyperparams))
    if not rows:
        raise SystemExit(f"No valid runs (config.json/cv_summary.json) under {pattern!r}")
    df = pd.DataFrame(rows)
    df = df.iloc[df[sort_by].map(_sort_key).argsort()[::-1]].reset_index(drop=True)
    print(df.to_markdown(index=False))
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv}")
    return df


# -------------------------------------------------------------------------
# validation-summary 3-bar headline
# -------------------------------------------------------------------------

VALIDATION_BAR_COLOURS = ['tab:gray', 'tab:orange', 'tab:green']


def _pick_best_exp1_dir(results_root):
    """Find the highest-subject-AUC exp1_*_ResNet50 dir under results_root."""
    best = (None, -1.0)
    for d in sorted(glob.glob(os.path.join(results_root, 'exp1_*_ResNet50'))):
        m = os.path.join(d, 'metrics_subject.json')
        if not os.path.isfile(m):
            continue
        auc = json.load(open(m)).get('roc_auc')
        if auc is None or auc != auc:
            continue
        if auc > best[1]:
            best = (d, auc)
    return best[0]


def plot_validation_summary(results_root=None, real_run=None, out_path=None):
    """Three-bar headline: real anxiety baseline vs negative vs positive control."""
    if results_root is None:
        results_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    if real_run is None:
        real_run = _pick_best_exp1_dir(results_root) or os.path.join(
            results_root, 'exp1_raw_sagittal_ResNet50')
    if out_path is None:
        out_path = os.path.join(results_root, '_compare', 'validation_summary_auc.png')

    runs = [
        ('Real anxiety labels\n(exp1 winning cell)',     real_run),
        ('Negative control\n(anxiety labels shuffled)',  os.path.join(results_root, 'validate_permute')),
        ('Positive control\n(predict sex)',              os.path.join(results_root, 'validate_sex')),
    ]
    labels, aucs = [], []
    for label, d in runs:
        m = os.path.join(d, 'metrics_subject.json')
        if not os.path.isfile(m):
            print(f"  [skip] {d} — metrics_subject.json missing")
            continue
        auc = json.load(open(m)).get('roc_auc')
        if auc is None:
            continue
        labels.append(label); aucs.append(auc)

    if not labels:
        print("  no validation runs found; have you run "
              "`python run_experiments.py validate`?")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, aucs, color=VALIDATION_BAR_COLOURS[:len(labels)],
                  edgecolor='k', linewidth=0.6)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='chance (AUC = 0.5)')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Subject-level ROC AUC')
    ax.set_title(f'Pipeline validation — {os.path.basename(os.path.normpath(real_run))}')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='upper left', fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  wrote {out_path}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest='cmd', required=True)

    # run
    ar = sub.add_parser('run', help='per-run plot set (auto-invoked by train.py)')
    ar.add_argument('run_dir', help='Path to results/<EXPERIMENT_NAME>')
    g = ar.add_mutually_exclusive_group()
    g.add_argument('--fold', type=int, help='Evaluate one fold only')
    g.add_argument('--all-folds', action='store_true', help='Evaluate every fold_<k>/')
    ar.add_argument('--participants', default=DEFAULT_PARTICIPANTS,
                    help='Path to GSP_participants.tsv (for prob-vs-STAI plot)')

    # compare
    ac = sub.add_parser('compare', help='cross-run prob-by-slice-position overlay')
    ac.add_argument('pattern', help='Glob of run dirs, e.g. "results/exp1_*"')

    # table
    at = sub.add_parser('table', help='cross-run comparison table (markdown + CSV)')
    at.add_argument('pattern', help='Glob of run dirs, e.g. "results/exp*"')
    at.add_argument('--hyperparams', nargs='+', default=TABLE_DEFAULT_HYPERPARAMS,
                    help='Config keys to include as columns')
    at.add_argument('--sort', default='subj_auc', choices=TABLE_METRICS,
                    help='Sort rows by this metric (descending)')
    at.add_argument('--out', help='Optional CSV output path')

    # validation-summary
    av = sub.add_parser('validation-summary',
                        help='3-bar headline: real anxiety vs negative vs positive control')
    av.add_argument('--real-run', help='Override the "real anxiety" run dir '
                                       '(default: auto-pick the best exp1_*_ResNet50)')
    av.add_argument('--out', help='Output PNG path (default: results/_compare/validation_summary_auc.png)')

    args = ap.parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.cmd == 'run':
        participants_df = _load_participants(args.participants)
        if participants_df is None:
            print(f"  [info] no participants TSV at {args.participants} — skipping prob-vs-STAI")
        if args.all_folds:
            evaluate_kfold(args.run_dir, participants_df=participants_df)
        elif args.fold is not None:
            evaluate_run(os.path.join(args.run_dir, f'fold_{args.fold}'),
                         participants_df=participants_df)
        else:
            evaluate_run(args.run_dir, participants_df=participants_df)

    elif args.cmd == 'compare':
        run_dirs = sorted(d for d in glob.glob(args.pattern) if os.path.isdir(d))
        out = os.path.join(repo_root, 'results', '_compare',
                           'prob_by_slice_position_all_runs.png')
        _plot_compare_slice_position(run_dirs, out)

    elif args.cmd == 'table':
        build_comparison_table(args.pattern, hyperparams=args.hyperparams,
                               sort_by=args.sort, out_csv=args.out)

    elif args.cmd == 'validation-summary':
        plot_validation_summary(real_run=args.real_run, out_path=args.out)


if __name__ == '__main__':
    main()
