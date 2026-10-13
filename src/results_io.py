"""
results_io.py — shared helpers for writing the canonical per-run output layout.

Layout (one directory per experiment):

    results/<EXPERIMENT_NAME>/
        ├── config.json
        ├── predictions_slice.csv   ← source of truth
        ├── metrics_slice.json
        ├── metrics_subject.json
        ├── training_history.json
        └── plots/

predictions_slice.csv is the SOURCE OF TRUTH; metrics_subject.json and any
subject-level plots are derived from it by aggregating slice probabilities to
the subject level (mean probability → threshold at 0.5).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
)


# ---------- writing the config snapshot ------------------------------------

def write_config_snapshot(config, run_dir):
    """Dump the Config class attributes (JSON-serialisable only) to config.json."""
    snapshot = {}
    for k in dir(config):
        if k.startswith('_') or callable(getattr(config, k)):
            continue
        v = getattr(config, k)
        try:
            json.dumps(v)
            snapshot[k] = v
        except TypeError:
            snapshot[k] = repr(v)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)


# ---------- predictions CSV ------------------------------------------------

def write_predictions_slice(run_dir, subject_ids, slice_indices, y_true, y_prob,
                            threshold=0.5):
    """Write per-slice predictions as the single source of truth."""
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'slice_idx': slice_indices,
        'true': np.asarray(y_true, dtype=int),
        'prob': np.asarray(y_prob, dtype=float),
        'pred': (np.asarray(y_prob) >= threshold).astype(int),
    })
    path = os.path.join(run_dir, 'predictions_slice.csv')
    df.to_csv(path, index=False)
    return df


# ---------- subject-level aggregation --------------------------------------

def aggregate_to_subject(slice_df, threshold=0.5):
    """
    Mean probability across a subject's slices, then threshold.
    Returns a DataFrame with columns: subject_id, true, mean_prob, pred.
    """
    agg = (slice_df.groupby('subject_id', as_index=False)
                   .agg(true=('true', 'first'),
                        mean_prob=('prob', 'mean')))
    agg['pred'] = (agg['mean_prob'] >= threshold).astype(int)
    return agg


# ---------- metrics --------------------------------------------------------

def _safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Standard binary metrics (Acc, Precision, Recall, ROC-AUC)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'roc_auc':   _safe_auc(y_true, y_prob),
        'n':         int(len(y_true)),
    }


def write_metrics(run_dir, slice_df):
    """Compute and write both metrics_slice.json and metrics_subject.json."""
    subj_df = aggregate_to_subject(slice_df)
    slice_metrics = compute_metrics(slice_df['true'].values, slice_df['prob'].values)
    subj_metrics = compute_metrics(subj_df['true'].values, subj_df['mean_prob'].values)

    with open(os.path.join(run_dir, 'metrics_slice.json'), 'w') as f:
        json.dump(slice_metrics, f, indent=2)
    with open(os.path.join(run_dir, 'metrics_subject.json'), 'w') as f:
        json.dump(subj_metrics, f, indent=2)
    return slice_metrics, subj_metrics


# ---------- training history ----------------------------------------------

def write_training_history(run_dir, history, training_time_seconds=None,
                           best_epoch=None, best_stage=None):
    """Serialise a Keras History.history dict (lists of floats) to JSON."""
    payload = {
        'training_time_seconds': training_time_seconds,
        'n_epochs': len(next(iter(history.history.values()))),
        'best_epoch': best_epoch,
        'best_stage': best_stage,
        'history': {k: [float(v) for v in vs] for k, vs in history.history.items()},
    }
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(payload, f, indent=2)
