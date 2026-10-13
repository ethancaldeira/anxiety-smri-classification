"""
train.py — train a single-split or k-fold CV anxiety MRI classifier.

Output layout per run (single canonical convention — see results_io.py):

    results/<EXPERIMENT_NAME>/                          (single split)
        ├── config.json
        ├── predictions_slice.csv       ← source of truth
        ├── metrics_slice.json
        ├── metrics_subject.json
        ├── training_history.json
        ├── best.keras
        └── plots/

    results/<EXPERIMENT_NAME>/                          (k-fold)
        ├── cv_summary.json             (aggregated mean ± std across folds)
        └── fold_<k>/                   (same files as above, one set per fold)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import re
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import Config
from data_loader import MRIDataLoader, _get_preprocess_input
from model import (
    create_model, get_stage1_callbacks, get_stage2_callbacks, make_checkpoint,
    unfreeze_for_stage2,
)
from mri_augment import MRIBatchSequence, make_train_transform
import results_io
# evaluate_run gives every training run the full plot set (slice + subject
# CM/ROC/PR, slice_importance, slice_misclassification, borderline_analysis,
# prob_by_slice_position, prob_vs_stai).
from evaluate import evaluate_run, _load_participants, DEFAULT_PARTICIPANTS


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------

def _slice_idx_from_filename(filename):
    """`SubNNNN_S1_slice_071.png` -> 71. Returns -1 if not parseable."""
    m = re.search(r'slice[_-]?(\d+)', os.path.basename(filename))
    return int(m.group(1)) if m else -1


def _class_weights_dict(y_train):
    """Balanced class weights via sklearn formula."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced',
                                   classes=classes, y=y_train)
    return dict(zip(classes, weights))


def _plot_training_history(history, run_dir):
    """4-panel grid of loss/acc/AUC/precision for train + val."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, key, title in zip(
        axes.flat,
        ['loss', 'accuracy', 'auc', 'precision'],
        ['Loss', 'Accuracy', 'AUC', 'Precision'],
    ):
        if key not in history.history:
            continue
        epochs = range(1, len(history.history[key]) + 1)
        ax.plot(epochs, history.history[key], 'bo-', label=f'train {title}')
        if f'val_{key}' in history.history:
            ax.plot(epochs, history.history[f'val_{key}'], 'ro-',
                    label=f'val {title}')
        ax.set_xlabel('epoch'); ax.set_ylabel(title); ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'plots', 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def _merge_histories(*histories):
    """Concatenate per-epoch lists across stage-1 and stage-2 keras Histories."""
    out = {}
    for h in histories:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(list(v))

    class _Merged:
        pass
    m = _Merged()
    m.history = out
    return m


def _train_one_fold(X_train, y_train, X_val, y_val,
                    X_test, y_test, files_test, subjects_test, subjects_test_per_slice,
                    run_dir, fold_label=''):
    """Train one model, write all outputs to run_dir, return metrics dict.

    Two-stage fine-tuning:
      Stage 1 — backbone frozen, head only, STAGE1_EPOCHS at STAGE1_LR.
      Stage 2 — backbone unfrozen, LEARNING_RATE with warmup+cosine, ES.
    The test set is scored with the lowest-val_loss epoch across both stages.
    """
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)

    results_io.write_config_snapshot(Config, run_dir)
    cw = _class_weights_dict(y_train) if Config.USE_CLASS_WEIGHTS else None

    print(f"\n{'-'*60}\n  Training{fold_label} — "
          f"train={len(X_train)} val={len(X_val)} test={len(X_test)}\n{'-'*60}")
    if cw:
        print(f"  Class weights: {cw}")

    # ---- Build TorchIO-augmented train sequence + plain val/test sequences.
    #      All three apply the backbone's preprocess_input per batch; only
    #      the train sequence applies augmentation.
    preprocess_fn = _get_preprocess_input(Config.BASE_MODEL, Config.USE_RADIMAGENET)
    train_aug = make_train_transform(Config) if Config.USE_AUGMENTATION else None
    train_seq = MRIBatchSequence(X_train, y_train, Config.BATCH_SIZE,
                                 preprocess_fn=preprocess_fn,
                                 train_transform=train_aug, shuffle=True)
    val_seq   = MRIBatchSequence(X_val, y_val, Config.BATCH_SIZE,
                                 preprocess_fn=preprocess_fn,
                                 train_transform=None, shuffle=False)

    checkpoint = make_checkpoint(run_dir)

    # ---- Stage 1: head-only warm-up ----
    Config.FREEZE_BASE = True
    model = create_model(Config)
    print(f"\n  === Stage 1 ===  frozen backbone, {Config.STAGE1_EPOCHS} epochs at lr={Config.STAGE1_LR}"
          f"  (augment={'ON' if train_aug is not None else 'OFF'})")
    t0 = time.time()
    hist_s1 = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=Config.STAGE1_EPOCHS,
        callbacks=get_stage1_callbacks(checkpoint),
        class_weight=cw,
        verbose=1,
    )

    # ---- Stage 2: unfreeze + fine-tune with warmup+cosine ----
    print(f"\n  === Stage 2 ===  unfreeze backbone, fine-tune (LR {Config.LEARNING_RATE} with warmup {Config.WARMUP_EPOCHS}, cosine -> {Config.MIN_LR}, ES patience {Config.EARLY_STOPPING_PATIENCE})")
    model = unfreeze_for_stage2(model, Config)
    hist_s2 = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=Config.EPOCHS - Config.STAGE1_EPOCHS,
        callbacks=get_stage2_callbacks(Config, checkpoint),
        class_weight=cw,
        verbose=1,
    )
    train_secs = time.time() - t0
    history = _merge_histories(hist_s1, hist_s2)

    # Best epoch over both stages (ModelCheckpoint keeps the first minimum)
    best_epoch = int(np.nanargmin(history.history['val_loss'])) + 1
    best_stage = 1 if best_epoch <= Config.STAGE1_EPOCHS else 2
    print(f"\n  Best epoch: {best_epoch} (stage {best_stage}), "
          f"val_loss={history.history['val_loss'][best_epoch - 1]:.4f}")

    # History + plot
    results_io.write_training_history(run_dir, history, train_secs,
                                      best_epoch=best_epoch, best_stage=best_stage)
    _plot_training_history(history, run_dir)

    model = tf.keras.models.load_model(checkpoint.filepath, compile=False)

    # Test predictions → source-of-truth slice CSV
    # Test inputs need the same backbone preprocess_input as train/val; the
    # loader returns raw 0-255 pixels and the model has no normalisation layer.
    probs = model.predict(preprocess_fn(X_test.copy()), batch_size=Config.BATCH_SIZE,
                          verbose=0).flatten()
    slice_indices = [_slice_idx_from_filename(f) for f in files_test]
    slice_df = results_io.write_predictions_slice(
        run_dir,
        subject_ids=subjects_test_per_slice,
        slice_indices=slice_indices,
        y_true=y_test,
        y_prob=probs,
    )

    # Full plot set + metric JSONs in one shot. evaluate_run reads the
    # predictions CSV we just wrote.
    participants_df = _load_participants(DEFAULT_PARTICIPANTS)
    metrics = evaluate_run(run_dir, participants_df=participants_df)
    if metrics is None:
        # extremely unlikely (we just wrote the CSV) — fall back to write_metrics
        slice_m, subj_m = results_io.write_metrics(run_dir, slice_df)
    else:
        slice_m, subj_m = metrics['slice'], metrics['subject']
    print(f"\n  {fold_label or 'Test'} (slice-level):   "
          f"acc={slice_m['accuracy']:.3f}  auc={slice_m['roc_auc']:.3f}")
    print(f"  {fold_label or 'Test'} (subject-level): "
          f"acc={subj_m['accuracy']:.3f}  auc={subj_m['roc_auc']:.3f}")

    return {'slice': slice_m, 'subject': subj_m, 'train_seconds': train_secs,
            'best_epoch': best_epoch, 'best_stage': best_stage}


# -------------------------------------------------------------------------
# entry point
# -------------------------------------------------------------------------

def train_model():
    print('='*60); print(f"EXPERIMENT: {Config.EXPERIMENT_NAME}"); print('='*60)
    Config.print_config()

    # Seed Python, NumPy and TensorFlow (weight init, shuffling, dropout).
    # DETERMINISTIC_OPS=1 also forces deterministic GPU kernels, making
    # repeated runs bit-identical at roughly 5x the training time.
    keras.utils.set_random_seed(Config.TRAIN_SEED)
    if Config.DETERMINISTIC_OPS:
        tf.config.experimental.enable_op_determinism()

    loader = MRIDataLoader(Config)
    Config.make_run_dirs()  # creates results/<EXP_NAME>/

    if Config.USE_KFOLD:
        # ---- k-fold ----
        folds = loader.get_kfold_splits(n_folds=Config.N_FOLDS)
        fold_results = []
        for k, data in enumerate(folds, 1):
            run_dir = os.path.join(Config.RUN_DIR, f'fold_{k}')
            metrics = _train_one_fold(
                *data['train'][:2], *data['val'][:2],
                *data['test'],
                run_dir=run_dir, fold_label=f' fold {k}/{Config.N_FOLDS}',
            )
            metrics['fold'] = k
            fold_results.append(metrics)

        # Aggregate
        def agg(level, key):
            vals = [r[level][key] for r in fold_results if not np.isnan(r[level][key])]
            return {'mean': float(np.mean(vals)),
                    'std':  float(np.std(vals))}

        summary = {
            'n_folds': Config.N_FOLDS,
            'slice':   {k: agg('slice',   k) for k in ('accuracy','precision','recall','roc_auc')},
            'subject': {k: agg('subject', k) for k in ('accuracy','precision','recall','roc_auc')},
            'fold_results': fold_results,
        }
        with open(os.path.join(Config.RUN_DIR, 'cv_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print('\n' + '='*60); print('K-FOLD SUMMARY (mean ± std)'); print('='*60)
        for level in ('slice', 'subject'):
            print(f"\n  {level}-level:")
            for k, v in summary[level].items():
                print(f"    {k:9s}  {v['mean']:.3f} ± {v['std']:.3f}")

    else:
        # ---- single split ----
        data = loader.load_and_split()
        _train_one_fold(
            *data['train'][:2], *data['val'][:2], *data['test'],
            run_dir=Config.RUN_DIR,
        )

    print(f"\nResults saved in: {Config.RUN_DIR}")


if __name__ == '__main__':
    train_model()
