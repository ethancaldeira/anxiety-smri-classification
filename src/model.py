"""
CNN model with transfer learning for anxiety classification.

- Backbone with ImageNet (default) or RadImageNet weights, plus a small
  dense head. Backbone preprocess_input is applied per batch by
  MRIBatchSequence (src/mri_augment.py), not inside the model.
- Two-stage fine-tuning helpers: stage 1 freezes the backbone and trains
  only the head; stage 2 unfreezes for joint fine-tuning. train.py drives
  the staging.
- Learning rate: linear warmup -> cosine decay to MIN_LR, via a callback.
"""

import math
import os

import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, InceptionV3, DenseNet121,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


BASE_MODELS = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'InceptionV3': InceptionV3,
    'DenseNet121': DenseNet121,
}


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
# NOTE: Augmentation lives in src/mri_augment.py (TorchIO-based) and runs
# per-batch in MRIBatchSequence on RAW pixels before backbone preprocessing.
# It is intentionally NOT a Keras layer here — see that module for rationale.


def _load_backbone(config):
    if config.BASE_MODEL not in BASE_MODELS:
        raise ValueError(f"Unknown base model: {config.BASE_MODEL}")
    cls = BASE_MODELS[config.BASE_MODEL]
    input_shape = (*config.MODEL_INPUT_SIZE, 3)

    if getattr(config, 'USE_RADIMAGENET', False):
        backbone = cls(weights=None, include_top=False, input_shape=input_shape)
        wpath = getattr(config, 'RADIMAGENET_WEIGHTS_PATH', None)
        if not wpath or not os.path.exists(wpath):
            raise FileNotFoundError(
                f"RadImageNet weights for {config.BASE_MODEL} not found: {wpath}\n"
                f"Download the no-top .h5 files into RADIMAGENET_DIR "
                f"(default: pretrained_weights/radimagenet/).")
        # Load by layer order, strictly. The RadImageNet .h5 layer names
        # don't match Keras 3's (InceptionV3 'conv2d_94' vs 'conv2d',
        # DenseNet 'conv1/conv' vs 'conv1_conv'), so a by-name load would
        # skip layers. Topology is identical, so the ordered load is exact
        # (376/376, 604/604, 318/318 tensors).
        backbone.load_weights(wpath)
        print(f"  RadImageNet weights loaded from {wpath}")
    else:
        backbone = cls(weights='imagenet', include_top=False, input_shape=input_shape)
        print(f"  ImageNet weights loaded ({config.BASE_MODEL})")
    return backbone


def create_model(config):
    """Build the transfer-learning model (backbone -> head)."""
    backbone = _load_backbone(config)
    backbone.trainable = not config.FREEZE_BASE
    print(f"  Backbone trainable: {backbone.trainable}")

    inputs = tf.keras.Input(shape=(*config.MODEL_INPUT_SIZE, 3), name='input_image')
    x = backbone(inputs, training=None)
    x = GlobalAveragePooling2D()(x)
    for i, units in enumerate(config.DENSE_LAYERS, start=1):
        x = Dense(units, activation=config.ACTIVATION, name=f'dense_{i}')(x)
        x = Dropout(config.DROPOUT_RATE, name=f'dropout_{i}')(x)
    outputs = Dense(config.NUM_CLASSES, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='anxiety_classifier')

    # Compiled here for stage-1 head-only training (config.STAGE1_LR if
    # frozen, else config.LEARNING_RATE). train.py recompiles for stage 2.
    initial_lr = (getattr(config, 'STAGE1_LR', config.LEARNING_RATE)
                  if config.FREEZE_BASE else config.LEARNING_RATE)
    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    print(f"  Model params: total={model.count_params():,}  trainable={trainable:,}")
    return model


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------

def make_lr_schedule(base_lr, total_epochs, warmup_epochs, min_lr):
    """Return a Python fn(epoch, current_lr) usable with LearningRateScheduler."""
    def fn(epoch, _current_lr):
        if epoch < warmup_epochs:
            # linear warmup from min_lr -> base_lr
            frac = (epoch + 1) / max(1, warmup_epochs)
            return float(min_lr + (base_lr - min_lr) * frac)
        # cosine decay from base_lr -> min_lr over remaining epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr + (base_lr - min_lr) * cos)
    return fn


# ---------------------------------------------------------------------------
# stage-aware callbacks
# ---------------------------------------------------------------------------

def make_checkpoint(run_dir):
    """Lowest-val_loss checkpoint, shared by both stages.

    ModelCheckpoint keeps its best-so-far value across fit() calls, so passing
    the same instance to stage 1 and stage 2 makes best.keras the best epoch
    over the whole run.
    """
    return ModelCheckpoint(
        os.path.join(run_dir, 'best.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
    )


def get_stage1_callbacks(checkpoint):
    """Stage 1 runs for STAGE1_EPOCHS with no early stopping."""
    return [checkpoint]


def get_stage2_callbacks(config, checkpoint):
    """Stage 2: warmup+cosine LR and early stopping."""
    total = max(1, config.EPOCHS - config.STAGE1_EPOCHS)
    schedule = make_lr_schedule(
        base_lr=config.LEARNING_RATE,
        total_epochs=total,
        warmup_epochs=min(config.WARMUP_EPOCHS, total),
        min_lr=config.MIN_LR,
    )
    return [
        LearningRateScheduler(schedule, verbose=0),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            verbose=1,
        ),
        checkpoint,
    ]


# ---------------------------------------------------------------------------
# helpers for train.py
# ---------------------------------------------------------------------------

def unfreeze_for_stage2(model, config):
    """Unfreeze every layer in the backbone (it lives inside model.layers)."""
    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                sub.trainable = True
    # Recompile with the stage-2 LR; the warmup+cosine schedule is applied by
    # the LearningRateScheduler callback.
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    trainable = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    print(f"  [stage 2] unfroze backbone. Trainable params now: {trainable:,}")
    return model
