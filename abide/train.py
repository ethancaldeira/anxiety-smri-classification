"""
Train one ABIDE sMRI classification cell.

Thin wrapper: imports abide.config (mutates src.Config in-place), then
dispatches to the shared src.train.train_model. Model architecture,
training loop, evaluation and output files are reused unchanged from src/.

Usage (run as a module from the repo root):
    EXPERIMENT_TYPE=raw VIEW_MODE=axial BASE_MODEL=ResNet50 \
        EXPERIMENT_NAME=ch7_abide_ants_axial_ResNet50 python -m abide.train
"""
from __future__ import annotations

# Importing abide.config mutates src.Config and adds src/ to sys.path.
# Must happen BEFORE we import anything from src.
import abide.config  # noqa: F401
from train import train_model  # noqa: E402  — src.train, resolved via sys.path


if __name__ == '__main__':
    train_model()
