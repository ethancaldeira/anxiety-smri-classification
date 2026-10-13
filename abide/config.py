"""
ABIDEConfig — overrides src.Config attributes for the ABIDE arm at import time.

Importing this module mutates the class-level attributes of src.Config so that
src.train_model, src.data_loader, src.evaluate etc. behave as if ABIDE were a
first-class dataset, WITHOUT modifying src/config.py itself. GSP-specific
code paths are gated on Config.DATASET == 'GSP' and are not triggered.

Anything that imports `from config import Config` AFTER this module loads
gets the ABIDE-mutated Config. Use abide/train.py as the canonical entry
point — it imports this module first, then dispatches to src.train_model.

The subject split is abide/data_split_ABIDE_seed42.json (774 subjects,
train 503 / val 116 / test 155; the same split anxiety-fmri-classification uses
for its Chapter 7 ABIDE runs). SPLIT_FILE overrides it.
"""
from __future__ import annotations

import os
import sys

# Make src/ importable as `config`, `train`, `data_loader`, etc.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(THIS_DIR), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import Config   # noqa: E402  (sys.path manipulation must come first)


# ---------------------------------------------------------------------------
# ABIDE paths
# ---------------------------------------------------------------------------

ABIDE_DATA_DIR = os.path.join(Config.BASE_DIR, 'data', 'ABIDE')
ABIDE_SPLIT_FILE = os.path.join(THIS_DIR, 'data_split_ABIDE_seed42.json')


def apply_abide_overrides() -> None:
    """Mutate Config attributes in place for ABIDE. Idempotent."""

    Config.DATASET = 'ABIDE'

    # Phenotype file built by abide/download_ants.py from
    # Phenotypic_V1_0b_preprocessed1.csv with an added DX_Binary column
    # (1 = ASD when DX_GROUP==1, 0 = Control when DX_GROUP==2).
    Config.PARTICIPANTS_FILE = os.path.join(ABIDE_DATA_DIR, 'abide_participants.tsv')
    Config.SUBJECT_ID_COLUMN = 'participant_id'
    Config.TARGET_COLUMN     = 'DX_Binary'

    # No analogue to GSP's STAI cutoff; the GSP-specific recompute path in
    # data_loader.py is gated on `self.dataset == 'GSP'`, so leaving this
    # attribute set is harmless (just unused).
    Config.STAI_CUTOFF = None

    # GSP-only subset filter — explicitly disabled.
    Config.USE_GSP_SUBSET = False

    # Fixed subject split shipped with this folder, unless SPLIT_FILE is set.
    Config.SPLIT_FILE = os.environ.get('SPLIT_FILE') or ABIDE_SPLIT_FILE

    # Slices live under data/ABIDE/slices/<raw|preprocessed>/<axis>/.
    # Same path derivation as src/config.py.
    Config.BASE_SLICES_DIR = os.path.join(
        ABIDE_DATA_DIR, 'slices', Config.EXPERIMENT_TYPE
    )
    Config.SLICES_DIR = os.path.join(Config.BASE_SLICES_DIR, Config.VIEW_MODE)


# Apply overrides at import time so this module is "import-and-forget."
apply_abide_overrides()


if __name__ == '__main__':
    # Quick sanity print: confirms which file Config is using, the label
    # column it expects, and where it'll look for slices.
    print(f"DATASET           = {Config.DATASET}")
    print(f"PARTICIPANTS_FILE = {Config.PARTICIPANTS_FILE}")
    print(f"TARGET_COLUMN     = {Config.TARGET_COLUMN}")
    print(f"SLICES_DIR        = {Config.SLICES_DIR}")
    print(f"SPLIT_FILE        = {Config.SPLIT_FILE}")
    print(f"BASE_SLICES_DIR   = {Config.BASE_SLICES_DIR}")
    print(f"EXPERIMENT_TYPE   = {Config.EXPERIMENT_TYPE}")
    print(f"VIEW_MODE         = {Config.VIEW_MODE}")
