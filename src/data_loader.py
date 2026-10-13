"""
Data loader for anxiety MRI classification (GSP; ABIDE via abide/config.py).
Handles loading slices, labels, and splitting by participant.
"""

import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split


def _get_preprocess_input(base_model_name, use_radimagenet):
    """
    Return the per-backbone preprocess_input function.

    For ImageNet weights, each Keras application has its own preprocessing
    (mean subtraction, channel order, scaling), matching the distribution the
    weights were trained on.

    For RadImageNet weights, inputs are scaled to [0, 1] (x / 255), as in the
    RadImageNet reference implementation.
    """
    if use_radimagenet:
        return lambda x: x / 255.0
    name = base_model_name.lower()
    if 'resnet' in name:
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif 'densenet' in name:
        from tensorflow.keras.applications.densenet import preprocess_input
    elif 'inception' in name:
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif 'vgg' in name:
        from tensorflow.keras.applications.vgg16 import preprocess_input
    else:
        # safe fallback
        return lambda x: (x / 127.5) - 1.0
    return preprocess_input


class MRIDataLoader:
    """Load MRI slices and labels, and split them by participant."""
    
    def __init__(self, config):
        self.config = config
        self.slices_dir = config.SLICES_DIR
        self.image_size = config.MODEL_INPUT_SIZE
        self.dataset = config.DATASET

        # Load participants metadata
        participants_file = config.PARTICIPANTS_FILE
        if not os.path.exists(participants_file):
            raise FileNotFoundError(f"Participants file not found: {participants_file}")
        
        self.participants_df = pd.read_csv(participants_file, sep='\t')
        print(f"Loaded {len(self.participants_df)} participants from metadata")
        
        # Filter to the GSP subset (USE_GSP_SUBSET=1)
        if self.dataset == 'GSP' and getattr(config, 'USE_GSP_SUBSET', False):
            subset_file = config.GSP_SUBSET_FILE
            if not os.path.exists(subset_file):
                raise FileNotFoundError(
                    f"GSP subset file not found: {subset_file}\n"
                    f"Generate it with `python splits/make_gsp_split.py`, or set "
                    f"USE_GSP_SUBSET=0 to use every subject."
                )
            with open(subset_file, 'r') as f:
                subset_subjects = [line.strip() for line in f.readlines()]

            self.participants_df = self.participants_df[
                self.participants_df['participant_id'].isin(subset_subjects)
            ].copy()

            print(f"  ✓ Filtered to GSP subset: {len(self.participants_df)} subjects")

        sex = getattr(config, 'SEX_FILTER', None)
        if sex:
            self.participants_df = self.participants_df[self.participants_df['Sex'] == sex].copy()
            print(f"  ✓ Filtered to Sex == {sex}: {len(self.participants_df)} subjects")
        
        # ---- Binary label from STAI_tAnxiety at the configured cutoff -------
        # Anxiety_Binary is computed from STAI_tAnxiety >= Config.STAI_CUTOFF;
        # any column of that name already in the participants TSV is
        # overwritten. GSP only.
        target_col = config.TARGET_COLUMN
        cutoff = getattr(config, 'STAI_CUTOFF', None)
        if self.dataset == 'GSP' and cutoff is not None and 'STAI_tAnxiety' in self.participants_df.columns:
            stai = pd.to_numeric(self.participants_df['STAI_tAnxiety'], errors='coerce')
            recomputed = (stai >= cutoff).astype('Int64')
            recomputed[stai.isna()] = pd.NA
            self.participants_df[target_col] = recomputed
            n_pos = int((recomputed == 1).sum())
            n_neg = int((recomputed == 0).sum())
            print(f"  Binary label recomputed at STAI_tAnxiety >= {cutoff}: "
                  f"{n_pos} positive / {n_neg} negative "
                  f"({100.0 * n_pos / max(1, n_pos + n_neg):.1f}% positive)")

        # ---- Pipeline-validation overrides ------------------------------
        # If TARGET_OVERRIDE is set, replace TARGET_COLUMN with a binary derived
        # from the override column. If LABEL_PERMUTE is set, shuffle the resulting
        # label at the subject level with a separate seed.
        override = getattr(config, 'TARGET_OVERRIDE', None)
        if override:
            if override not in self.participants_df.columns:
                raise ValueError(
                    f"TARGET_OVERRIDE '{override}' not found in participants file.\n"
                    f"Available columns: {list(self.participants_df.columns)}"
                )
            if override == 'Sex':
                # F=1, M=0 (arbitrary but documented)
                mapped = self.participants_df[override].map({'F': 1, 'M': 0})
            else:
                # Generic numeric coercion fallback
                mapped = pd.to_numeric(self.participants_df[override], errors='coerce')
            self.participants_df[target_col] = mapped
            print(f"  [sanity] TARGET_OVERRIDE active: using '{override}' as binary target")

        if target_col not in self.participants_df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in participants file.\n"
                f"Available columns: {list(self.participants_df.columns)}"
            )

        if getattr(config, 'LABEL_PERMUTE', False):
            seed = getattr(config, 'PERMUTE_SEED', 1234)
            rng = np.random.default_rng(seed)
            valid_mask = self.participants_df[target_col].notna()
            shuffled = self.participants_df.loc[valid_mask, target_col].values.copy()
            rng.shuffle(shuffled)
            self.participants_df.loc[valid_mask, target_col] = shuffled
            print(f"  [sanity] LABEL_PERMUTE active: shuffled '{target_col}' "
                  f"({valid_mask.sum()} subjects, seed={seed})")

        # Print label distribution
        label_counts = Counter(self.participants_df[target_col].dropna())
        print(f"Label distribution for {target_col}: {dict(label_counts)}")
    
    def get_label_for_subject(self, subject_id):
        """
        Get anxiety label for a subject.
        
        Args:
            subject_id: Subject identifier (GSP: SubNNNN_S1, ABIDE: sub-NNNNN)

        Returns:
            0 (control), 1 (anxiety), or None if not found
        """
        subject_row = self.participants_df[
            self.participants_df[self.config.SUBJECT_ID_COLUMN] == subject_id
        ]

        if len(subject_row) > 0:
            label = subject_row.iloc[0][self.config.TARGET_COLUMN]
            return int(label) if pd.notna(label) else None
        
        return None
    
    def load_image(self, image_path):
        """Load a single image as raw 0–255 RGB.

        Returns raw intensities; backbone-specific preprocess_input is applied
        per-batch by MRIBatchSequence (after TorchIO augmentation, which must
        run on raw intensities for RandomBiasField to be physically meaningful).
        """
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize(self.image_size, Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32)            # 0–255
            img_array = np.stack([img_array] * 3, axis=-1)         # tile to RGB
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_all_slices(self):
        """
        Load all slices from all subjects.
        
        Returns:
            Tuple of (images, labels, filenames, subject_ids)
        """
        images = []
        labels = []
        filenames = []
        subject_ids = []
        
        # Subject directory names: GSP SubNNNN_S1, ABIDE sub-NNNNN
        pattern = 'Sub*_S*' if self.dataset == 'GSP' else 'sub-*'
        
        subject_dirs = sorted(glob.glob(os.path.join(self.slices_dir, pattern)))

        print(f"\nFound {len(subject_dirs)} subject directories")
        
        subjects_with_labels = 0
        subjects_without_labels = 0
        
        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)

            # Get label
            label = self.get_label_for_subject(subject_id)
            
            if label is None:
                subjects_without_labels += 1
                continue
            
            subjects_with_labels += 1
            
            # Find all PNG slices
            slice_files = sorted(glob.glob(os.path.join(subject_dir, '*.png')))
            
            # === CENTRAL SLICE SELECTION ===
            if hasattr(self.config, 'USE_CENTRAL_SLICES') and self.config.USE_CENTRAL_SLICES:
                n_total = len(slice_files)
                n_select = getattr(self.config, 'N_CENTRAL_SLICES', 30)
                
                if n_total > n_select:
                    # Extract slice numbers
                    slice_info = []
                    for idx, f in enumerate(slice_files):
                        basename = os.path.basename(f)
                        try:
                            if '_slice_' in basename:
                                slice_num = int(basename.split('_slice_')[1].split('.')[0])
                            else:
                                slice_num = idx
                            slice_info.append((idx, slice_num, f))
                        except:
                            slice_info.append((idx, idx, f))
                    
                    # Find median (brain center)
                    slice_numbers = [s[1] for s in slice_info]
                    median_slice = np.median(slice_numbers)
                    
                    # Calculate distance from median
                    slice_distances = [(s[0], s[2], abs(s[1] - median_slice)) for s in slice_info]
                    
                    # Take N closest to median
                    slice_distances.sort(key=lambda x: x[2])
                    selected = slice_distances[:n_select]
                    selected.sort(key=lambda x: x[0])  # Restore order
                    
                    slice_files = [s[1] for s in selected]
                    
                    print(f"  {subject_id}: Selected {len(slice_files)}/{n_total} "
                          f"central slices (median={median_slice:.0f}, label={label})")
                else:
                    print(f"  {subject_id}: Using all {n_total} slices "
                          f"(fewer than {n_select} requested, label={label})")
            else:
                print(f"  {subject_id}: {len(slice_files)} slices, label={label}")
            
            # Load slices
            for slice_file in slice_files:
                img = self.load_image(slice_file)
                if img is None:
                    continue
                
                images.append(img)
                labels.append(label)
                filenames.append(os.path.basename(slice_file))
                subject_ids.append(subject_id)
        
        print(f"\n✓ Subjects with labels: {subjects_with_labels}")
        print(f"✗ Subjects without labels: {subjects_without_labels}")
        print(f"Total slices loaded: {len(images)}")
        print(f"Label distribution: {Counter(labels)}")
        
        return (
            np.array(images),
            np.array(labels),
            filenames,
            subject_ids
        )
    
    def save_split(self, train_subjects, val_subjects, test_subjects, split_file):
        """Save train/val/test subject split to JSON file"""
        split_data = {
            'train': list(train_subjects),
            'val': list(val_subjects),
            'test': list(test_subjects),
            'random_seed': self.config.RANDOM_SEED,
            'dataset': self.config.DATASET,
            'experiment_type': self.config.EXPERIMENT_TYPE,
            'view_mode': self.config.VIEW_MODE
        }
        
        os.makedirs(os.path.dirname(split_file), exist_ok=True)

        if not os.path.exists(split_file):
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"✓ Split saved to: {split_file}")
        else:
            print(f"✓ Split already exists, not overwriting: {split_file}")
    
    def load_split(self, split_file):
        """Load train/val/test subject split from JSON file"""
        if not os.path.exists(split_file):
            return None
        
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        print(f"\n✓ Loaded existing split from: {split_file}")
        print(f"  Random seed: {split_data.get('random_seed', 'N/A')}")
        print(f"  Train subjects: {len(split_data['train'])}")
        print(f"  Val subjects: {len(split_data['val'])}")
        print(f"  Test subjects: {len(split_data['test'])}")
        
        return split_data['train'], split_data['val'], split_data['test']
    
    def split_by_participant(self, images, labels, filenames, subject_ids, force_new_split=False):
        """
        Split data by participant (not by slices), so that all slices of a
        participant fall in exactly one of train/val/test.

        Args:
            force_new_split: If True, creates new split even if saved split exists
        """
        # Define split file path (shared across all experiments for same dataset/view)
        sex = getattr(self.config, 'SEX_FILTER', None)
        split_filename = (f"data_split_{self.config.DATASET}{'_' + sex if sex else ''}"
                          f"_seed{self.config.RANDOM_SEED}.json")
        split_file = (getattr(self.config, 'SPLIT_FILE', None)
                      or os.path.join(self.config.DATA_DIR, split_filename))

        # Try to load existing split first
        if not force_new_split:
            existing_split = self.load_split(split_file)
            if existing_split is not None:
                train_subjects, val_subjects, test_subjects = existing_split

                # Verify all subjects exist in current data
                all_subjects = set(subject_ids)
                train_set = set(train_subjects)
                val_set = set(val_subjects)
                test_set = set(test_subjects)

                if train_set.issubset(all_subjects) and val_set.issubset(all_subjects) and test_set.issubset(all_subjects):
                    print("✓ Using saved split - ensures consistency across experiments!")

                    # Convert to arrays for subsequent processing
                    train_subjects = np.array(train_subjects)
                    val_subjects = np.array(val_subjects)
                    test_subjects = np.array(test_subjects)
                elif getattr(self.config, 'SPLIT_FILE', None):
                    # An explicitly given split must be used as is, so that
                    # runs sharing it are evaluated on the same subjects.
                    missing = sorted((train_set | val_set | test_set) - all_subjects)
                    shown = ', '.join(missing[:20])
                    more = f" and {len(missing) - 20} more" if len(missing) > 20 else ""
                    raise ValueError(
                        f"SPLIT_FILE {split_file} lists {len(missing)} subject(s) "
                        f"with no slices in the loaded data: {shown}{more}")
                else:
                    print("⚠ Warning: Saved split contains subjects not in current data")
                    print("  Creating new split...")
                    existing_split = None
        else:
            existing_split = None
            print("⚠ Force new split requested - creating fresh split")
        
        # Create new split if needed
        if existing_split is None:
            # Get unique subjects (sorted, so the split depends only on the seed)
            unique_subjects = sorted(set(subject_ids))
            print(f"\nCreating NEW split...")
            print(f"Total unique subjects: {len(unique_subjects)}")
            
            # Map subjects to labels (for stratification)
            subject_labels = {}
            for sid in unique_subjects:
                idx = subject_ids.index(sid)
                subject_labels[sid] = labels[idx]
            
            subjects_array = np.array(unique_subjects)
            labels_array = np.array([subject_labels[s] for s in unique_subjects])
            
            print(f"Subject label distribution: {Counter(labels_array)}")
            
            # Split subjects: train / (val + test)
            train_subjects, temp_subjects = train_test_split(
                subjects_array,
                test_size=(self.config.VAL_SPLIT + self.config.TEST_SPLIT),
                random_state=self.config.RANDOM_SEED,
                stratify=labels_array
            )
            
            # Split temp: val / test
            temp_labels = np.array([subject_labels[s] for s in temp_subjects])
            
            val_subjects, test_subjects = train_test_split(
                temp_subjects,
                test_size=self.config.TEST_SPLIT / (self.config.VAL_SPLIT + self.config.TEST_SPLIT),
                random_state=self.config.RANDOM_SEED,
                stratify=temp_labels
            )
            
            # Save the split for future use
            self.save_split(train_subjects, val_subjects, test_subjects, split_file)
        
        print(f"\nTrain subjects: {len(train_subjects)}")
        print(f"Val subjects: {len(val_subjects)}")
        print(f"Test subjects: {len(test_subjects)}")
        
        # Create masks
        train_mask = np.array([sid in train_subjects for sid in subject_ids])
        val_mask = np.array([sid in val_subjects for sid in subject_ids])
        test_mask = np.array([sid in test_subjects for sid in subject_ids])
        
        # Split data
        X_train = images[train_mask]
        y_train = labels[train_mask]
        files_train = [f for f, m in zip(filenames, train_mask) if m]
        
        X_val = images[val_mask]
        y_val = labels[val_mask]
        files_val = [f for f, m in zip(filenames, val_mask) if m]
        
        X_test = images[test_mask]
        y_test = labels[test_mask]
        files_test = [f for f, m in zip(filenames, test_mask) if m]

        # Per-slice subject IDs (length matches X/y/files) — needed for
        # subject-level metric aggregation in results_io.
        subjects_train_per_slice = [s for s, m in zip(subject_ids, train_mask) if m]
        subjects_val_per_slice   = [s for s, m in zip(subject_ids, val_mask) if m]
        subjects_test_per_slice  = [s for s, m in zip(subject_ids, test_mask) if m]

        print(f"\nTrain slices: {len(X_train)} (labels: {Counter(y_train)})")
        print(f"Val slices: {len(X_val)} (labels: {Counter(y_val)})")
        print(f"Test slices: {len(X_test)} (labels: {Counter(y_test)})")

        # Tuple layout: (X, y, files, unique_subjects, subjects_per_slice).
        # train.py reads index [4] (per-slice array) for the predictions CSV.
        return {
            'train': (X_train, y_train, files_train, list(train_subjects), subjects_train_per_slice),
            'val':   (X_val,   y_val,   files_val,   list(val_subjects),   subjects_val_per_slice),
            'test':  (X_test,  y_test,  files_test,  list(test_subjects),  subjects_test_per_slice),
        }
    
    def load_and_split(self):
        """Main method: load all data and split by participant"""
        print("=" * 60)
        print("LOADING MRI SLICES")
        print("=" * 60)
        
        images, labels, filenames, subject_ids = self.load_all_slices()
        
        if len(images) == 0:
            raise ValueError("No images loaded! Check data paths and labels.")
        
        print("\n" + "=" * 60)
        print("SPLITTING BY PARTICIPANT")
        print("=" * 60)
        
        splits = self.split_by_participant(images, labels, filenames, subject_ids)
        
        return splits
    
    def get_kfold_splits(self, n_folds=5):
        """
        Stratified k-fold cross-validation by subject.

        Each fold is the test set exactly once. Within the remaining folds, a
        stratified VAL_SPLIT share of the whole subset is held out for
        validation and the rest is used for training (for 200 subjects and
        5 folds: 130 train / 30 val / 40 test per fold).

        Returns:
            List of k splits with train/val/test data (same tuple layout as
            split_by_participant).
        """
        print("=" * 60)
        print("LOADING MRI SLICES FOR K-FOLD CV")
        print("=" * 60)

        images, labels, filenames, subject_ids = self.load_all_slices()

        if len(images) == 0:
            raise ValueError("No images loaded!")

        subject_labels = {}
        for sid, label in zip(subject_ids, labels):
            subject_labels.setdefault(sid, label)
        subjects_array = np.array(sorted(subject_labels))
        labels_array = np.array([subject_labels[s] for s in subjects_array])
        subject_ids_array = np.array(subject_ids)

        print(f"\nTotal unique subjects: {len(subjects_array)}")
        print(f"Subject label distribution: {Counter(labels_array)}")

        seed = self.config.RANDOM_SEED
        val_fraction = self.config.VAL_SPLIT / (1 - 1 / n_folds)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        fold_splits = []
        for fold_idx, (rest_idx, test_idx) in enumerate(skf.split(subjects_array, labels_array)):
            test_subjects = subjects_array[test_idx]
            train_subjects, val_subjects = train_test_split(
                subjects_array[rest_idx],
                test_size=val_fraction,
                stratify=labels_array[rest_idx],
                random_state=seed,
            )

            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
            print(f"Train / val / test subjects: "
                  f"{len(train_subjects)} / {len(val_subjects)} / {len(test_subjects)}")

            fold = {}
            for part, part_subjects in (('train', train_subjects),
                                        ('val', val_subjects),
                                        ('test', test_subjects)):
                mask = np.isin(subject_ids_array, part_subjects)
                fold[part] = (
                    images[mask],
                    labels[mask],
                    [f for f, m in zip(filenames, mask) if m],
                    list(part_subjects),
                    [s for s, m in zip(subject_ids, mask) if m],
                )
                print(f"{part.capitalize()} slices: {mask.sum()} "
                      f"(labels: {Counter(labels[mask])})")
            fold_splits.append(fold)

        return fold_splits


# Test the data loader
if __name__ == '__main__':
    from config import Config
    
    Config.print_config()
    
    loader = MRIDataLoader(Config)
    data = loader.load_and_split()
    
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60)
    print(f"Train: {data['train'][0].shape}")
    print(f"Val: {data['val'][0].shape}")
    print(f"Test: {data['test'][0].shape}")
