"""
CAT12 preprocessing of the GSP T1w volumes (one CAT12 job per subject).

Usage (from the repo root; paths and CAT12 location: config_datasets.py):
    # Preprocess every subject in the participants file, 8 jobs in parallel
    python preprocessing/scripts/pipeline.py --dataset gsp --all --parallel 8

    # Preprocess the subjects listed in a file (one ID per line)
    python preprocessing/scripts/pipeline.py --dataset gsp --subject-list data/GSP/GSP_subject_list_200.txt

    # Preprocess specific subjects
    python preprocessing/scripts/pipeline.py --dataset gsp --subjects SubNNNN_S1 SubNNNN_S1
"""

import argparse
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from functools import partial
from typing import List
import sys

from config_datasets import DATASETS
from preprocess import preprocess_subject


def read_subject_list(path: Path) -> List[str]:
    """Subject IDs from a text file, one per line (blank lines ignored)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def process_wrapper(subject_id: str, config) -> dict:
    """Wrapper for parallel processing"""
    success = preprocess_subject(subject_id, config, verbose=False)
    return {'subject': subject_id, 'success': success}


def main():
    parser = argparse.ArgumentParser(
        description="CAT12 preprocessing pipeline (paths: see config_datasets.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all GSP subjects with 8 parallel jobs
  python preprocessing/scripts/pipeline.py --dataset gsp --all --parallel 8

  # Process the subjects listed in a file
  python preprocessing/scripts/pipeline.py --dataset gsp --subject-list data/GSP/GSP_subject_list_200.txt

  # Process specific subjects
  python preprocessing/scripts/pipeline.py --dataset gsp --subjects SubNNNN_S1 SubNNNN_S1
        """
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['gsp'],
        help='Dataset to preprocess'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Preprocess all subjects in dataset'
    )
    
    parser.add_argument(
        '--subject-list',
        type=Path,
        metavar='FILE',
        help='Preprocess the subject IDs listed in FILE (one per line)'
    )
    
    parser.add_argument(
        '--subjects',
        nargs='+',
        metavar='ID',
        help='Specific subject IDs to preprocess'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        metavar='N',
        help='Number of parallel jobs (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.subject_list, args.subjects]):
        parser.error("Must specify one of: --all, --subject-list FILE, or --subjects")
    
    # Load dataset config
    config = DATASETS[args.dataset]
    
    print("=" * 70)
    print(f"PREPROCESSING PIPELINE: {config.name}")
    print("=" * 70)
    print(f"Age group: {config.age_group}")
    print(f"TPM: {config.tpm_path.name}")
    print(f"Raw dir: {config.raw_dir}")
    print(f"Output dir: {config.preprocessed_dir}")
    print(f"Parallel jobs: {args.parallel}")
    print("=" * 70)
    
    # Determine subjects to process
    if args.all:
        df = pd.read_csv(config.participants_file, sep='\t')
        subjects = df['participant_id'].tolist()
        print(f"\nProcessing ALL {len(subjects)} subjects\n")
        
    elif args.subject_list:
        subjects = read_subject_list(args.subject_list)
        print(f"\nProcessing {len(subjects)} subjects from {args.subject_list}\n")

    elif args.subjects:
        subjects = args.subjects
        print(f"\nProcessing {len(subjects)} specified subjects\n")
    
    # Create output directory
    config.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process subjects
    print("=" * 70)
    print("STARTING PREPROCESSING")
    print("=" * 70)
    
    if args.parallel > 1:
        # Parallel processing
        print(f"Using {args.parallel} parallel workers\n")
        
        with Pool(args.parallel) as pool:
            process_fn = partial(process_wrapper, config=config)
            results = pool.map(process_fn, subjects)
        
        # Summary
        successes = sum(1 for r in results if r['success'])
        failures = len(results) - successes
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total subjects: {len(subjects)}")
        print(f"Successful: {successes}")
        print(f"Failed: {failures}")
        
        if failures > 0:
            print("\nFailed subjects:")
            for r in results:
                if not r['success']:
                    print(f"  - {r['subject']}")
    
    else:
        # Sequential processing
        print("Sequential processing (use --parallel N for faster processing)\n")
        
        successes = 0
        failures = 0
        
        for i, subject_id in enumerate(subjects, 1):
            print(f"\n[{i}/{len(subjects)}] ", end="")
            success = preprocess_subject(subject_id, config, verbose=True)
            
            if success:
                successes += 1
            else:
                failures += 1
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Total subjects: {len(subjects)}")
        print(f"Successful: {successes}")
        print(f"Failed: {failures}")
    
    print("=" * 70)
    
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
