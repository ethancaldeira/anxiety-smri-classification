#!/usr/bin/env python3
"""
Extract 2D slices (axial, coronal or sagittal) from 3D MRI volumes.

Used for the raw GSP T1w volumes, the CAT12 mwp1 maps and the ABIDE ANTs
cortical-thickness maps. Each volume is reoriented to RAS. Along the chosen
axis, every slice in the index range [start, end) that contains non-zero
voxels (by default start and end are the first and last non-empty slice) is
min-max scaled to 0-255, rotated 90 degrees for display and saved as

    <output_dir>/<axis>/<subject>/<subject>_slice_<index>.png

where <index> is the slice position in the RAS-reoriented volume and
<subject> is the file name without the _T1w / mwp1 / .nii(.gz) parts. The
selection of the 40 slices nearest each subject's median slice happens at
load time (src/data_loader.py).

Usage:
    python preprocessing/scripts/extract_slices.py INPUT_DIR OUTPUT_DIR \
        --axis axial --pattern '*_T1w.nii.gz'
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image
import argparse
from pathlib import Path

def normalize_slice(slice_2d):
    """Normalize slice to 0-255 range for PNG"""
    slice_min = slice_2d.min()
    slice_max = slice_2d.max()
    
    if slice_max > slice_min:
        normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255)
    else:
        normalized = np.zeros_like(slice_2d)
    
    return normalized.astype(np.uint8)

def extract_slices(input_file, output_dir, axis='axial', start_slice=None, end_slice=None):
    """
    Extract 2D slices from 3D volume
    
    Args:
        input_file: Path to .nii file
        output_dir: Directory to save slices
        axis: 'axial', 'sagittal', or 'coronal'
        start_slice: First slice to extract (None = auto)
        end_slice: Last slice to extract (None = auto)
    """
    
    # Load image. Force RAS (Right-Anterior-Superior) orientation so the
    # script's axis_map ({axial:2, coronal:1, sagittal:0}) is guaranteed to
    # match anatomical labels regardless of the scanner-native orientation
    # the NIfTI was stored in.
    try:
        img = nib.load(input_file)
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return 0

    print(f"Processing: {input_file}")
    print(f"  Shape: {data.shape}  (canonical RAS)")
    print(f"  Value range: {data.min():.4f} to {data.max():.4f}")
    
    # Remove common prefixes/suffixes
    subject_id = Path(input_file).stem
    if subject_id.endswith('.nii'):
        subject_id = subject_id[:-4]
    for pattern in ['_T1w', 'mwp1', 'mwp2']:
        subject_id = subject_id.replace(pattern, '')
    subject_id = subject_id.strip('_') # Clean up extra underscores
    
    # Create output directory
    output_path = Path(output_dir) / axis / subject_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine slice axis
    axis_map = {'axial': 2, 'sagittal': 0, 'coronal': 1}
    slice_axis = axis_map.get(axis, 2)
    
    # Auto-detect non-zero slice range if not specified
    if start_slice is None or end_slice is None:
        # Find slices with brain tissue (non-zero values)
        slice_sums = np.sum(data, axis=tuple(i for i in range(3) if i != slice_axis))
        non_zero_slices = np.where(slice_sums > 0)[0]
        
        if len(non_zero_slices) == 0:
            print(f"  WARNING: No non-zero slices found!")
            return 0
        
        if start_slice is None:
            start_slice = non_zero_slices[0]
        if end_slice is None:
            end_slice = non_zero_slices[-1]
    
    # Extract slices
    num_slices = data.shape[slice_axis]
    start_slice = max(0, start_slice)
    end_slice = min(num_slices, end_slice)
    
    slices_extracted = 0
    
    for slice_idx in range(start_slice, end_slice):
        # Extract 2D slice
        if slice_axis == 0:
            slice_2d = data[slice_idx, :, :]
        elif slice_axis == 1:
            slice_2d = data[:, slice_idx, :]
        else:  # axial (2)
            slice_2d = data[:, :, slice_idx]
        
        # Skip empty slices
        if slice_2d.max() == 0:
            continue
        
        # Normalize to 0-255
        slice_normalized = normalize_slice(slice_2d)
        
        # Rotate 90° for display orientation
        slice_normalized = np.rot90(slice_normalized)
        
        # Save as PNG
        output_filename = output_path / f"{subject_id}_slice_{slice_idx:03d}.png"
        Image.fromarray(slice_normalized).save(output_filename)
        
        slices_extracted += 1
    
    print(f"  Extracted {slices_extracted} slices to {output_path}")
    return slices_extracted

def main():
    parser = argparse.ArgumentParser(description='Extract 2D slices from 3D MRI volumes')
    parser.add_argument('input_dir', help='Directory containing .nii files')
    parser.add_argument('output_dir', help='Directory to save slices')
    parser.add_argument('--axis', default='axial', choices=['axial', 'sagittal', 'coronal'])
    parser.add_argument('--start', type=int, default=None, help='First slice index')
    parser.add_argument('--end', type=int, default=None, help='Last slice index')
    parser.add_argument('--pattern', default='*.nii', help='File pattern to match')
    
    args = parser.parse_args()
    
    # Find all matching files
    input_path = Path(args.input_dir)
    nii_files = sorted(input_path.glob(args.pattern))
    
    if not nii_files:
        print(f"No files found matching {args.pattern} in {args.input_dir}")
        return
    
    print(f"Found {len(nii_files)} files to process")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    total_slices = 0
    for nii_file in nii_files:
        slices = extract_slices(
            str(nii_file), 
            args.output_dir, 
            axis=args.axis,
            start_slice=args.start,
            end_slice=args.end
        )
        total_slices += slices
        print()
    
    print("="*60)
    print(f"COMPLETE: Extracted {total_slices} total slices from {len(nii_files)} volumes")

if __name__ == '__main__':
    main()
