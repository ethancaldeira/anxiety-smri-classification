"""CAT12 segmentation of one subject.

Writes a CAT12 batch for the subject and runs it with the CAT12 standalone
(MATLAB Runtime) build. Settings: SPM tissue probability map, affine
regularisation to the ICBM/MNI template ('mni'), medium bias correction
(biasstr 0.5), CAT12's default non-linear registration (geodesic shooting),
no smoothing. The
output used downstream is the modulated warped grey-matter map
<subject>/mri/mwp1<subject>_T1w.nii.
"""

import subprocess
from pathlib import Path
import shutil

from config_datasets import DatasetConfig, CAT12_DIR, MCR_DIR


def create_batch_file(input_file: Path, tpm_path: Path, output_dir: Path) -> Path:
    """Create subject-specific MATLAB batch file"""
    
    template = f"""% CAT12 segmentation batch
matlabbatch{{1}}.spm.tools.cat.estwrite.data = {{'{input_file}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.data_wmh = {{''}};
matlabbatch{{1}}.spm.tools.cat.estwrite.useprior = '';
matlabbatch{{1}}.spm.tools.cat.estwrite.nproc = 0;

% Tissue probability map
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.tpm = {{'{tpm_path}'}};
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.affreg = 'mni';
matlabbatch{{1}}.spm.tools.cat.estwrite.opts.biasstr = 0.5;
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.segmentation.APP = 1070;

% Output: GM/WM/CSF (warped + modulated)
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.native = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.warped = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.GM.mod = 1;

matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.native = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.warped = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.WM.mod = 1;

matlabbatch{{1}}.spm.tools.cat.estwrite.output.CSF.native = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.CSF.warped = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.CSF.mod = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.CSF.dartel = 0;

matlabbatch{{1}}.spm.tools.cat.estwrite.output.label.native = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.label.warped = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.label.dartel = 0;

matlabbatch{{1}}.spm.tools.cat.estwrite.output.bias.native = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.bias.warped = 1;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.bias.dartel = 0;

matlabbatch{{1}}.spm.tools.cat.estwrite.output.las.native = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.las.warped = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.las.dartel = 0;

matlabbatch{{1}}.spm.tools.cat.estwrite.output.jacobianwarped = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.warps = [1 0];
matlabbatch{{1}}.spm.tools.cat.estwrite.output.surface = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.output.BIDS.BIDSno = 1;

matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.admin.lazy = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.admin.ignoreErrors = 0;
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.admin.verb = 2;
matlabbatch{{1}}.spm.tools.cat.estwrite.extopts.admin.print = 2;
"""
    
    batch_file = output_dir / "batch_temp.m"
    batch_file.write_text(template)
    return batch_file


def preprocess_subject(subject_id: str, config: DatasetConfig, verbose: bool = True) -> bool:
    """
    Preprocess a single subject with CAT12
    
    Args:
        subject_id: Subject identifier (e.g., 'SubNNNN_S1')
        config: Dataset configuration
        verbose: Print progress
        
    Returns:
        True if successful, False otherwise
    """
    
    # Setup paths
    input_file = config.raw_dir / f"{subject_id}{config.file_suffix}"
    output_dir = config.preprocessed_dir / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        if verbose:
            print(f"✗ Input file not found: {input_file}")
        return False
    
    if verbose:
        print(f"Processing {subject_id} ({config.name} dataset)...")
    
    # Copy input to output dir (CAT12 writes outputs next to input)
    shutil.copy(input_file, output_dir)
    
    # Create batch file
    batch_file = create_batch_file(
        input_file=output_dir / input_file.name,
        tpm_path=config.tpm_path,
        output_dir=output_dir
    )
    
    # Run CAT12
    log_file = output_dir / "processing_log.txt"
    
    cmd = [
        str(CAT12_DIR / "run_spm25.sh"),
        str(MCR_DIR),
        "batch",
        str(batch_file)
    ]
    
    try:
        with open(log_file, 'w') as f:
            subprocess.run(
                cmd,
                cwd=CAT12_DIR,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800  # 30 min timeout per subject
            )
        
        if verbose:
            print(f"✓ Success: {subject_id}")
        return True
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if verbose:
            print(f"✗ Failed: {subject_id}")
        return False
