"""Estimate betas with GLMSingle for BIDS-formatted fMRI data preprocessed with fmriprep."""

import os

import click
import nibabel as nib
import pandas as pd
from glmsingle import GLM_single
from loguru import logger


@click.command()
@click.option("--prep_input_dir", type=str, required=True, help="Folder with NIfTI (.nii/.nii.gz) files.")
@click.option("--events_input_dir", type=str, required=True, help="BIDS folder (unpreprocessed) with events.tsv files.")
@click.option("--output_dir", type=str, required=True, help="Folder to save beta estimates to.")
@click.option("--tr", type=float, default=1.5, help="Repetition time in seconds.")
@click.option("--stim_dur", type=float, default=3.0, help="Duration of the stimulus in seconds.")
@click.option("--isi", type=float, default=3.0, help="Inter-stimulus interval in seconds.")
@click.option("--file_pattern", type=str, default="desc-prep", help="Pattern to match NIfTI files.")
@click.option("--subjects", type=str, multiple=True, help="List of subjects to process.", default=["sub-01"])
def estimate_betas(
    prep_input_dir: str,
    events_input_dir: str,
    output_dir: str,
    tr: float = 1.5,
    stim_dur: float = 3.0,
    isi: float = 3.0,
    file_pattern: str = "desc-prep",
    subjects: list[str] = ["sub-01"],
) -> None:
    """Estimate beta coefficients for each trial in a BIDS-formatted fMRI dataset.

    :param prep_input_dir: Path to the folder with NIfTI files.
    :type prep_input_dir: str
    :param events_input_dir: Path to the BIDS folder with events.tsv files.
    :type events_input_dir: str
    :param output_dir: Path to the folder to save beta estimates to.
    :type output_dir: str
    :param tr: Repetition time (TR) of the fMRI experiment in seconds, defaults to 1.5.
    :type tr: float
    :param stim_dur: Duration of the stimulus in seconds, defaults to 3.0.
    :type stim_dur: float
    :param isi: Inter-stimulus interval in seconds, defaults to 3.0.
    :type isi: float
    :param file_pattern: Pattern to match NIfTI files (type of preprocessed file), defaults to "desc-prep".
    :type file_pattern: str
    :param subjects: List of subjects to process, defaults to ["sub-01"].
    :type subjects: list[str]
    """
    # Create the output directory
    output_dir = os.path.join(output_dir, "beta_estimates")
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all subjects
    for subject in subjects:
        # See what sessions are available
        session_folders = [
            f for f in os.listdir(prep_input_dir) if os.path.isdir(os.path.join(prep_input_dir, f)) and "ses" in f
        ]
        nifti_files = []
        events_files = []
        session_indicator = []

        # Get all NIfTI files and events.tsvs for all sessions
        for ses_idx, session_folder in enumerate(session_folders):
            nifti_files += [
                nib.load(f)
                for f in os.listdir(os.path.join(prep_input_dir, session_folder, "func"))
                if (f.endswith(".nii") or f.endswith(".nii.gz")) and file_pattern in f
            ]
            events_files += [
                f
                for f in os.listdir(os.path.join(events_input_dir, session_folder, "func"))
                if f.endswith("events.tsv") and file_pattern in f
            ]
            session_indicator += [ses_idx + 1] * len(nifti_files)

        logger.info(f"Found {len(nifti_files)} NIfTI files in {prep_input_dir}. Starting beta estimation.")

        # TODO change from here
        for nifti_file in nifti_files:
            base_name = nifti_file.split(".nii")[0]
            events_tsv_path = os.path.join(prep_input_dir, base_name + "_events.tsv")
            if not os.path.exists(events_tsv_path):
                logger.warning(f"No matching events file for {nifti_file}, skipping.")
                continue

        events_df = pd.read_csv(events_tsv_path, sep="\t")

        model = GLM_single()
        model = model.fit(nifti_files, events_df, tr)
        design_matrix = model.design_matrices_[0]

        betas = {}
        for column in design_matrix.columns:
            if column.startswith("trial_type"):
                beta_img = model.compute_contrast(column, output_type="effect_size")
                beta_out_path = os.path.join(output_dir, f"{base_name}_beta_{column}.nii.gz")
                beta_img.to_filename(beta_out_path)
                betas[column] = beta_out_path

        logger.info(f"Finished estimated betas for {subject}")


@click.group()
def cli() -> None:
    """Command group for fMRI beta estimations."""


if __name__ == "__main__":
    cli.add_command(estimate_betas)
    cli()
