"""Estimate betas with GLMSingle for BIDS-formatted fMRI data preprocessed with fmriprep."""

import os

import click
import nibabel as nib
import numpy as np
import pandas as pd
from glmsingle import GLM_single
from loguru import logger

from compositionality_study.constants import (
    BETAS_DIR,
    BIDS_DIR,
    COCO_LOCAL_STIMULI_DIR,
    PREPROC_MRI_DIR,
)
from compositionality_study.utils import get_coco_df, get_stimulus_features_lookup, get_stimulus_data


def map_events_files(
    event_file: str,
    design_matrix_mapping_file: str = os.path.join(COCO_LOCAL_STIMULI_DIR, "design_matrix_mapping.csv"),  # noqa
    tr: float = 1.5,
    stim_dur: float = 3.0,
    isi: float = 3.0,
    dummy_scan_duration: float = 12.0,
) -> np.array:
    """Map events files to design matrix columns.

    :param event_file: Path to the events file.
    :type event_file: str
    :param design_matrix_mapping_file: Path to the design matrix mapping file,
        defaults to os.path.join(COCO_LOCAL_STIMULI_DIR, "design_matrix_mapping.csv")
    :type design_matrix_mapping_file: str
    :param tr: Repetition time (TR) of the fMRI experiment in seconds, defaults to 1.5.
    :type tr: float
    :param stim_dur: Duration of the stimulus in seconds, defaults to 3.0.
    :type stim_dur: float
    :param isi: Inter-stimulus interval in seconds, defaults to 3.0.
    :type isi: float
    :param dummy_scan_duration: Duration of the dummy scan at the begin and end of the sequence in seconds,
        defaults to 12.0.
    :return: Design matrix array.
    :rtype: np.array
    """
    # Read the design matrix mapping file
    design_matrix_mapping = pd.read_csv(design_matrix_mapping_file)
    # Read the events file
    events_df = pd.read_csv(event_file, sep="\t")

    # Calculate the total number of TRs in the sequence
    # Dummy scans are at the beginning and end of the sequence
    # Each trial consists of a stimulus and an inter-stimulus interval
    n_tr_per_trial = int((stim_dur + isi) // tr)
    n_dummy_trs = int(dummy_scan_duration // tr)
    total_trs = int(n_dummy_trs * 2 + n_tr_per_trial * len(events_df))

    # The design matrix is a number of TRs x number of conditions matrix
    design_matrix = np.zeros((total_trs, len(design_matrix_mapping)))

    # Iterate over all events
    for idx, event in events_df.iterrows():
        if "blank" not in event["modality"]:
            event_name = str(int(event["cocoid"])) + "_" + event["modality"]
            # Look up the event in the design matrix mapping to get the column index
            # .item() extracts the single value as a native python int
            event_idx = design_matrix_mapping[
                design_matrix_mapping["coco_id"] == event_name
            ]["design_matrix_idx"].item()
            
            # Calculate the TR index/index in time for the event
            # Explicit int() cast handles the 'Hashable' warning for runtime safety
            tr_idx = n_dummy_trs + int(idx) * n_tr_per_trial  # type: ignore

            # Set the corresponding point in time to 1
            design_matrix[tr_idx, event_idx] = 1

    return design_matrix


def map_confounds_files(
    event_file: str,
    tr: float = 1.5,
    stim_dur: float = 3.0,
    isi: float = 3.0,
    dummy_scan_duration: float = 12.0,
) -> np.array:
    """Generate extra confounds based on stimulus properties."""
    coco_df = get_coco_df()
    events_df = pd.read_csv(event_file, sep="\t")

    n_tr_per_trial = int((stim_dur + isi) // tr)
    n_dummy_trs = int(dummy_scan_duration // tr)
    total_trs = int(n_dummy_trs * 2 + n_tr_per_trial * len(events_df))

    text_cols = ["sentence_length", "amr_n_nodes"]
    img_cols = ["coco_a_nodes", "ic_score", "aspect_ratio", "coco_person"]
    all_cols = text_cols + img_cols

    confounds = pd.DataFrame(0.0, index=range(total_trs), columns=all_cols)
    txt_df, img_df = get_stimulus_features_lookup(coco_df)

    for idx, row in events_df.iterrows():
        data = get_stimulus_data(
            modality=row.get("modality"),
            stimulus=row.get("stimulus"),
            cocoid=row.get("cocoid"),
            txt_df=txt_df,
            img_df=img_df,
        )
        
        if data is None:
            continue
            
        start_tr = n_dummy_trs + idx * n_tr_per_trial
        n_stim_trs = int(round(stim_dur / tr))
        end_tr = min(start_tr + n_stim_trs, total_trs)
        
        cols = text_cols if row.get("modality") == "text" else img_cols
        valid_cols = [c for c in cols if c in data]
        if valid_cols:
            confounds.iloc[start_tr:end_tr, confounds.columns.get_indexer(valid_cols)] = data[valid_cols].values

    return confounds.values


@click.command()
@click.option(
    "--prep_input_dir",
    type=str,
    default=PREPROC_MRI_DIR,
    help="Folder with NIfTI (.nii/.nii.gz) files.",
)
@click.option(
    "--events_input_dir",
    type=str,
    default=BIDS_DIR,
    help="BIDS folder (unpreprocessed) with events.tsv files.",
)
@click.option(
    "--output_dir",
    type=str,
    default=BETAS_DIR,
    help="Folder to save beta estimates to.",
)
@click.option(
    "--design_matrix_mapping_file",
    type=str,
    default=os.path.join(COCO_LOCAL_STIMULI_DIR, "design_matrix_mapping.csv"),
    help="Design matrix mapping file.",
)
@click.option("--tr", type=float, default=1.5, help="Repetition time in seconds.")
@click.option("--stim_dur", type=float, default=3.0, help="Duration of the stimulus in seconds.")
@click.option("--isi", type=float, default=3.0, help="Inter-stimulus interval in seconds.")
@click.option(
    "--dummy_scan_duration",
    type=float,
    default=12.0,
    help="Duration of the dummy scan at the begin and end of the sequence in seconds.",
)
@click.option("--file_pattern", type=str, default="desc-prep", help="Pattern to match NIfTI files.")
@click.option("--chunklen", type=int, default=100000, help="Chunk length for the GLM_single model.")
@click.option("--n_folds", type=int, default=3, help="Number of cross-validation folds.")
@click.option(
    "--use_stimulus_confounds/--no-use_stimulus_confounds",
    default=False,
    help="Include low-level stimulus features (text/image) as extra regressors.",
)
@click.option("--subjects", type=str, multiple=True, help="List of subjects to process.", default=["sub-02"])
def estimate_betas(
    prep_input_dir: str = PREPROC_MRI_DIR,
    events_input_dir: str = BIDS_DIR,
    output_dir: str = BETAS_DIR,
    design_matrix_mapping_file: str = os.path.join(COCO_LOCAL_STIMULI_DIR, "design_matrix_mapping.csv"),  # noqa
    tr: float = 1.5,
    stim_dur: float = 3.0,
    isi: float = 3.0,
    dummy_scan_duration: float = 12.0,
    file_pattern: str = "desc-prep",
    chunklen: int = 100000,
    n_folds: int = 3,
    use_stimulus_confounds: bool = False,
    subjects: list[str] = ["sub-01"],  # noqa
) -> None:
    """Estimate beta coefficients for each trial in a BIDS-formatted fMRI dataset.

    :param prep_input_dir: Path to the folder with NIfTI files, defaults to PREPROC_MRI_DIR.
    :type prep_input_dir: str
    :param events_input_dir: Path to the BIDS folder with events.tsv files, defaults to BIDS_DIR.
    :type events_input_dir: str
    :param output_dir: Path to the folder to save beta estimates to, defaults to BETAS_DIR.
    :type output_dir: str
    :param design_matrix_mapping_file: Path to the design matrix mapping file,
        defaults to os.path.join(COCO_LOCAL_STIMULI_DIR, "design_matrix_mapping.csv").
    :type design_matrix_mapping_file: str
    :param tr: Repetition time (TR) of the fMRI experiment in seconds, defaults to 1.5.
    :type tr: float
    :param stim_dur: Duration of the stimulus in seconds, defaults to 3.0.
    :type stim_dur: float
    :param isi: Inter-stimulus interval in seconds, defaults to 3.0.
    :type isi: float
    :param dummy_scan_duration: Duration of the dummy scan at the begin and end of the sequence in seconds,
        defaults to 12.0.
    :type dummy_scan_duration: float
    :param file_pattern: Pattern to match NIfTI files (type of preprocessed file), defaults to "desc-prep".
    :type file_pattern: str
    :param chunklen: Chunk length for the GLM_single model, i.e., how many voxels are processed at the same time,
        defaults to 100000.
    :type chunklen: int
    :param n_folds: Number of cross-validation folds, defaults to 3.
    :type n_folds: int
    :param use_stimulus_confounds: Whether to include low-level stimulus features as extra regressors.
    :type use_stimulus_confounds: bool
    :param subjects: List of subjects to process, defaults to ["sub-01"].
    :type subjects: list[str]
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all subjects
    for subject in subjects:
        # See what sessions are available
        session_folders = [
            f
            for f in os.listdir(os.path.join(prep_input_dir, subject))
            if os.path.isdir(os.path.join(prep_input_dir, subject, f)) and "ses" in f
        ]
        # but ignore the ses-01-02-03 folder if it exists
        session_folders = [f for f in session_folders if f != "ses-01-02-03"]
        nifti_files = []
        events_files = []
        extra_regressors = []
        session_indicator = []

        for ses_idx, session_folder in enumerate(session_folders):
            nifti_files += [
                nib.load(os.path.join(prep_input_dir, subject, session_folder, "func", f)).get_fdata()
                for f in sorted(os.listdir(os.path.join(prep_input_dir, subject, session_folder, "func")))
                if (f.endswith(".nii") or f.endswith(".nii.gz")) and file_pattern in f
            ]
            events_files += [
                map_events_files(
                    os.path.join(events_input_dir, subject, session_folder, "func", f),
                    design_matrix_mapping_file=design_matrix_mapping_file,
                    tr=tr,
                    stim_dur=stim_dur,
                    isi=isi,
                    dummy_scan_duration=dummy_scan_duration,
                )
                for f in sorted(os.listdir(os.path.join(events_input_dir, subject, session_folder, "func")))
                if f.endswith("events.tsv")
            ]
            
            # Map confounds (extra regressors) if requested
            if use_stimulus_confounds:
                extra_regressors += [
                    map_confounds_files(
                        os.path.join(events_input_dir, subject, session_folder, "func", f),
                        tr=tr,
                        stim_dur=stim_dur,
                        isi=isi,
                        dummy_scan_duration=dummy_scan_duration,
                    )
                    for f in sorted(os.listdir(os.path.join(events_input_dir, subject, session_folder, "func")))
                    if f.endswith("events.tsv")
                ]

            session_indicator += [ses_idx + 1] * len(
                [
                    f
                    for f in sorted(os.listdir(os.path.join(events_input_dir, subject, session_folder, "func")))
                    if f.endswith("events.tsv")
                ]
            )

        logger.info(
            f"Found {len(nifti_files)} NIfTI files in {prep_input_dir} for {subject}. Starting beta estimation."
        )

        # Initialize the GLM_single model
        opt = {
            "wantmemoryoutputs": [0, 0, 0, 1],
            "chunklen": chunklen,
            "sessionindicator": np.array(session_indicator),
            "xvalscheme": np.array_split(np.arange(len(nifti_files)), n_folds),
        }

        if use_stimulus_confounds and extra_regressors:
            opt["extra_regressors"] = extra_regressors

        model = GLM_single(opt)
        model = model.fit(
            design=events_files,
            data=nifti_files,
            stimdur=stim_dur,
            tr=tr,
            outputdir=os.path.join(output_dir, subject),
            figuredir=os.path.join(output_dir, subject),
        )

        logger.info(f"Finished estimated betas for {subject}")


@click.group()
def cli() -> None:
    """Command group for fMRI beta estimations."""


if __name__ == "__main__":
    cli.add_command(estimate_betas)
    cli()
