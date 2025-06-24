"""Searchlight-based MVPA decoding to predict compositional complexity from estimated betas."""

import os

import click
import nibabel as nib
import pandas as pd
from nilearn.decoding import SearchLight
from nilearn.image import smooth_img

# from nilearn.masking import apply_mask
from sklearn.svm import SVC

from compositionality_study.constants import BETAS_DIR, MVPA_DIR, PREPROC_MRI_DIR


@click.command()
@click.option("--events_dir", required=True, type=str, help="Path to the events CSV containing trial info.")
@click.option("--betas_dir", default=BETAS_DIR, type=str, help="Path to the estimated betas.")
@click.option("--on_off_mask_dir", default=PREPROC_MRI_DIR, type=str, help="Path to the 'on-off' brain mask.")
@click.option(
    "--output_dir",
    default=MVPA_DIR,
    type=str,
    help="Where to save the searchlight decoding results.",
)
@click.option(
    "--fwhm",
    type=float,
    default=4.0,
    help="Full-width half-maximum for smoothing the 'on-off' mask.",
)
def run_searchlight_decoding(
    events_dir: str,
    betas_dir: str = BETAS_DIR,
    on_off_mask_dir: str = PREPROC_MRI_DIR,
    output_dir: str = MVPA_DIR,
    fwhm: float = 4.0,
) -> None:
    """Perform searchlight-based decoding to predict compositional complexity.

    :param betas_dir: Path to the estimated betas.
    :type betas_dir: str
    :param events_dir: Path to the events CSV containing trial info (e.g., compositional complexity labels).
    :type events_dir: str
    :param on_off_mask_dir: Path to the 'on-off' brain mask.
    :type on_off_mask_dir: str
    :param output_dir: Where to save the searchlight decoding results.
    :type output_dir: str
    :param fwhm: Full-width half-maximum for smoothing the 'on-off' mask, defaults to 4.0.
    :type fwhm: float
    """
    # Create output dir if needed
    os.makedirs(output_dir, exist_ok=True)

    # TODO Load betas and events for each run
    betas_img = nib.load(betas_dir)
    events_df = pd.read_csv(events_dir)
    labels = events_df["complexity_label"].values  # e.g., numeric or binary labels
    # TODO filter out blanks in the events

    # Load and smooth 'on-off' mask
    on_off_mask = nib.load(on_off_mask_dir)
    on_off_mask_smoothed = smooth_img(on_off_mask, fwhm=fwhm)

    # Prepare data for searchlight
    # betas_data = apply_mask(betas_img, on_off_mask_smoothed)

    # Set up searchlight
    searchlight = SearchLight(
        on_off_mask_smoothed,
        radius=5,  # example radius in voxels
        scoring="accuracy",
        estimator=SVC(kernel="linear"),
        n_jobs=1,
    )

    # Fit the model
    searchlight.fit(betas_img, labels)

    # Save results as a NIfTI
    results_img = searchlight.scores_
    nib.save(results_img, os.path.join(output_dir, "searchlight_decoding.nii.gz"))
