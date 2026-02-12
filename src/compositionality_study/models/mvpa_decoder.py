"""Searchlight-based MVPA decoding to predict compositional complexity from estimated betas."""

import os
import glob
from typing import Dict, Union

import numpy as np
import pandas as pd
import nibabel as nib
import click
from glmsingle.gmm.findtailthreshold import findtailthreshold
from loguru import logger
from nilearn.decoding import SearchLight
from nilearn.image import index_img, load_img, smooth_img, new_img_like
from nilearn.glm.second_level import SecondLevelModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from nilearn.masking import compute_multi_epi_mask
from nilearn import image

from compositionality_study.constants import (
    BETAS_DIR,
    BIDS_DIR,
    MVPA_DIR,
    PREPROC_MRI_DIR,
)
from compositionality_study.utils import save_brain_map

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Constants
TR = 1.5
RADIUS_MM = 6.0
SMOOTHING_FWHM = 6.0
RELIABILITY_THRESHOLD = 0.1
DEFAULT_VOXEL_P = 0.001
DEFAULT_CLUSTER_THRESHOLD = 50


def load_events_for_subject(
    bids_dir: str, preproc_dir: str, subject: str
) -> pd.DataFrame:
    """Load events in the exact order used for beta estimation.
    
    :param bids_dir: BIDS directory
    :type bids_dir: str
    :param preproc_dir: Preprocessed MRI data directory
    :type preproc_dir: str
    :param subject: Subject ID
    :type subject: str
    :return: Events DataFrame
    :rtype: pd.DataFrame
    """
    subject_preproc_dir = os.path.join(preproc_dir, f"sub-{subject}")
    subject_bids_dir = os.path.join(bids_dir, f"sub-{subject}")

    session_folders = sorted([
        f for f in os.listdir(subject_preproc_dir)
        if os.path.isdir(os.path.join(subject_preproc_dir, f)) and "ses" in f and f != "ses-01-02-03"
    ])

    all_events = []
    
    for session in session_folders:
        func_dir = os.path.join(subject_bids_dir, session, "func")
        if not os.path.exists(func_dir):
            continue
            
        events_files = sorted([
            f for f in os.listdir(func_dir)
            if f.endswith("_events.tsv")
        ])
        
        for f in events_files:
            df = pd.read_csv(os.path.join(func_dir, f), sep="\t")
            # Remove any blank trials if present based on "blank" in the "modality" column
            df = df[df['modality'] != 'blank'].reset_index(drop=True)
            all_events.append(df)

    if not all_events:
        raise ValueError(f"No events found for subject {subject}")

    return pd.concat(all_events, ignore_index=True)


def loading_mask(
    subject: str,
    preproc_dir: str,
):
    """Load brain mask for the subject.
    
    :param subject: Subject ID
    :type subject: str
    :param preproc_dir: Preprocessed MRI data directory
    :type preproc_dir: str
    """
    subj_dir = os.path.join(preproc_dir, f"sub-{subject}")
    masks = glob.glob(os.path.join(subj_dir, "**", "*_desc-brain_mask.nii.gz"), recursive=True)
    if masks:
        return load_img(masks[0])
    return None


def load_and_preprocess_subject(
    subject: str,
    betas_dir: str,
    bids_dir: str,
    preproc_dir: str,
) -> Union[Dict, None]:
    """Load, filter, average betas, and prepare for MVPA.

    :param subject: Subject ID
    :type subject: str
    :param betas_dir: Directory containing beta estimates
    :type betas_dir: str
    :param bids_dir: BIDS directory
    :type bids_dir: str
    :param preproc_dir: Preprocessed MRI data directory
    :type preproc_dir: str
    :return: Dictionary with processed data for MVPA
    :rtype: dict
    """
    logger.info(f"Processing subject {subject}...")
    
    # 1. Load GLM Results
    logger.info(f"[{subject}] Step 1/5: Loading GLM betas...")
    res_path = os.path.join(betas_dir, f"sub-{subject}", "TYPED_FITHRF_GLMDENOISE_RR.npy")
    results = np.load(res_path, allow_pickle=True).item()
    betas_raw = results['betasmd']
    r2_map = results['R2']

    # 2. Load Events
    logger.info(f"[{subject}] Step 2/5: Loading BIDS events...")
    events_df = load_events_for_subject(bids_dir, preproc_dir, subject)

    # 3. Load Mask / Geometry
    logger.info(f"[{subject}] Step 3/5: Loading geometrical reference...")
    subj_prep = os.path.join(preproc_dir, f"sub-{subject}")
    bold_files = glob.glob(os.path.join(subj_prep, "**", "*_desc-preproc_bold.nii.gz"), recursive=True)
    ref_img = nib.load(bold_files[0])  # type: ignore
    
    # 4. Voxel Selection
    logger.info(f"[{subject}] Step 4/5: Computing voxel masks (R2 threshold)...")
    r2_threshold = findtailthreshold(r2_map.flatten())[0]
    # Log R2 threshold
    logger.info(f"[{subject}] R2 threshold: {r2_threshold:.4f}")
    valid_r2 = r2_map > r2_threshold
    # Save thresholded R2 map for visualization
    valid_r2_img = new_img_like(ref_img, valid_r2)  # type: ignore
    save_brain_map(
        valid_r2_img,
        os.path.join(betas_dir, f"sub-{subject}", "valid_R2_map.nii.gz"),
        is_z_map=False,
        make_surface_plot=True,
    )
    # Log the percentage of voxels passing R2 threshold
    logger.info(f"[{subject}] R2 pass: {np.sum(valid_r2)} / {valid_r2.size} voxels")

    # 5. Average Betas
    logger.info(f"[{subject}] Step 5/5: Averaging betas by stimulus ID...")
    unique_stims = events_df['cocoid'].unique()
    averaged_betas = []
    averaged_labels = []

    for stim in unique_stims:
        indices = events_df.index[events_df['cocoid'] == stim].tolist()
        mean_beta = np.mean(betas_raw[..., indices], axis=-1)
        row = events_df.iloc[indices[0]]
        averaged_betas.append(mean_beta)
        averaged_labels.append(row) 
    
    # Stack to (X, Y, Z, n_stims)
    averaged_betas = np.stack(averaged_betas, axis=-1)
    # Zero out non-selected voxels 
    averaged_betas[~valid_r2] = 0.0
    
    # Convert labels to DataFrame
    labels_df = pd.DataFrame(averaged_labels).reset_index(drop=True)

    return {
        "betas": averaged_betas,
        "labels": labels_df,
        "mask_data": valid_r2,
        "ref_affine": ref_img.affine,  # type: ignore
        "ref_shape": ref_img.shape[:3],  # type: ignore
    }


def run_mvpa_searchlight(
    subject_data: dict,
    output_dir: str,
    subject: str,
    n_jobs_searchlight: int,
):
    """Run MVPA searchlight analysis for a single subject.
    
    :param subject_data: Processed data for the subject
    :type subject_data: dict
    :param output_dir: Output directory
    :type output_dir: str
    :param subject: Subject ID
    :type subject: str
    :param n_jobs_searchlight: Number of parallel jobs for searchlight
    :type n_jobs_searchlight: int
    """
    betas = subject_data['betas']
    labels_df = subject_data['labels']
    mask_data = subject_data['mask_data']
    ref_affine = subject_data['ref_affine']
    
    # Create 4D image directly
    full_img = nib.Nifti1Image(betas, ref_affine)  # type: ignore
    # Apply smoothing
    full_img = smooth_img(full_img, SMOOTHING_FWHM)
    
    # Mask image
    mask_img = nib.Nifti1Image(mask_data.astype(np.int8), ref_affine)  # type: ignore

    # Output directory for subject
    subj_out = os.path.join(output_dir, f"sub-{subject}")
    os.makedirs(subj_out, exist_ok=True)
    # Valid samples mask (not blank trials)
    valid_mask = labels_df['complexity'].notna()

    def run_sl(X_idx, y, cv, name, n_test_samples):
        logger.info(f"[{subject}] Running Searchlight Analysis: {name}")
        sl = SearchLight(
            mask_img,
            radius=RADIUS_MM,
            estimator=LinearSVC(),  # type: ignore
            n_jobs=n_jobs_searchlight,
            cv=cv,
            verbose=1,
        )
        subset_img = index_img(full_img, X_idx)
        sl.fit(subset_img, y)

        out_path = os.path.join(subj_out, f"{name}_accuracy.nii.gz")
        save_brain_map(sl.scores_img_, out_path)

        # Compute Variance Map for Fixed Effects Group Analysis
        # Var = p * (1-p) / N_test_samples
        # Where p is accuracy.
        acc_data = sl.scores_img_.get_fdata()  # type: ignore
        var_data = (acc_data * (1.0 - acc_data)) / float(n_test_samples)
        var_data = np.maximum(var_data, 1e-6)
        var_img = new_img_like(sl.scores_img_, var_data)
        var_path = os.path.join(subj_out, f"{name}_var.nii.gz")
        save_brain_map(var_img, var_path, make_surface_plot=False)

        return sl.scores_img_

    # 1 & 2 Within Modality
    logger.info(f"[{subject}] Starting Within-Modality Decoding analyses...")
    for mod in ['text', 'image']:
        idx = labels_df.index[valid_mask & (labels_df['modality'] == mod)].to_numpy(dtype=int)
        y = labels_df.loc[idx, 'complexity'].to_numpy()
        n_samples = len(idx)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        run_sl(idx, y, cv_strategy, f"within_{mod}", n_test_samples=n_samples)

    # 3 & 4 Cross Modality
    logger.info(f"[{subject}] Starting Cross-Modality Decoding analyses...")
    idx_text = labels_df.index[valid_mask & (labels_df['modality'] == 'text')].tolist()
    idx_img = labels_df.index[valid_mask & (labels_df['modality'] == 'image')].tolist()
    
    y_text = labels_df.loc[idx_text, 'complexity'].to_numpy()
    y_img = labels_df.loc[idx_img, 'complexity'].to_numpy()
    combined_idx = idx_text + idx_img
    combined_y = np.concatenate([y_text, y_img])
    n_text = len(idx_text)
    n_img = len(idx_img)
    
    # Train Text -> Test Image (Total test samples = n_img)
    cv_txt2img = [(np.arange(n_text, dtype=int), np.arange(n_text, n_text + n_img, dtype=int))]
    run_sl(combined_idx, combined_y, cv_txt2img, "cross_text2image", n_test_samples=n_img)
    
    # Train Image -> Test Text (Total test samples = n_text)
    cv_img2txt = [(np.arange(n_text, n_text + n_img, dtype=int), np.arange(n_text, dtype=int))]
    run_sl(combined_idx, combined_y, cv_img2txt, "cross_image2text", n_test_samples=n_text)


def run_group_analysis(
    subjects: tuple,
    output_dir: str,
    betas_dir: str,
):
    """Run group-level random-effects (one-sample) analysis.

    Each subject contributes a chance-centered effect map (accuracy - 0.5) and
    an estimated variance map. ``SecondLevelModel`` fits an intercept-only model
    across subjects (subject-wise RFX; variance maps give precision weighting).

    :param subjects: Tuple of subject IDs
    :type subjects: tuple
    :param output_dir: Output directory
    :type output_dir: str
    :param betas_dir: Directory containing beta estimates and R2 maps
    :type betas_dir: str
    """
    logger.info("Running group analysis ...")
    contrasts = [
        "within_text",
        "within_image",
        "cross_text2image",
        "cross_image2text",
    ]

    output_dir_group = os.path.join(output_dir, "group_analysis")
    os.makedirs(output_dir_group, exist_ok=True)
    
    # Build subject masks from saved R2 maps (raw or pre-thresholded)
    mask_imgs = []
    for sub in subjects:
        r2_path = os.path.join(betas_dir, f"sub-{sub}", "R2_map.nii.gz")
        valid_r2_path = os.path.join(betas_dir, f"sub-{sub}", "valid_R2_map.nii.gz")

        if os.path.exists(valid_r2_path):
            r2_img = load_img(valid_r2_path)
            r2_data = r2_img.get_fdata()
            subj_mask_data = r2_data > 0
        elif os.path.exists(r2_path):
            r2_img = load_img(r2_path)
            r2_data = r2_img.get_fdata()
            r2_threshold = findtailthreshold(r2_data.flatten())[0]
            subj_mask_data = r2_data > r2_threshold
        else:
            raise FileNotFoundError(f"Missing R2 map for subject {sub}: {r2_path} or {valid_r2_path}")

        subj_mask_img = new_img_like(r2_img, subj_mask_data.astype(np.int8))
        mask_imgs.append(subj_mask_img)

        # Save per-subject mask for transparency/debugging
        subj_mask_out = os.path.join(output_dir, f"sub-{sub}", "mask.nii.gz")
        os.makedirs(os.path.dirname(subj_mask_out), exist_ok=True)
        save_brain_map(
            subj_mask_img,
            subj_mask_out,
            is_z_map=False,
            make_surface_plot=False,
        )
    group_mask = compute_multi_epi_mask(mask_imgs, threshold=0.25)
    # Save group mask for reference
    save_brain_map(
        group_mask,
        os.path.join(output_dir_group, "group_mask.nii.gz"),
        is_z_map=False,
        make_surface_plot=True,
    )

    for contrast in tqdm(contrasts, desc="Group Analysis"):
        rows = []
        for sub in subjects:
            subj_dir = os.path.join(output_dir, f"sub-{sub}")
            acc_path = os.path.join(subj_dir, f"{contrast}_accuracy.nii.gz")
            effect_path = os.path.join(subj_dir, f"{contrast}_effect.nii.gz")

            if not os.path.exists(effect_path):
                acc_img = load_img(acc_path)
                centered = image.math_img("img - 0.5", img=acc_img)
                centered.to_filename(effect_path)  # type: ignore

            rows.append(
                {
                    "subject_label": sub,
                    "map_name": contrast,
                    "effects_map_path": effect_path,
                }
            )

        second_level_df = pd.DataFrame(rows).sort_values("subject_label").reset_index(drop=True)
        design_matrix = pd.DataFrame({"intercept": np.ones(len(second_level_df))})

        # Apply RFX analysis here
        group_model = SecondLevelModel(
            smoothing_fwhm=None,
            mask_img=group_mask,
            n_jobs=1,
            verbose=1,
        ).fit(second_level_df, design_matrix=design_matrix)

        z_score_img = group_model.compute_contrast(
            second_level_contrast="intercept",
            first_level_contrast=contrast,
            output_type="z_score",
        )

        thresholds = [
            (0.05, 0, "p05_cluster0"),
            (0.001, 0, "p001_cluster0"),
            (0.001, 50, "p001_cluster50"),
        ]

        for voxel_p, cluster_thr, tag in thresholds:
            save_brain_map(
                z_score_img,
                os.path.join(output_dir_group, f"group_{contrast}_zmap_{tag}.nii.gz"),
                is_z_map=True,
                voxel_p=voxel_p,
                cluster_threshold=cluster_thr,
                make_surface_plot=False,
                sided="positive",
            )


@click.command()
@click.option("--subjects", multiple=True)
@click.option("--betas-dir", default=BETAS_DIR, type=click.Path(exists=True))
@click.option("--bids-dir", default=BIDS_DIR, type=click.Path(exists=True))
@click.option("--preproc-dir", default=PREPROC_MRI_DIR, type=click.Path(exists=True))
@click.option("--output-dir", default=MVPA_DIR, type=click.Path())
@click.option("--n-jobs-searchlight", default=32, type=int)
@click.option(
    "--override-results/--no-override-results",
    default=False,
    show_default=True,
    help="Recalculate results even if outputs already exist.",
)
def main(
    subjects: tuple,
    betas_dir: str,
    bids_dir: str,
    preproc_dir: str,
    output_dir: str,
    n_jobs_searchlight: int,
    override_results: bool,
):
    """Run MVPA Analysis.
    
    :param subjects: Tuple of subject IDs
    :type subjects: tuple
    :param betas_dir: Directory containing beta estimates
    :type betas_dir: str
    :param bids_dir: BIDS directory
    :type bids_dir: str
    :param preproc_dir: Preprocessed MRI data directory
    :type preproc_dir: str
    :param output_dir: Output directory
    :type output_dir: str
    :param n_jobs_searchlight: Number of parallel jobs for searchlight
    :type n_jobs_searchlight: int
    :param override_results: Whether to recalculate results even if outputs exist
    :type override_results: bool
    """
    if not subjects:
        if os.path.exists(betas_dir):
            subjects = tuple([d.split("-")[-1] for d in os.listdir(betas_dir) if d.startswith("sub-")])
        
    for sub in subjects:
        subj_out = os.path.join(output_dir, f"sub-{sub}")
        contrasts = ["within_text", "within_image", "cross_text2image", "cross_image2text"]
        expected = [os.path.join(subj_out, f"{c}_accuracy.nii.gz") for c in contrasts]
        
        if not override_results and os.path.isdir(subj_out) and all(os.path.exists(p) for p in expected):
            logger.info(f"Skipping subject {sub} - results exist.")
            continue

        data = load_and_preprocess_subject(sub, betas_dir, bids_dir, preproc_dir)  # type: ignore
        run_mvpa_searchlight(data, output_dir, sub, int(n_jobs_searchlight))  # type: ignore

    run_group_analysis(subjects, output_dir, betas_dir)


if __name__ == "__main__":
    main()
