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
from nilearn.image import load_img, math_img, smooth_img, threshold_img
from nilearn.glm.second_level import non_parametric_inference
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from compositionality_study.constants import (
    BETAS_DIR,
    BIDS_DIR,
    MVPA_DIR,
    PREPROC_MRI_DIR,
)
from compositionality_study.utils import save_brain_map

# Constants
TR = 1.5
RADIUS_MM = 6.0
SMOOTHING_FWHM = 6.0
RELIABILITY_THRESHOLD = 0.1
DEFAULT_VOXEL_P = 0.001
DEFAULT_CLUSTER_P = 0.05


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
            all_events.append(df)

    if not all_events:
        raise ValueError(f"No events found for subject {subject}")

    return pd.concat(all_events, ignore_index=True)


def compute_reliability_mask(
    betas: np.ndarray,
    events: pd.DataFrame,
    threshold: float = RELIABILITY_THRESHOLD,
) -> np.ndarray:
    """Compute split-half reliability and threshold voxels.
    
    :param betas: Beta estimates (n_voxels, n_samples)
    :type betas: np.ndarray
    :param events: Events DataFrame
    :type events: pd.DataFrame
    :param threshold: Reliability threshold
    :type threshold: float
    :return: Boolean mask of reliable voxels
    :rtype: np.ndarray
    """
    unique_stims = events['cocoid'].unique()
    
    n_voxels = betas.shape[0]
    n_stims = len(unique_stims)
    
    split1_responses = np.zeros((n_voxels, n_stims))
    split2_responses = np.zeros((n_voxels, n_stims))
    
    valid_stims_mask = np.ones(n_stims, dtype=bool)
    
    for i, stim in enumerate(unique_stims):
        indices = events.index[events['cocoid'] == stim].tolist()
        n_reps = len(indices)
        midpoint = n_reps // 2 
        idx1 = indices[:midpoint]
        idx2 = indices[midpoint:]

        split1_responses[:, i] = np.mean(betas[:, idx1], axis=1)
        split2_responses[:, i] = np.mean(betas[:, idx2], axis=1)

    split1_responses = split1_responses[:, valid_stims_mask]
    split2_responses = split2_responses[:, valid_stims_mask]
    
    s1_centered = split1_responses - split1_responses.mean(axis=1, keepdims=True)
    s2_centered = split2_responses - split2_responses.mean(axis=1, keepdims=True)
    
    ss1 = np.sum(s1_centered ** 2, axis=1)
    ss2 = np.sum(s2_centered ** 2, axis=1)
    
    numerator = np.sum(s1_centered * s2_centered, axis=1)
    denominator = np.sqrt(ss1 * ss2)
    
    correlations = np.zeros(n_voxels)
    nonzero = denominator != 0
    correlations[nonzero] = numerator[nonzero] / denominator[nonzero]
    
    return correlations > threshold


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
    res_path = os.path.join(betas_dir, f"sub-{subject}", "TYPED_FITHRF_GLMDENOISE_RR.npy")
    results = np.load(res_path, allow_pickle=True).item()
    betas_raw = results['betasmd']
    r2_map = results['R2']

    # 2. Load Events
    events_df = load_events_for_subject(bids_dir, preproc_dir, subject)

    # 3. Load Mask / Geometry
    subj_prep = os.path.join(preproc_dir, f"sub-{subject}")
    bold_files = glob.glob(os.path.join(subj_prep, "**", "*_desc-preproc_bold.nii.gz"), recursive=True)
    ref_img = nib.load(bold_files[0])  # type: ignore
    
    # 4. Voxel Selection
    type_a_path = res_path.replace("TYPED_FITHRF_GLMDENOISE_RR", "TYPEA_ONOFF")
    onoffR2 = np.load(type_a_path, allow_pickle=True).item()['R2']
    r2_threshold = findtailthreshold(onoffR2.flatten())[0]
    # Log R2 threshold
    logger.info(f"R2 threshold for {subject}: {r2_threshold:.4f}")
    valid_r2 = r2_map > r2_threshold
    # Log the percentage of voxels passing R2 threshold
    logger.info(f"Voxels passing R2 threshold for {subject}: {np.sum(valid_r2)} / {valid_r2.size}")
    # Compute the test-retest reliability mask 
    rel_mask = compute_reliability_mask(betas_raw, events_df, RELIABILITY_THRESHOLD)
    final_mask_indices = valid_r2 & rel_mask
    logger.info(f"Voxels selected for {subject}: {np.sum(final_mask_indices)} / {final_mask_indices.size}")
    
    # 5. Average Betas
    unique_stims = events_df['cocoid'].unique()
    averaged_betas = []
    averaged_labels = []
    for stim in unique_stims:
        indices = events_df.index[events_df['cocoid'] == stim].tolist()
        mean_beta = np.mean(betas_raw[:, indices], axis=1)
        row = events_df.iloc[indices[0]]
        averaged_betas.append(mean_beta)
        averaged_labels.append(row) 
    averaged_betas = np.array(averaged_betas).T # (n_voxels, n_samples)
    # Zero out non-selected voxels
    averaged_betas[~final_mask_indices, :] = 0
    # Convert labels to DataFrame
    labels_df = pd.DataFrame(averaged_labels)

    return {
        "betas": averaged_betas,
        "labels": labels_df,
        "mask_indices": final_mask_indices,
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
    mask_indices = subject_data['mask_indices']
    ref_affine = subject_data['ref_affine']
    ref_shape = subject_data['ref_shape']
    
    # Reconstruct 4D image
    # Betas are now already in the full voxel space (V, samples), just need reshaping
    n_samples = betas.shape[1]
    full_img_data = betas.reshape(ref_shape + (n_samples,))
    full_img = nib.Nifti1Image(full_img_data, ref_affine)  # type: ignore
    # Apply smoothing
    full_img = smooth_img(full_img, SMOOTHING_FWHM)
    # Mask image
    mask_vol_data = np.zeros(np.prod(ref_shape))
    mask_vol_data[mask_indices] = 1
    mask_vol = mask_vol_data.reshape(ref_shape)
    mask_img = nib.Nifti1Image(mask_vol, ref_affine)  # type: ignore

    # Output directory for subject
    subj_out = os.path.join(output_dir, f"sub-{subject}")
    os.makedirs(subj_out, exist_ok=True)
    # Valid samples mask (not blank trials)
    valid_mask = labels_df['complexity'].notna()

    def run_sl(X_idx, y, cv, name):
        logger.info(f"Running Searchlight: {name} for {subject}")
        sl = SearchLight(
            mask_img,
            radius=RADIUS_MM,
            estimator=SVC(kernel="linear"),  # type: ignore
            n_jobs=n_jobs_searchlight,
            cv=cv,
            verbose=1,
        )
        sl.fit(full_img.slicer[..., X_idx], y)  # type: ignore
        out_path = os.path.join(subj_out, f"{name}_accuracy.nii.gz")
        save_brain_map(
            sl.scores_img_, 
            out_path, 
            plot=True, 
            plot_kwargs={"title": f"Searchlight Accuracy: {name} (sub-{subject})", "display_mode": "z", "cut_coords": 5}
        )

        return sl.scores_img_

    # 1 & 2 Within Modality
    for mod in ['text', 'image']:
        idx = labels_df.index[valid_mask & (labels_df['modality'] == mod)].tolist()
        y = labels_df.loc[idx, 'complexity'].values
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        run_sl(idx, y, cv_strategy, f"within_{mod}")

    # 3 & 4 Cross Modality
    idx_text = labels_df.index[valid_mask & (labels_df['modality'] == 'text')].tolist()
    idx_img = labels_df.index[valid_mask & (labels_df['modality'] == 'image')].tolist()
    
    y_text = labels_df.loc[idx_text, 'complexity'].values
    y_img = labels_df.loc[idx_img, 'complexity'].values
    combined_idx = idx_text + idx_img
    combined_y = np.concatenate([y_text, y_img])
    n_text = len(idx_text)
    n_img = len(idx_img)
    
    # Train Text -> Test Image
    cv_txt2img = [(np.arange(n_text), np.arange(n_text, n_text + n_img))]
    run_sl(combined_idx, combined_y, cv_txt2img, "cross_text2image")
    
    # Train Image -> Test Text
    cv_img2txt = [(np.arange(n_text, n_text + n_img), np.arange(n_text))]
    run_sl(combined_idx, combined_y, cv_img2txt, "cross_image2text")


def run_group_analysis(
    subjects: tuple,
    output_dir: str,
):
    """Run group-level non-parametric analysis.
    
    :param subjects: Tuple of subject IDs
    :type subjects: tuple
    :param output_dir: Output directory
    :type output_dir: str
    """
    logger.info("Running group analysis...")
    contrasts = [
        "within_text",
        "within_image",
        "cross_text2image",
        "cross_image2text",
    ]

    output_dir_group = os.path.join(output_dir, "group_analysis")
    os.makedirs(output_dir_group, exist_ok=True)

    for contrast in contrasts:
        maps = []
        for sub in subjects:
            f = os.path.join(
                output_dir, f"sub-{sub}", f"{contrast}_accuracy.nii.gz"
            )
            if os.path.exists(f):
                maps.append(f)

        if not maps:
            continue

        # Load and subtract chance
        imgs = [load_img(m) for m in maps]
        imgs_minus_chance = [
            math_img("img - 0.5", img=img) for img in imgs
        ]

        # One-sample design
        design_matrix = pd.DataFrame(
            {"intercept": [1] * len(imgs_minus_chance)}
        )

        # Second-level analysis
        neg_log_pvals, outputs = non_parametric_inference(  # type: ignore
            second_level_input=imgs_minus_chance,
            design_matrix=design_matrix,
            second_level_contrast="intercept",
            n_perm=5000,                 
            threshold=DEFAULT_VOXEL_P,   # cluster-forming p < 0.001
            two_sided_test=False,        # one-sided: accuracy > chance
            smoothing_fwhm=None,         # critical for searchlight
            n_jobs=8,
            verbose=1,
        )

        # Save the unthresholded z-equivalent map (optional but useful)
        nib.save(   # type: ignore
            neg_log_pvals,  # type: ignore
            os.path.join(output_dir_group, f"group_{contrast}_neglogp.nii.gz"),
        )

        # Cluster-level FWE correction
        cluster_logp = outputs["logp_max_size"]  # type: ignore
        cluster_logp_thr = -np.log10(DEFAULT_CLUSTER_P)
        cluster_corrected_map = threshold_img(
            cluster_logp, threshold=cluster_logp_thr
        )

        save_brain_map(
            cluster_corrected_map,
            os.path.join(
                output_dir_group,
                f"group_{contrast}_clusterFWE.nii.gz",
            ),
            plot=True,
            plot_kwargs={"title": f"Group Cluster FWE: {contrast}"},
            make_cluster_table=True,
            cluster_table_kwargs={"stat_threshold": 0},
        )


@click.command()
@click.option("--subjects", multiple=True)
@click.option("--betas-dir", default=BETAS_DIR, type=click.Path(exists=True))
@click.option("--bids-dir", default=BIDS_DIR, type=click.Path(exists=True))
@click.option("--preproc-dir", default=PREPROC_MRI_DIR, type=click.Path(exists=True))
@click.option("--output-dir", default=MVPA_DIR, type=click.Path())
@click.option("--n_jobs_searchlight", default=32, type=int)
def main(
    subjects: tuple,
    betas_dir: str,
    bids_dir: str,
    preproc_dir: str,
    output_dir: str,
    n_jobs_searchlight: int,
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
    """
    if not subjects:
        if os.path.exists(betas_dir):
            subjects = tuple([d.split("-")[-1] for d in os.listdir(betas_dir) if d.startswith("sub-")])
        
    for sub in subjects:
        data = load_and_preprocess_subject(sub, betas_dir, bids_dir, preproc_dir, output_dir)  # type: ignore
        run_mvpa_searchlight(data, output_dir, sub, n_jobs_searchlight)

    run_group_analysis(subjects, output_dir)


if __name__ == "__main__":
    main()
