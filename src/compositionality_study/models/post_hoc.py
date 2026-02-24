"""Post-hoc univariate analysis focusing on the image_high - image_low contrast inside a pre-defined mask."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import click
import numpy as np
import pandas as pd
from loguru import logger
from nilearn import image, masking
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel

from compositionality_study.constants import BIDS_DIR, PREPROC_MRI_DIR, MRI_DIR
from compositionality_study.utils import (
    collect_runs,
    find_gm_mask,
    load_events,
    load_stimulus_confounds,
    save_brain_map,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

TR = 1.5
SMOOTHING_FWHM = 6.0
GM_THRESHOLD = 0.3
DEFAULT_VOXEL_P = 0.001
DEFAULT_CLUSTER_THRESHOLD = 20

CONTRAST_NAME = "img_high_vs_low"
CONTRAST_EXPR = "image_high - image_low"
DEFAULT_OUTPUT_DIR = Path(MRI_DIR) / "post-hoc"
DEFAULT_MASK_PATH = (
    Path(MRI_DIR)
    / "univariate-standard"
    / "group_level"
    / "group_main_modality_z_score_p001_cluster50_pos_thr.nii.gz"
)

MOTION_COLS = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
]


def _fit_subject(
    bold_imgs: Sequence[Path],
    events_files: Sequence[Path],
    confounds_files: Sequence[Path],
    gm_probseg: Path | object,
    roi_mask_img: object | None,
    output_dir: Path,
    tr: float,
    smoothing_fwhm: float,
    gm_threshold: float,
    voxel_p: float,
    cluster_threshold: int,
    use_stimulus_confounds: bool,
) -> Dict[str, Path]:
    """Run a single-subject GLM for the target contrast and write maps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    events = [load_events(f) for f in events_files]
    motion_confounds = [pd.read_csv(f, sep="\t")[MOTION_COLS].fillna(0.0) for f in confounds_files]

    confounds: List[pd.DataFrame] = []
    for ev, mot in zip(events, motion_confounds):
        if use_stimulus_confounds:
            stim_conf = load_stimulus_confounds(ev, len(mot), tr)
            confounds.append(pd.concat([mot, stim_conf], axis=1))
        else:
            confounds.append(mot)

    gm_mask_img = image.math_img(f"img > {gm_threshold}", img=gm_probseg)
    gm_mask_img = image.resample_to_img(gm_mask_img, image.load_img(bold_imgs[0]), interpolation="nearest")

    combined_mask = gm_mask_img
    if roi_mask_img is not None:
        resampled_roi = image.resample_to_img(roi_mask_img, gm_mask_img, interpolation="nearest")
        roi_bin = image.math_img("img > 0", img=resampled_roi)
        combined_mask = image.math_img("gm * roi", gm=gm_mask_img, roi=roi_bin)

    glm = FirstLevelModel(
        t_r=tr,
        hrf_model="glover",
        drift_model="cosine",
        smoothing_fwhm=smoothing_fwhm,
        noise_model="ar1",
        standardize="psc",  # type: ignore
        signal_scaling=False,
        minimize_memory=False,
        mask_img=combined_mask,
        n_jobs=1,
        verbose=1,
    )
    glm = glm.fit(run_imgs=list(bold_imgs), events=events, confounds=confounds)

    eff_maps = glm.compute_contrast(CONTRAST_EXPR, output_type="all")  # type: ignore
    saved: Dict[str, Path] = {}
    for result_name, eff_map in eff_maps.items():  # type: ignore
        out_path = output_dir / f"{CONTRAST_NAME}_{result_name}.nii.gz"
        save_brain_map(
            eff_map,
            out_path,
            is_z_map=result_name == "z_score",
            voxel_p=voxel_p,
            cluster_threshold=cluster_threshold,
            make_surface_plot=False,
            sided="positive",
        )
        saved[result_name] = out_path
    return saved


def _summarize_roi(map_path: Path, mask_img: object) -> float:
    """Average map value inside the provided mask."""
    data = masking.apply_mask(image.load_img(map_path), mask_img)
    return float(np.nanmean(data))


@click.command()
@click.option(
    "--fmriprep-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=PREPROC_MRI_DIR,
    show_default=True,
    help="Directory containing fMRIPrep derivatives.",
)
@click.option(
    "--bids-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=BIDS_DIR,
    show_default=True,
    help="BIDS directory with events.tsv files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Folder for subject-level and group-level outputs.",
)
@click.option(
    "--mask-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_MASK_PATH,
    show_default=True,
    help="Mask used for ROI averaging and group model fitting.",
)
@click.option(
    "--subjects",
    type=str,
    multiple=True,
    default=("02", "03", "04", "05", "06", "07", "08", "09"),
    show_default=True,
    help="Subject IDs without the 'sub-' prefix; auto-discovers if omitted.",
)
@click.option("--tr", type=float, default=TR, show_default=True, help="Repetition time in seconds.")
@click.option(
    "--smoothing-fwhm",
    type=float,
    default=SMOOTHING_FWHM,
    show_default=True,
    help="Spatial smoothing FWHM in millimeters.",
)
@click.option(
    "--gm-threshold",
    type=float,
    default=GM_THRESHOLD,
    show_default=True,
    help="Probability threshold for the GM mask.",
)
@click.option(
    "--voxel-p",
    type=float,
    default=DEFAULT_VOXEL_P,
    show_default=True,
    help="Voxel-wise p-value threshold applied to z-maps (one-sided positive).",
)
@click.option(
    "--cluster-threshold",
    type=int,
    default=DEFAULT_CLUSTER_THRESHOLD,
    show_default=True,
    help="Cluster size threshold in voxels.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=1,
    show_default=True,
    help="Number of CPUs for the group model (subject fits run single-threaded).",
)
@click.option(
    "--use-stimulus-confounds/--no-use-stimulus-confounds",
    default=False,
    show_default=True,
    help="Include low-level stimulus features as confounds.",
)
@click.option(
    "--reuse-first-level/--no-reuse-first-level",
    default=True,
    show_default=True,
    help="Reuse cached first-level maps if present instead of refitting.",
)
@click.option(
    "--univariate-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    show_default=True,
    help="Existing subject-level directory (e.g. univariate-standard/subject_level). "
        "When provided, first-level fitting is skipped and maps are read from there.",
)
def run_post_hoc(
    fmriprep_dir: Path,
    bids_dir: Path,
    output_dir: Path,
    mask_path: Path,
    subjects: Tuple[str, ...],
    tr: float,
    smoothing_fwhm: float,
    gm_threshold: float,
    voxel_p: float,
    cluster_threshold: int,
    n_jobs: int,
    use_stimulus_confounds: bool,
    reuse_first_level: bool,
    univariate_dir: Path | None,
) -> None:
    """Run a minimal GLM and summarize the target contrast inside the supplied mask."""
    subject_ids = (
        list(subjects)
        if subjects
        else [p.name.replace("sub-", "") for p in sorted(fmriprep_dir.glob("sub-*")) if p.is_dir()]
    )
    mask_img = image.math_img("img > 0", img=image.load_img(mask_path))
    subject_level_dir = output_dir / "subject_level"
    group_level_dir = output_dir / "group_level"

    second_level_rows: List[dict] = []
    roi_rows: List[dict] = []

    for subject in subject_ids:
        logger.info(f"Subject {subject}: fitting first-level model")
        runs = collect_runs(fmriprep_dir=fmriprep_dir, bids_dir=bids_dir, subject=subject, max_runs=None)
        gm_mask = find_gm_mask(fmriprep_dir, subject)
        bold_imgs, events_files, confounds_files = zip(*runs)
        subj_mask = image.resample_to_img(mask_img, image.load_img(bold_imgs[0]), interpolation="nearest")
        subject_output = subject_level_dir / f"sub-{subject}"

        def _map_paths(base_dir: Path) -> Dict[str, Path]:
            return {
                "effect_size": base_dir / f"{CONTRAST_NAME}_effect_size.nii.gz",
                "effect_variance": base_dir / f"{CONTRAST_NAME}_effect_variance.nii.gz",
                "z_score": base_dir / f"{CONTRAST_NAME}_z_score.nii.gz",
            }

        if univariate_dir is not None:
            saved_paths = _map_paths(univariate_dir / f"sub-{subject}")
            if not all(p.exists() for p in saved_paths.values()):
                raise FileNotFoundError(
                    f"Expected first-level maps not found for sub-{subject} in {univariate_dir}"
                )
            logger.info(f"Subject {subject}: reusing maps from {univariate_dir}")
        else:
            paths = _map_paths(subject_output)
            if reuse_first_level and all(p.exists() for p in paths.values()):
                saved_paths = paths
            else:
                saved_paths = _fit_subject(
                    bold_imgs=bold_imgs,
                    events_files=events_files,
                    confounds_files=confounds_files,
                    gm_probseg=gm_mask,
                    roi_mask_img=None,  # GM mask only at subject level; ROI mask applied at group level
                    output_dir=subject_output,
                    tr=tr,
                    smoothing_fwhm=smoothing_fwhm,
                    gm_threshold=gm_threshold,
                    voxel_p=voxel_p,
                    cluster_threshold=cluster_threshold,
                    use_stimulus_confounds=use_stimulus_confounds,
                )

        second_level_rows.append(
            {
                "subject_label": subject,
                "map_name": CONTRAST_NAME,
                "effects_map_path": str(saved_paths["effect_size"]),
                "variance_map_path": str(saved_paths["effect_variance"]),
            }
        )

        roi_rows.append(
            {
                "subject": subject,
                "mean_beta": _summarize_roi(saved_paths["effect_size"], subj_mask),
                "mean_z": _summarize_roi(saved_paths["z_score"], subj_mask),
            }
        )

    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(output_dir / "roi_summary.csv", index=False)

    logger.info("Fitting group model inside mask")
    design_matrix = pd.DataFrame({"intercept": np.ones(len(second_level_rows))})
    group_model = SecondLevelModel(mask_img=mask_img, smoothing_fwhm=None, n_jobs=n_jobs, verbose=1)
    group_model.fit(pd.DataFrame(second_level_rows), design_matrix=design_matrix)

    group_maps = group_model.compute_contrast(
        second_level_contrast="intercept",
        first_level_contrast=CONTRAST_NAME,
        output_type="all",
    )
    group_z = group_maps.get("z_score")  # type: ignore
    if group_z is not None:
        save_brain_map(
            group_z,
            group_level_dir / f"group_{CONTRAST_NAME}_z_score.nii.gz",
            is_z_map=True,
            voxel_p=voxel_p,
            cluster_threshold=cluster_threshold,
            make_surface_plot=False,
            sided="positive",
        )

        alt_thresholds = [
            (0.05, 0, "p05_cluster0"),
            (0.001, 0, "p001_cluster0"),
            (0.001, 50, "p001_cluster50"),
        ]
        for voxel_p_alt, cluster_thr_alt, tag in alt_thresholds:
            save_brain_map(
                group_z,
                group_level_dir / f"group_{CONTRAST_NAME}_z_score_{tag}.nii.gz",
                is_z_map=True,
                voxel_p=voxel_p_alt,
                cluster_threshold=cluster_thr_alt,
                make_surface_plot=False,
                sided="positive",
            )

        group_roi_mean = _summarize_roi(group_z, mask_img)
        (output_dir / "roi_group_mean.txt").write_text(f"mean_z_in_mask={group_roi_mean:.4f}\n")


if __name__ == "__main__":
    run_post_hoc()
