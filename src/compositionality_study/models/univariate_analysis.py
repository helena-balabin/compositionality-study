"""Run a univariate GLM on fMRIPrep outputs and aggregate group-level maps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import click
import numpy as np
import pandas as pd
from loguru import logger
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from tqdm import tqdm

from compositionality_study.constants import BIDS_DIR, MODELS_DIR, PREPROC_MRI_DIR
from compositionality_study.utils import (
    collect_runs,
    conjunction_map,
    find_gm_mask,
    load_events,
    load_stimulus_confounds,
    save_brain_map,
    split_runs_by_session,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

TR = 1.5
SMOOTHING_FWHM = 6.0
GM_THRESHOLD = 0.3

MOTION_COLS = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
]

CONTRASTS = {
    "img_high_vs_low": "image_high - image_low",
    "txt_high_vs_low": "text_high - text_low",
    "main_compositionality": "(image_high + text_high)/2 - (image_low + text_low)/2",
    "main_modality": "(image_high + image_low)/2 - (text_high + text_low)/2",
    "interaction": "(image_high - image_low) - (text_high - text_low)",
}

DEFAULT_OUTPUT_DIR = Path(MODELS_DIR) / "univariate_glm"
DEFAULT_VOXEL_P = 0.001
DEFAULT_CLUSTER_THRESHOLD = 20


def run_subject_glm(
    bold_imgs: Sequence[Path],
    events_files: Sequence[Path],
    confounds_files: Sequence[Path],
    gm_probseg: Path | object,
    output_dir: Path,
    tr: float = TR,
    smoothing_fwhm: float = SMOOTHING_FWHM,
    gm_threshold: float = GM_THRESHOLD,
    cluster_threshold: int = DEFAULT_CLUSTER_THRESHOLD,
    voxel_p: float = DEFAULT_VOXEL_P,
    n_jobs: int = 1,
    use_stimulus_confounds: bool = False,
) -> FirstLevelModel:
    """Run a first-level GLM and write subject-level contrast maps."""
    # Load data and prepare confounds
    output_dir.mkdir(parents=True, exist_ok=True)
    events = [load_events(f) for f in events_files]
    motion_confounds = [pd.read_csv(f, sep="\t")[MOTION_COLS].fillna(0.0) for f in confounds_files]

    # Combine motion regressors with stimulus confounds, if specified
    confounds = []
    for ev, mot in zip(events, motion_confounds):
        if use_stimulus_confounds:
            n_scans = len(mot)
            stim_conf = load_stimulus_confounds(ev, n_scans, tr)
            confounds.append(pd.concat([mot, stim_conf], axis=1))
        else:
            confounds.append(mot)

    # Create grey matter mask
    logger.info("Processing GM mask...")
    gm_mask_img = image.math_img(f"img > {gm_threshold}", img=gm_probseg)
    ref_img = image.load_img(bold_imgs[0])
    gm_mask_img = image.resample_to_img(
        source_img=gm_mask_img,
        target_img=ref_img,
        interpolation="nearest"
    )

    # Save the mask (using the utils function for consistency)
    mask_out = output_dir / "gm_mask.nii.gz"
    save_brain_map(gm_mask_img, mask_out, make_surface_plot=False)

    # Run the FirstLevelModel
    logger.info(f"Fitting FirstLevelModel (n_jobs={n_jobs})... this may take a few minutes.")
    glm = FirstLevelModel(
        t_r=tr,
        hrf_model="glover",
        drift_model="cosine",
        smoothing_fwhm=smoothing_fwhm,
        noise_model="ar1",
        standardize="psc",  # type: ignore
        signal_scaling=False,
        minimize_memory=False,
        mask_img=gm_mask_img,
        n_jobs=n_jobs,
        verbose=1,
    )
    try:
        glm = glm.fit(run_imgs=list(bold_imgs), events=events, confounds=confounds)
    except ValueError as exc:
        if "mask is invalid" in str(exc).lower():
            logger.warning("Mask invalid/empty; retrying FirstLevelModel with mask_img=None to continue.")
            glm = FirstLevelModel(
                t_r=tr,
                hrf_model="glover",
                drift_model="cosine",
                smoothing_fwhm=smoothing_fwhm,
                noise_model="ar1",
                standardize="psc",  # type: ignore
                signal_scaling=False,
                minimize_memory=False,
                mask_img=None,
                n_jobs=n_jobs,
                verbose=1,
            )
            glm = glm.fit(run_imgs=list(bold_imgs), events=events, confounds=confounds)
        else:
            raise

    logger.info(f"Computing {len(CONTRASTS)} contrasts...")
    for name, expression in tqdm(CONTRASTS.items(), desc="Contrasts"):
        eff_maps: dict = glm.compute_contrast(expression, output_type="all")  # type: ignore
        for result_name, eff_map in eff_maps.items():
            eff_file = output_dir / f"{name}_{result_name}.nii.gz"
            save_brain_map(
                eff_map,
                eff_file,
                is_z_map=result_name == "z_score",
                voxel_p=voxel_p,
                cluster_threshold=cluster_threshold,
                make_surface_plot=False,
            )

    # Generate a report
    report = glm.generate_report(
        contrasts=list(CONTRASTS.values()),
        title="First-Level GLM Report",
        plot_type="glass",
        two_sided=True,
        alpha=voxel_p,
    )
    report_file = output_dir / "first_level_report.html"
    report.save_as_html(report_file)

    return glm


def run_group_level(
    glms: List[FirstLevelModel],
    output_dir: Path,
    voxel_p: float = DEFAULT_VOXEL_P,
    cluster_threshold: int = DEFAULT_CLUSTER_THRESHOLD,
    n_jobs: int = 1,
    second_level_df: Optional[pd.DataFrame] = None,
) -> None:
    """Run Second Level group analysis using fitted FirstLevelModel objects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting group-level analysis using %s",
        "precomputed maps" if second_level_df is not None else f"{len(glms)} subjects",
    )

    # Fit once if GLMs are provided; fit per contrast if using precomputed maps
    base_group_model = None
    if second_level_df is None:
        base_group_model = SecondLevelModel(smoothing_fwhm=None, n_jobs=n_jobs, verbose=1)
        base_group_model.fit(glms)

    for contrast_name, contrast_expr in tqdm(CONTRASTS.items(), desc="Group Analysis"):
        # Compute contrast at group level by propagating the first-level contrast expression
        compute_kwargs = {"output_type": "all"}
        if second_level_df is not None:
            contrast_df = (
                second_level_df[second_level_df["map_name"] == contrast_name]
                .sort_values("subject_label")
                .reset_index(drop=True)
            )
            if contrast_df.empty:
                raise ValueError(f"No maps found for contrast {contrast_name} in second_level_df")
            logger.info(f"Fitting second-level for {contrast_name} with {len(contrast_df)} maps")
            design_matrix = pd.DataFrame({"intercept": np.ones(len(contrast_df))})
            group_model = SecondLevelModel(
                smoothing_fwhm=None,
                n_jobs=n_jobs,
                verbose=1,
                minimize_memory=False,
            )
            group_model.fit(contrast_df, design_matrix=design_matrix)
            compute_kwargs["second_level_contrast"] = "intercept"
            compute_kwargs["first_level_contrast"] = contrast_name
        else:
            group_model = base_group_model
            compute_kwargs["first_level_contrast"] = contrast_expr

        eff_maps: dict = group_model.compute_contrast(**compute_kwargs)  # type: ignore
        # Save all, similar to FirstLevelModel
        for result_name, eff_map in eff_maps.items():
            eff_file = output_dir / f"group_{contrast_name}_{result_name}.nii.gz"
            if result_name == "z_score":
                # Keep the original default-threshold output for consistency
                save_brain_map(
                    eff_map,
                    eff_file,
                    is_z_map=True,
                    voxel_p=voxel_p,
                    cluster_threshold=cluster_threshold,
                    sided="both",
                )

                thresholds = [
                    (0.05, 0, "p05_cluster0"),
                    (0.001, 0, "p001_cluster0"),
                    (0.001, 50, "p001_cluster50"),
                ]

                for voxel_p_alt, cluster_thr_alt, tag in thresholds:
                    save_brain_map(
                        eff_map,
                        output_dir / f"group_{contrast_name}_{result_name}_{tag}.nii.gz",
                        is_z_map=True,
                        voxel_p=voxel_p_alt,
                        cluster_threshold=cluster_thr_alt,
                        sided="both",
                    )
            else:
                save_brain_map(
                    eff_map,
                    eff_file,
                    is_z_map=False,
                    voxel_p=voxel_p,
                    cluster_threshold=cluster_threshold,
                    sided="positive",
                )

        # Generate report for each contrast
        report = group_model.generate_report(  # type: ignore
            contrasts={contrast_name: "intercept"},
            title=f"Group GLM Report: {contrast_name}",
            plot_type="glass",
            alpha=voxel_p,
            two_sided=True,
            first_level_contrast=None if second_level_df is None else contrast_name,
        )
        report.save_as_html(output_dir / f"group_{contrast_name}_report.html")


def run_functional_localizer(
    session_runs: List[Tuple[Path, Path, Path]],
    gm_mask: Path,
    output_dir: Path,
    tr: float,
    smoothing_fwhm: float,
    gm_threshold: float,
    voxel_p: float = DEFAULT_VOXEL_P,
    cluster_threshold: int = DEFAULT_CLUSTER_THRESHOLD,
    n_jobs: int = 1,
    use_stimulus_confounds: bool = False,
) -> object | None:
    """Fit a localizer (session-1) GLM for compositionality and return thresholded mask."""
    bold_imgs, events_files, confounds_files = zip(*session_runs)
    loc_out = output_dir / "localizer"

    glm = run_subject_glm(
        bold_imgs=bold_imgs,
        events_files=events_files,
        confounds_files=confounds_files,
        gm_probseg=gm_mask,
        output_dir=loc_out,
        tr=tr,
        smoothing_fwhm=smoothing_fwhm,
        gm_threshold=gm_threshold,
        cluster_threshold=cluster_threshold,
        voxel_p=voxel_p,
        n_jobs=n_jobs,
        use_stimulus_confounds=use_stimulus_confounds,
    )

    z_map = glm.compute_contrast(CONTRASTS["main_compositionality"], output_type="z_score")
    thr_mask = save_brain_map(
        z_map,
        loc_out / "localizer_zmap.nii.gz",
        is_z_map=True,
        voxel_p=voxel_p,
        cluster_threshold=0,
        make_surface_plot=False,
        sided="both",
    )
    return thr_mask


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
    help="Root folder for subject-level and group-level outputs.",
)
@click.option(
    "--subjects",
    type=str,
    multiple=True,
    help="Subject IDs without the 'sub-' prefix; if omitted, auto-discover from fMRIPrep outputs.",
    default=("02", "03", "04", "05", "06", "07", "08", "09"),
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
    help="Voxel-wise p-value threshold applied to z-maps (two-sided).",
)
@click.option(
    "--cluster-threshold",
    type=int,
    default=DEFAULT_CLUSTER_THRESHOLD,
    show_default=True,
    help="Cluster size threshold in voxels.",
)
@click.option(
    "--run-localizer/--no-run-localizer",
    default=True,
    show_default=True,
    help="Whether to run the session-1 functional localizer workflow.",
)
@click.option(
    "--max-runs",
    type=int,
    default=None,
    help="Debug: Limit number of runs processed per subject.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=-1,
    help="Number of CPUs to use for parallel processing. -1 uses all available.",
)
@click.option(
    "--use-stimulus-confounds/--no-use-stimulus-confounds",
    default=False,
    help="Include low-level stimulus features (text/image) as confounds.",
)
@click.option(
    "--reuse-first-level/--no-reuse-first-level",
    default=True,
    help="Skip refitting FirstLevelModel if maps already exist; reuse effect/variance paths for group level.",
)
def run_univariate_glm(
    fmriprep_dir: Path,
    bids_dir: Path,
    output_dir: Path,
    subjects: Tuple[str, ...],
    tr: float,
    smoothing_fwhm: float,
    gm_threshold: float,
    voxel_p: float,
    cluster_threshold: int,
    run_localizer: bool,
    max_runs: int | None,
    n_jobs: int,
    use_stimulus_confounds: bool,
    reuse_first_level: bool,
) -> None:
    """Run subject-level and group-level univariate GLM analyses.
    
    :param fmriprep_dir: Directory with fMRIPrep outputs.
    :type fmriprep_dir: Path
    :param bids_dir: BIDS directory with events.tsv files.
    :type bids_dir: Path
    :param output_dir: Root folder for subject-level and group-level outputs.
    :type output_dir: Path
    :param subjects: Subject IDs without the 'sub-' prefix; if omitted, auto-dis
        cover from fMRIPrep outputs.
    :type subjects: Tuple[str, ...]
    :param tr: Repetition time in seconds.
    :type tr: float
    :param smoothing_fwhm: Spatial smoothing FWHM in millimeters.
    :type smoothing_fwhm: float
    :param gm_threshold: Probability threshold for the GM mask.
    :type gm_threshold: float
    :param voxel_p: Voxel-wise p-value threshold applied to z-maps (two
        -sided).
    :type voxel_p: float
    :param cluster_threshold: Cluster size threshold in voxels.
    :type cluster_threshold: int
    :param run_localizer: Whether to run the session-1 functional localizer
        workflow.
    :type run_localizer: bool
    :param max_runs: Debug: Limit number of runs processed per subject.
    :type max_runs: int | None
    :param n_jobs: Number of CPUs to use for parallel processing. -1 uses all
        available.
    :type n_jobs: int
    :param use_stimulus_confounds: Include low-level stimulus features (text/image)
        as confounds.
    :type use_stimulus_confounds: bool
    :param reuse_first_level: Reuse cached first-level maps if available instead of refitting.
    :type reuse_first_level: bool
    """
    subject_ids = (
        list(subjects)
        if subjects
        else [p.name.replace("sub-", "") for p in sorted(fmriprep_dir.glob("sub-*")) if p.is_dir()]
    )
    subject_level_dir = output_dir / "subject_level"
    group_level_dir = output_dir / "group_level"

    all_glms: List[FirstLevelModel] = []
    all_func_loc_glms: List[FirstLevelModel] = []
    second_level_rows: List[dict] = []
    func_loc_rows: List[dict] = []
    localizer_masks: List[object] = []

    for subject in subject_ids:
        logger.info(f"Running univariate GLM for subject {subject}")
        runs = collect_runs(
            fmriprep_dir=fmriprep_dir, bids_dir=bids_dir, subject=subject, max_runs=max_runs,
        )
        gm_mask = find_gm_mask(fmriprep_dir, subject)
        subject_output = subject_level_dir / f"sub-{subject}"
        sessions = split_runs_by_session(runs)
        all_runs = [item for sublist in sessions.values() for item in sublist]
        bold_imgs, events_files, confounds_files = zip(*all_runs)
        
        def _subject_has_maps(base_dir: Path, contrast: str) -> tuple[Path, Path]:
            effect = base_dir / f"{contrast}_effect_size.nii.gz"
            variance = base_dir / f"{contrast}_effect_variance.nii.gz"
            return effect, variance

        has_all_maps = all(all(_subject_has_maps(subject_output, c)[i].exists() for i in (0, 1)) for c in CONTRASTS)

        if reuse_first_level and has_all_maps:
            logger.info(f"Using cached first-level maps for subject {subject}")
            main_glm = None
        else:
            main_glm = run_subject_glm(
                bold_imgs=bold_imgs,
                events_files=events_files,
                confounds_files=confounds_files,
                gm_probseg=gm_mask,
                output_dir=subject_output,
                tr=tr,
                smoothing_fwhm=smoothing_fwhm,
                gm_threshold=gm_threshold,
                cluster_threshold=cluster_threshold,
                voxel_p=voxel_p,
                n_jobs=n_jobs,
                use_stimulus_confounds=use_stimulus_confounds,
            )
        if main_glm is not None:
            all_glms.append(main_glm)

        for contrast in CONTRASTS:
            effect_map, variance_map = _subject_has_maps(subject_output, contrast)
            if not (effect_map.exists() and variance_map.exists()):
                raise FileNotFoundError(
                    f"Missing first-level maps for subject {subject}, contrast {contrast}"
                )
            second_level_rows.append(
                {
                    "subject_label": subject,
                    "map_name": contrast,
                    "effects_map_path": str(effect_map),
                    "variance_map_path": str(variance_map),
                }
            )

        if run_localizer:
            ses1_runs = sessions.get("ses-01", [])
            heldout_runs = [item for sess, items in sessions.items() if sess != "ses-01" for item in items]

            loc_mask = run_functional_localizer(
                session_runs=ses1_runs,
                gm_mask=gm_mask,
                output_dir=subject_output,
                tr=tr,
                smoothing_fwhm=smoothing_fwhm,
                gm_threshold=gm_threshold,
                cluster_threshold=cluster_threshold,
                voxel_p=voxel_p,
                n_jobs=n_jobs,
                use_stimulus_confounds=use_stimulus_confounds,
            )

            # Save the individual localizer mask
            loc_mask_out = subject_output / "localizer_mask.nii.gz"
            save_brain_map(loc_mask, loc_mask_out, make_surface_plot=False)
            localizer_masks.append(loc_mask)

            # Compute maps for independent runs (ses-2/3) using the functional localizer from ses-1
            fl_bold, fl_events, fl_conf = zip(*heldout_runs)
            fl_glm = run_subject_glm(
                bold_imgs=fl_bold,
                events_files=fl_events,
                confounds_files=fl_conf,
                gm_probseg=loc_mask,
                output_dir=subject_output / "func_loc",
                tr=tr,
                smoothing_fwhm=smoothing_fwhm,
                gm_threshold=gm_threshold,
                cluster_threshold=cluster_threshold,
                voxel_p=voxel_p,
                n_jobs=n_jobs,
                use_stimulus_confounds=use_stimulus_confounds,
            )
            all_func_loc_glms.append(fl_glm)

            for contrast in CONTRASTS:
                effect_map, variance_map = _subject_has_maps(subject_output / "func_loc", contrast)
                if effect_map.exists() and variance_map.exists():
                    func_loc_rows.append(
                        {
                            "subject_label": subject,
                            "map_name": contrast,
                            "effects_map_path": str(effect_map),
                            "variance_map_path": str(variance_map),
                        }
                    )

    run_group_level(
        glms=all_glms,
        output_dir=group_level_dir,
        voxel_p=voxel_p,
        cluster_threshold=cluster_threshold,
        n_jobs=n_jobs,
        second_level_df=pd.DataFrame(second_level_rows) if second_level_rows else None,
    )
    if run_localizer:
        run_group_level(
            glms=all_func_loc_glms,
            output_dir=group_level_dir / "func_loc",
            voxel_p=voxel_p,
            cluster_threshold=cluster_threshold,
            n_jobs=n_jobs,
            second_level_df=pd.DataFrame(func_loc_rows) if func_loc_rows else None,
        )

    # Compute group-level conjunction
    img_contrast, txt_contrast = "img_high_vs_low", "txt_high_vs_low"
    img_map = group_level_dir / f"group_{img_contrast}_z_score_thr.nii.gz"
    txt_map = group_level_dir / f"group_{txt_contrast}_z_score_thr.nii.gz"
    conj_out = conjunction_map(img_map, txt_map)
    logger.info(f"Saved group-level conjunction map to {conj_out}")


if __name__ == "__main__":
    run_univariate_glm()
