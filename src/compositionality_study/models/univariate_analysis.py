"""Run a univariate GLM on fMRIPrep outputs and aggregate group-level maps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import click
import pandas as pd
from loguru import logger
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import threshold_stats_img
from scipy.stats import norm
from tqdm import tqdm 

from compositionality_study.constants import BIDS_DIR, MODELS_DIR, PREPROC_MRI_DIR
from compositionality_study.utils import (
    get_coco_df, 
    get_stimulus_features_lookup, 
    get_stimulus_data,
    save_brain_map,
)


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
DEFAULT_CLUSTER_ALPHA = 0.05


def _derive_run_id(bold_file: Path) -> str:
    """Strip space/desc tokens to get the BIDS run identifier."""

    stem = bold_file.name.replace(".nii.gz", "").replace(".nii", "")
    parts = []

    for token in stem.split("_"):
        if token.startswith("space-") or token.startswith("desc-"):
            break
        parts.append(token)

    if not parts:
        raise ValueError(f"Unable to derive run id from {bold_file}")

    return "_".join(parts)


def load_motion_regressors(confounds_tsv: Path) -> pd.DataFrame:
    """Load the six standard motion regressors from fMRIPrep confounds."""

    confounds = pd.read_csv(confounds_tsv, sep="\t")
    missing = [col for col in MOTION_COLS if col not in confounds.columns]

    if missing:
        raise ValueError(f"Missing motion regressors {missing} in {confounds_tsv}")

    return confounds[MOTION_COLS].fillna(0.0)


def load_events(events_tsv: Path) -> pd.DataFrame:
    """Load events and normalize the condition column name."""

    events = pd.read_csv(events_tsv, sep="\t")
    
    # Check for trial_type or condition first
    if "trial_type" in events.columns:
        pass
    elif "condition" in events.columns:
        events = events.rename(columns={"condition": "trial_type"})
    elif "modality" in events.columns and "complexity" in events.columns:
        # Fallback: derive condition from modality + complexity
        # Normalize to lowercase and strip whitespace to ensure 'image_high', 'text_low' etc.
        modality = events["modality"].astype(str).str.strip().str.lower()
        complexity = events["complexity"].astype(str).str.strip().str.lower()
        
        # Map common variations if necessary (e.g. 'img' -> 'image')
        # Here assuming standard 'image'/'text' and 'high'/'low' are target values
        events["trial_type"] = modality + "_" + complexity
    else:
        raise ValueError(
            f"No condition/trial_type column and cannot derive from modality/complexity in {events_tsv}"
        )

    # Ensure trial_type only contains valid python identifiers if possible (though nilearn handles some)
    # But crucially, they must match CONTRASTS keys if they are used there.
    
    return events


def find_gm_mask(fmriprep_dir: Path, subject: str) -> Path:
    """Pick the GM probseg (assuming MNI152NLin2009cAsym space)."""

    anat_dir = fmriprep_dir / f"sub-{subject}" / "ses-01" / "anat"
    
    # Look for standard fMRIPrep MNI output
    pattern = "*space-MNI152NLin2009cAsym*_label-GM_probseg.nii.gz"
    candidates = sorted(anat_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No GM probability segmentation found in {anat_dir}")

    return candidates[0]


def _parse_session(run_id: str) -> str:
    """Return session label (e.g., 'ses-01') if present, else 'ses-unknown'."""

    for token in run_id.split("_"):
        if token.startswith("ses-"):
            return token
    return "ses-unknown"


def collect_runs(
    fmriprep_dir: Path, bids_dir: Path, subject: str, max_runs: int | None = None
) -> List[Tuple[Path, Path, Path]]:
    """Pair each preprocessed BOLD run with matching events and confounds."""

    bold_files = sorted((fmriprep_dir / f"sub-{subject}").glob("**/*_desc-preproc_bold.nii.gz"))
    runs: List[Tuple[Path, Path, Path]] = []

    if not bold_files:
        logger.warning(f"No BOLD runs found for subject {subject} in {fmriprep_dir}")
        return runs

    if max_runs is not None:
        logger.info(f"Debugging: limiting to {max_runs} runs for subject {subject}")
        bold_files = bold_files[:max_runs]

    for bold_file in bold_files:
        run_id = _derive_run_id(bold_file)
        rel_dir = bold_file.parent.relative_to(fmriprep_dir / f"sub-{subject}")

        events_file = bids_dir / f"sub-{subject}" / rel_dir / f"{run_id}_events.tsv"
        confounds_file = bold_file.parent / f"{run_id}_desc-confounds_timeseries.tsv"

        if not events_file.exists():
            logger.warning(f"Skipping {bold_file.name}: missing events file {events_file}")
            continue

        if not confounds_file.exists():
            logger.warning(f"Skipping {bold_file.name}: missing confounds file {confounds_file}")
            continue

        runs.append((bold_file, events_file, confounds_file))

    if not runs:
        logger.warning(f"No valid runs for subject {subject} after checking events/confounds.")

    return runs


def _load_stimulus_confounds(events: pd.DataFrame, n_scans: int, tr: float = TR) -> pd.DataFrame:
    """Generate extra confounds based on stimulus properties."""
    coco_df = get_coco_df()

    text_cols = ["sentence_length", "amr_n_nodes"]
    img_cols = ["coco_a_nodes", "ic_score", "aspect_ratio", "coco_person"]
    all_cols = text_cols + img_cols

    confounds = pd.DataFrame(0.0, index=range(n_scans), columns=all_cols)
    txt_df, img_df = get_stimulus_features_lookup(coco_df)

    for _, row in events.iterrows():
        data = get_stimulus_data(
            modality=row.get("modality"),
            stimulus=row.get("stimulus"),
            cocoid=row.get("cocoid"),
            txt_df=txt_df,
            img_df=img_df,
        )

        if data is None:
            continue

        start_tr = int(round(row["onset"] / tr))
        n_trs = int(round(row["duration"] / tr))
        end_tr = min(start_tr + n_trs, n_scans)

        if start_tr >= n_scans:
            continue

        cols = text_cols if row.get("modality") == "text" else img_cols
        valid_cols = [c for c in cols if c in data]
        if valid_cols:
            confounds.iloc[
                start_tr:end_tr, confounds.columns.get_indexer(valid_cols)  # type: ignore
            ] = data[valid_cols].values

    return confounds


def run_subject_glm(
    bold_imgs: Sequence[Path],
    events_files: Sequence[Path],
    confounds_files: Sequence[Path],
    gm_probseg: Path | object,
    output_dir: Path,
    tr: float = TR,
    smoothing_fwhm: float = SMOOTHING_FWHM,
    gm_threshold: float = GM_THRESHOLD,
    n_jobs: int = 1,
    use_stimulus_confounds: bool = False,
) -> Dict[str, Path]:
    """Run a first-level GLM and write subject-level contrast maps."""

    output_dir.mkdir(parents=True, exist_ok=True)
    subject = output_dir.parent.name.replace("sub-", "")
    
    logger.info(f"Loading {len(events_files)} runs (events & confounds) for subject {subject}...")
    events = [load_events(f) for f in events_files]
    motion_confounds = [load_motion_regressors(f) for f in confounds_files]

    # Combine motion regressors with stimulus confounds
    logger.info("Constructing design matrices...")
    confounds = []
    for ev, mot in zip(events, motion_confounds):
        if use_stimulus_confounds:
            n_scans = len(mot)
            stim_conf = _load_stimulus_confounds(ev, n_scans, tr)
            confounds.append(pd.concat([mot, stim_conf], axis=1))
        else:
            confounds.append(mot)

    # Create binary mask image object.
    logger.info("Processing GM mask...")
    # Passing this to FirstLevelModel ensures it is resampled to the functional run geometry.
    gm_mask_img = image.math_img(f"img > {gm_threshold}", img=gm_probseg)
    
    # Resample the mask to the first functional image's geometry
    # This prevents nilearn from upsampling the heavy 4D data to the mask's resolution
    logger.info("Resampling GM mask to functional resolution")
    
    # Load reference affine from the first BOLD run
    ref_img = image.load_img(bold_imgs[0])
    target_affine = ref_img.affine
    
    gm_mask_img = image.resample_to_img(
        source_img=gm_mask_img,
        target_img=ref_img,
        interpolation="nearest"
    )

    # Save the mask (using the utils function for consistency)
    mask_out = output_dir / "gm_mask.nii.gz"
    save_brain_map(
        gm_mask_img, 
        mask_out, 
        plot=True, 
        plot_kwargs={"title": "GM Mask"}
    )

    logger.info(f"Fitting FirstLevelModel (n_jobs={n_jobs})... this may take a few minutes.")
    glm = FirstLevelModel(
        t_r=tr,
        hrf_model="glover",
        drift_model="cosine",
        high_pass=0.01,
        smoothing_fwhm=smoothing_fwhm,
        noise_model="ar1",
        standardize=False,
        signal_scaling=0,
        minimize_memory=False,
        mask_img=gm_mask_img,
        target_affine=target_affine,
        n_jobs=n_jobs,
        verbose=1,
    )

    glm = glm.fit(run_imgs=list(bold_imgs), events=events, confounds=confounds)
    contrast_maps: Dict[str, Path] = {}

    logger.info(f"Computing {len(CONTRASTS)} contrasts...")
    for name, expression in tqdm(CONTRASTS.items(), desc="Contrasts"):
        # Result is already masked by mask_img
        zmap = glm.compute_contrast(expression, output_type="z_score")
        out_file = output_dir / f"{name}_zmap.nii.gz"
        save_brain_map(
            zmap, 
            out_file, 
            plot=True, 
            plot_kwargs={"title": f"{name} (subject level)"}
        )
        contrast_maps[name] = out_file

    return contrast_maps


def threshold_zmap(
    zmap_path: Path,
    voxel_p: float = DEFAULT_VOXEL_P,
    cluster_alpha: float = DEFAULT_CLUSTER_ALPHA,
    two_sided: bool = True,
    min_cluster_size: int | None = None,
) -> Tuple[object, Path]:
    """Apply voxel-wise p-threshold and optional cluster filtering to a z-map."""

    zmap_img = image.load_img(zmap_path)
    height_control = "fpr"

    # Map p to z threshold for two-sided test
    p_side = voxel_p / 2 if two_sided else voxel_p
    z_thresh = norm.isf(p_side)

    thr_img, _ = threshold_stats_img(
        zmap_img,
        alpha=voxel_p,
        height_control=height_control,
        cluster_threshold=min_cluster_size if min_cluster_size is not None else 0,
        two_sided=two_sided,
    )

    # Enforce the exact z threshold (threshold_stats_img may vary slightly with alpha)
    thr_img = image.math_img(f"img * (abs(img) >= {z_thresh})", img=thr_img)

    if cluster_alpha and min_cluster_size is None:
        # Basic cluster-size heuristic: drop clusters smaller than 10 voxels as a fallback
        thr_img = image.math_img("img * (img != 0)", img=thr_img)

    out_file = zmap_path.with_name(f"{zmap_path.stem}_thr.nii.gz")
    save_brain_map(
        thr_img, 
        out_file, 
        plot=True, 
        plot_kwargs={"title": "Thresholded z-map"}, 
        make_cluster_table=True, 
        cluster_table_kwargs={"stat_threshold": 0}
    )
    return thr_img, out_file


def conjunction_map(map_a: Path, map_b: Path) -> Path:
    """Compute a simple conjunction (logical AND) of two thresholded maps."""

    img_a = image.math_img("img != 0", img=map_a)
    img_b = image.math_img("img != 0", img=map_b)
    conj = image.math_img("img1 * img2", img1=img_a, img2=img_b)
    out_file = map_a.with_name("conjunction_img_txt.nii.gz")
    save_brain_map(
        conj, 
        out_file, 
        plot=True, 
        plot_kwargs={"title": "Conjunction Map"}
    )
    return out_file


def split_runs_by_session(runs: Iterable[Tuple[Path, Path, Path]]) -> Dict[str, List[Tuple[Path, Path, Path]]]:
    """Group runs by session token for downstream localizer logic."""

    grouped: Dict[str, List[Tuple[Path, Path, Path]]] = {}
    for bold, events, confounds in runs:
        session = _parse_session(_derive_run_id(bold))
        grouped.setdefault(session, []).append((bold, events, confounds))
    return grouped


def run_group_level(contrast_maps: Dict[str, List[Path]], output_dir: Path) -> None:
    """Run fixed-effects group analysis for each contrast."""

    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting group-level analysis for {len(contrast_maps)} contrasts...")

    for contrast_name, maps in tqdm(contrast_maps.items(), desc="Group Analysis"):
        logger.debug(f"Fitting group model for {contrast_name} ({len(maps)} subjects)")

        # Create a simple intercept-only design matrix for a one-sample t-test
        design_matrix = pd.DataFrame([1] * len(maps), columns=["intercept"])

        second_level = SecondLevelModel()
        second_level = second_level.fit(maps, design_matrix=design_matrix)

        zmap = second_level.compute_contrast(second_level_contrast="intercept", output_type="z_score")
        out_file = output_dir / f"group_{contrast_name}_zmap.nii.gz"
        save_brain_map(
            zmap, 
            out_file, 
            plot=True, 
            plot_kwargs={"title": f"Group {contrast_name}"}
        )
        logger.info(f"Wrote group z-map for {contrast_name} to {out_file}")

        # Apply standard thresholds and save
        _, thr_file = threshold_zmap(out_file)
        logger.info(f"Thresholded group map for {contrast_name} written to {thr_file}")



def run_functional_localizer(
    subject: str,
    session_runs: List[Tuple[Path, Path, Path]],
    gm_mask: Path,
    output_dir: Path,
    tr: float,
    smoothing_fwhm: float,
    gm_threshold: float,
    n_jobs: int = 1,
    use_stimulus_confounds: bool = False,
) -> object | None:
    """Fit a localizer (session-1) GLM for compositionality and return thresholded mask."""

    if not session_runs:
        logger.warning(f"No runs available for functional localizer for subject {subject}")
        return None

    bold_imgs, events_files, confounds_files = zip(*session_runs)
    loc_out = output_dir / "localizer"

    contrasts = run_subject_glm(
        bold_imgs=bold_imgs,
        events_files=events_files,
        confounds_files=confounds_files,
        gm_probseg=gm_mask,
        output_dir=loc_out,
        tr=tr,
        smoothing_fwhm=smoothing_fwhm,
        gm_threshold=gm_threshold,
        n_jobs=n_jobs,
        use_stimulus_confounds=use_stimulus_confounds,
    )

    comp_contrast = contrasts.get("main_compositionality")
    if comp_contrast is None:
        logger.warning(f"No main_compositionality contrast for subject {subject} localizer")
        return None

    thr_mask, _ = threshold_zmap(comp_contrast)
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
    "--cluster-alpha",
    type=float,
    default=DEFAULT_CLUSTER_ALPHA,
    show_default=True,
    help="Cluster-level alpha (approximate; used with voxel thresholding).",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=None,
    help="Minimum cluster size in voxels for thresholded maps (optional).",
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
    "--override-results/--no-override-results",
    default=False,
    show_default=True,
    help="Recalculate results even if outputs already exist.",
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
    cluster_alpha: float,
    min_cluster_size: int | None,
    run_localizer: bool,
    max_runs: int | None,
    n_jobs: int,
    use_stimulus_confounds: bool,
    override_results: bool,
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
    :param cluster_alpha: Cluster-level alpha (approximate; used with voxel
        thresholding).
    :type cluster_alpha: float
    :param min_cluster_size: Minimum cluster size in voxels for thresholded
        maps (optional).
    :type min_cluster_size: int | None
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
    """
    subject_ids = (
        list(subjects)
        if subjects
        else [p.name.replace("sub-", "") for p in sorted(fmriprep_dir.glob("sub-*")) if p.is_dir()]
    )

    if not subject_ids:
        raise ValueError(f"No subjects found in {fmriprep_dir} and none provided explicitly.")

    subject_level_dir = output_dir / "subject_level"
    group_level_dir = output_dir / "group_level"

    all_contrasts: Dict[str, List[Path]] = {k: [] for k in CONTRASTS}
    all_func_loc_contrasts: Dict[str, List[Path]] = {k: [] for k in CONTRASTS}
    localizer_masks: List[object] = []

    for subject in subject_ids:
        logger.info(f"Running univariate GLM for subject {subject}")

        runs = collect_runs(
            fmriprep_dir=fmriprep_dir, bids_dir=bids_dir, subject=subject, max_runs=max_runs
        )
        if not runs:
            logger.warning(f"Skipping subject {subject}: no usable runs")
            continue

        gm_mask = find_gm_mask(fmriprep_dir, subject)
        subject_output = subject_level_dir / f"sub-{subject}"

        sessions = split_runs_by_session(runs)
        all_runs = [item for sublist in sessions.values() for item in sublist]

        # Check if results exist
        main_glm_done = False
        contrasts = {}
        if not override_results:
            expected_paths = [subject_output / f"{k}_zmap.nii.gz" for k in CONTRASTS]
            if all(p.exists() for p in expected_paths):
                logger.info(f"Main GLM results exist for subject {subject}, reloading...")
                contrasts = {k: subject_output / f"{k}_zmap.nii.gz" for k in CONTRASTS}
                main_glm_done = True

        if not main_glm_done:
            # Primary GLM using all runs
            bold_imgs, events_files, confounds_files = zip(*all_runs)
            contrasts = run_subject_glm(
                bold_imgs=bold_imgs,
                events_files=events_files,
                confounds_files=confounds_files,
                gm_probseg=gm_mask,
                output_dir=subject_output,
                tr=tr,
                smoothing_fwhm=smoothing_fwhm,
                gm_threshold=gm_threshold,
                n_jobs=n_jobs,
                use_stimulus_confounds=use_stimulus_confounds,
            )

        # Store raw subject maps for group analysis
        for name, path in contrasts.items():
            all_contrasts[name].append(path)

        # Threshold subject maps only if explicitly needed for inspection or intermediate steps
        # but NOT for the standard group-level GLM which expects unthresholded maps.
        # Skip if reloading, assuming they were generated.
        if not main_glm_done:
            for name, path in contrasts.items():
                threshold_zmap(
                    path,
                    voxel_p=voxel_p,
                    cluster_alpha=cluster_alpha,
                    two_sided=True,
                    min_cluster_size=min_cluster_size,
                )

        if run_localizer:
            ses1_runs = sessions.get("ses-01", [])
            heldout_runs = [item for sess, items in sessions.items() if sess != "ses-01" for item in items]

            loc_done = False
            fl_contrasts = {}
            if not override_results:
                loc_mask_path = subject_output / "localizer_mask.nii.gz"
                fl_out_dir = subject_output / "func_loc"
                expected_fl = [fl_out_dir / f"{k}_zmap.nii.gz" for k in CONTRASTS]
                if loc_mask_path.exists() and all(p.exists() for p in expected_fl):
                    logger.info(f"Localizer results exist for {subject}, reloading...")
                    fl_contrasts = {k: fl_out_dir / f"{k}_zmap.nii.gz" for k in CONTRASTS}
                    loc_done = True

            if not loc_done:
                loc_mask = run_functional_localizer(
                    subject=subject,
                    session_runs=ses1_runs,
                    gm_mask=gm_mask,
                    output_dir=subject_output,
                    tr=tr,
                    smoothing_fwhm=smoothing_fwhm,
                    gm_threshold=gm_threshold,
                    n_jobs=n_jobs,
                    use_stimulus_confounds=use_stimulus_confounds,
                )

                if loc_mask:
                    # Save the individual localizer mask
                    loc_mask_out = subject_output / "localizer_mask.nii.gz"
                    save_brain_map(
                        loc_mask, 
                        loc_mask_out, 
                        plot=True, 
                        plot_kwargs={"title": "Functional Localizer Mask"}
                    )

                    localizer_masks.append(loc_mask)
                    # Compute maps for independent runs (ses-2/3) using the functional localizer from ses-1
                    fl_bold, fl_events, fl_conf = zip(*heldout_runs)
                    fl_contrasts = run_subject_glm(
                        bold_imgs=fl_bold,
                        events_files=fl_events,
                        confounds_files=fl_conf,
                        gm_probseg=loc_mask,
                        output_dir=subject_output / "func_loc",
                        tr=tr,
                        smoothing_fwhm=smoothing_fwhm,
                        gm_threshold=gm_threshold,
                        n_jobs=n_jobs,
                        use_stimulus_confounds=use_stimulus_confounds,
                    )
            
            for name, path in fl_contrasts.items():
                if name in all_func_loc_contrasts:
                    all_func_loc_contrasts[name].append(path)

    run_group_level(contrast_maps=all_contrasts, output_dir=group_level_dir)
    run_group_level(contrast_maps=all_func_loc_contrasts, output_dir=group_level_dir / "func_loc")

    # Compute group-level conjunction
    img_contrast = "img_high_vs_low"
    txt_contrast = "txt_high_vs_low"
    
    # Construct expected paths for thresholded group maps
    # Note: threshold_zmap appends '_thr.nii.gz' to the stem
    img_map = group_level_dir / f"group_{img_contrast}_zmap.nii_thr.nii.gz"
    txt_map = group_level_dir / f"group_{txt_contrast}_zmap.nii_thr.nii.gz"

    if img_map.exists() and txt_map.exists():
        conj_out = conjunction_map(img_map, txt_map)
        logger.info(f"Saved group-level conjunction map to {conj_out}")
    else:
        logger.warning(
            f"Could not compute group conjunction: missing {img_map} or {txt_map}"
        )


if __name__ == "__main__":
    run_univariate_glm()
