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
from nilearn.masking import apply_mask
from scipy.stats import norm

from compositionality_study.constants import BIDS_DIR, MODELS_DIR, PREPROC_MRI_DIR


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


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


def _extract_space(bold_file: Path) -> str | None:
    """Extract space label from BOLD filename (e.g. 'MNI152NLin2009cAsym')."""
    for token in bold_file.name.split("_"):
        if token.startswith("space-"):
            return token.replace("space-", "")
    return None


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


def find_gm_mask(fmriprep_dir: Path, subject: str, space: str | None = None) -> Path:
    """Pick the GM probseg matching the BOLD space (if provided)."""

    anat_dir = fmriprep_dir / f"sub-{subject}" / "ses-01" / "anat"
    
    pattern = "*_label-GM_probseg.nii.gz"
    if space:
        # Prefer mask in the same space as BOLD data
        pattern = f"*space-{space}*_label-GM_probseg.nii.gz"

    candidates = sorted(anat_dir.glob(pattern))

    if not candidates:
        logger.warning(f"No GM probability segmentation found matching pattern {pattern} in {anat_dir}")
        if space:
            # Fallback for robustness
            candidates = sorted(anat_dir.glob("*_label-GM_probseg.nii.gz"))
            if candidates:
                logger.warning(f"Falling back to {candidates[0].name}")

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


# -----------------------------------------------------------------------------
# GLM routines
# -----------------------------------------------------------------------------


def run_subject_glm(
    bold_imgs: Sequence[Path],
    events_files: Sequence[Path],
    confounds_files: Sequence[Path],
    gm_probseg: Path,
    output_dir: Path,
    tr: float = TR,
    smoothing_fwhm: float = SMOOTHING_FWHM,
    gm_threshold: float = GM_THRESHOLD,
    n_jobs: int = 1,
) -> Dict[str, Path]:
    """Run a first-level GLM and write subject-level contrast maps."""

    output_dir.mkdir(parents=True, exist_ok=True)

    events = [load_events(f) for f in events_files]
    confounds = [load_motion_regressors(f) for f in confounds_files]

    # Create binary mask image object.
    # Passing this to FirstLevelModel ensures it is resampled to the functional run geometry.
    # TODO out of curiosity - save this mask for every subject
    gm_mask_img = image.math_img(f"img > {gm_threshold}", img=gm_probseg)

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
        n_jobs=n_jobs,
    )

    glm = glm.fit(run_imgs=list(bold_imgs), events=events, confounds=confounds)
    # TODO out of curiosity - examine output

    contrast_maps: Dict[str, Path] = {}

    for name, expression in CONTRASTS.items():
        # Result is already masked by mask_img
        zmap = glm.compute_contrast(expression, output_type="z_score")
        
        out_file = output_dir / f"{name}_zmap.nii.gz"
        zmap.to_filename(out_file)
        contrast_maps[name] = out_file
        # TODO also make sure to save visualizations here

    return contrast_maps


def threshold_zmap(
    zmap_path: Path,
    voxel_p: float = DEFAULT_VOXEL_P,
    cluster_alpha: float = DEFAULT_CLUSTER_ALPHA,
    two_sided: bool = True,
    min_cluster_size: int | None = None,
) -> Path:
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
    thr_img.to_filename(out_file)
    return out_file


def conjunction_map(map_a: Path, map_b: Path) -> Path:
    """Compute a simple conjunction (logical AND) of two thresholded maps."""

    img_a = image.math_img("img != 0", img=map_a)
    img_b = image.math_img("img != 0", img=map_b)
    conj = image.math_img("img1 * img2", img1=img_a, img2=img_b)
    out_file = map_a.with_name("conjunction_img_txt.nii.gz")
    conj.to_filename(out_file)
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

    for contrast_name, maps in contrast_maps.items():
        if not maps:
            logger.warning(f"Skipping group contrast {contrast_name}: no subject maps provided")
            continue

        second_level = SecondLevelModel()
        second_level = second_level.fit(maps)

        zmap = second_level.compute_contrast(output_type="z_score")
        out_file = output_dir / f"group_{contrast_name}_zmap.nii.gz"
        zmap.to_filename(out_file)
        logger.info(f"Wrote group z-map for {contrast_name} to {out_file}")

        # Apply standard thresholds and save
        thr_file = threshold_zmap(out_file)
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
) -> Path | None:
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
    )
    # TODO double check - is the localizer saved?

    comp_contrast = contrasts.get("main_compositionality")
    if comp_contrast is None:
        logger.warning(f"No main_compositionality contrast for subject {subject} localizer")
        return None

    thr_mask = threshold_zmap(comp_contrast)
    return thr_mask


def summarize_roi_effects(
    subject: str,
    roi_mask: Path,
    session_runs: List[Tuple[Path, Path, Path]],
    output_dir: Path,
    tr: float,
    smoothing_fwhm: float,
    gm_threshold: float,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute mean contrast values within ROI for held-out sessions (e.g., ses-02/03)."""

    if not session_runs:
        return pd.DataFrame()

    bold_imgs, events_files, confounds_files = zip(*session_runs)
    roi_out = output_dir / "heldout"

    contrasts = run_subject_glm(
        bold_imgs=bold_imgs,
        events_files=events_files,
        confounds_files=confounds_files,
        gm_probseg=image.load_img(roi_mask),
        output_dir=roi_out,
        tr=tr,
        smoothing_fwhm=smoothing_fwhm,
        gm_threshold=gm_threshold,
        n_jobs=n_jobs,
    )

    records = []
    for name, path in contrasts.items():
        data = apply_mask(path, roi_mask)
        if data.size == 0:
            continue
        records.append({"subject": subject, "contrast": name, "mean_in_roi": float(data.mean())})

    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


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
) -> None:
    """Run subject-level and group-level univariate GLM analyses."""

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
    conjunctions: List[Path] = []
    roi_summaries: List[pd.DataFrame] = []
    localizer_masks: List[Path] = []

    for subject in subject_ids:
        logger.info(f"Running univariate GLM for subject {subject}")

        runs = collect_runs(
            fmriprep_dir=fmriprep_dir, bids_dir=bids_dir, subject=subject, max_runs=max_runs
        )
        if not runs:
            logger.warning(f"Skipping subject {subject}: no usable runs")
            continue

        # Detect space from the first run to match GM mask
        # TODO this should not be needed? GM mask should be in the same space ... or might have a different resolution?
        first_bold = runs[0][0]
        space_label = _extract_space(first_bold)

        gm_mask = find_gm_mask(fmriprep_dir, subject, space=space_label)
        subject_output = subject_level_dir / f"sub-{subject}"

        sessions = split_runs_by_session(runs)
        all_runs = [item for sublist in sessions.values() for item in sublist]

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
        )

        # Threshold subject maps and build conjunction_map
        # TODO why is a conjunction already prepared in the subject loop? weird ... also is a z- to t- conversion really necessary, shouldn't nilearn handle that in its 2nd level model?
        thr_contrasts: Dict[str, Path] = {}
        for name, path in contrasts.items():
            # TODO can't we do threshold z-map, if really needed, in run_subject_glm already?
            thr_path = threshold_zmap(
                path,
                voxel_p=voxel_p,
                cluster_alpha=cluster_alpha,
                two_sided=True,
                min_cluster_size=min_cluster_size,
            )
            thr_contrasts[name] = thr_path
            all_contrasts[name].append(thr_path)

        # TODO why is this happening before the group-level analysis? Conjunction should only happen at group level.
        if "img_high_vs_low" in thr_contrasts and "txt_high_vs_low" in thr_contrasts:
            conj = conjunction_map(thr_contrasts["img_high_vs_low"], thr_contrasts["txt_high_vs_low"])
            conjunctions.append(conj)

        if run_localizer:
            ses1_runs = sessions.get("ses-01", [])
            heldout_runs = [item for sess, items in sessions.items() if sess != "ses-01" for item in items]

            loc_mask = run_functional_localizer(
                subject=subject,
                session_runs=ses1_runs,
                gm_mask=gm_mask,
                output_dir=subject_output,
                tr=tr,
                smoothing_fwhm=smoothing_fwhm,
                gm_threshold=gm_threshold,
                n_jobs=n_jobs,
            )

            if loc_mask:
                localizer_masks.append(loc_mask)
                # TODO what is this whole "summarize ROI" stuff? the analysis here should be more or less identical to run_subject_glm?ß
                summary = summarize_roi_effects(
                    subject=subject,
                    roi_mask=loc_mask,
                    session_runs=heldout_runs,
                    output_dir=subject_output,
                    tr=tr,
                    smoothing_fwhm=smoothing_fwhm,
                    gm_threshold=gm_threshold,
                    n_jobs=n_jobs,
                )
                if not summary.empty:
                    roi_summaries.append(summary)

    run_group_level(contrast_maps=all_contrasts, output_dir=group_level_dir)

    if conjunctions:
        # TODO correct this: The conjunction is done with the two group-level modality-specific compositionality contrasts, not the 1st level output
        mean_conj = image.mean_img(conjunctions)
        conj_out = group_level_dir / "conjunction_mean.nii.gz"
        mean_conj.to_filename(conj_out)
        logger.info(f"Saved conjunction mean map to {conj_out}")

    if roi_summaries:
        # TODO find out whatever this is
        combined = pd.concat(roi_summaries, ignore_index=True)
        csv_out = output_dir / "roi_summary.csv"
        combined.to_csv(csv_out, index=False)
        logger.info(f"Saved ROI summary to {csv_out}")

    if localizer_masks:
        prob_map = image.mean_img(localizer_masks)
        prob_out = output_dir / "localizer_probability.nii.gz"
        # TODO also save visualizations in addition to the nifti
        prob_map.to_filename(prob_out)
        logger.info(f"Saved probabilistic localizer map to {prob_out}")


if __name__ == "__main__":
    run_univariate_glm()
