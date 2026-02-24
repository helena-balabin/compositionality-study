"""Utils for the compositionality project."""

from typing import Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path

import amrlib
import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
import penman
from datasets import load_dataset
from loguru import logger
from nilearn import image, plotting
from nilearn.glm.thresholding import threshold_stats_img
from penman.exceptions import DecodeError
from PIL import Image, ImageOps
from spacy.language import Language  # type: ignore
from spacy.tokens import Doc, Token

from compositionality_study.constants import HF_DATASET_NAME

# Set up the spacy amrlib extension
amrlib.setup_spacy_extension()

_COCO_DF = None

DEFAULT_VOXEL_P = 0.001
DEFAULT_CLUSTER_THRESHOLD = 50


def get_coco_df() -> pd.DataFrame:
    """Load and cache the COCO dataset."""
    global _COCO_DF
    if _COCO_DF is None:
        ds = load_dataset(HF_DATASET_NAME, split="train")
        df = ds.to_pandas()
        if isinstance(df, pd.DataFrame):
            _COCO_DF = df
        else:
            raise ValueError("Expected DataFrame")
    return _COCO_DF # type: ignore


def get_stimulus_features_lookup(coco_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create optimized lookup tables for text and image features."""
    txt_df = (
        coco_df.drop_duplicates("sentences_raw").set_index("sentences_raw")
        if "sentences_raw" in coco_df.columns
        else pd.DataFrame()
    )
    img_df = (
        coco_df.drop_duplicates("cocoid").set_index("cocoid")
        if "cocoid" in coco_df.columns
        else pd.DataFrame()
    )
    return txt_df, img_df


def get_stimulus_data(
    modality: Optional[str],
    stimulus: Optional[str],
    cocoid: Union[str, float, int, None],
    txt_df: pd.DataFrame,
    img_df: pd.DataFrame,
) -> Union[pd.Series, None]:  # type: ignore
    """Retrieve features for a single stimulus event."""
    if modality == "text" and stimulus in txt_df.index:
        res = txt_df.loc[stimulus]
        return res if isinstance(res, pd.Series) else res.iloc[0]
    elif modality == "image":
        if pd.notna(cocoid):
            if cocoid in img_df.index:
                res = img_df.loc[cocoid]
                return res if isinstance(res, pd.Series) else res.iloc[0]
            try:
                # Handle potential float/string mismatches
                int_cid = int(cocoid)  # type: ignore
                if int_cid in img_df.index:
                    res = img_df.loc[int_cid]
                    return res if isinstance(res, pd.Series) else res.iloc[0]
            except (ValueError, TypeError):
                pass
    return None


def derive_run_id(bold_file: Path) -> str:
    """Strip space/desc tokens to get the BIDS run identifier."""
    stem = bold_file.name.replace(".nii.gz", "").replace(".nii", "")
    parts: List[str] = []

    for token in stem.split("_"):
        if token.startswith("space-") or token.startswith("desc-"):
            break
        parts.append(token)

    return "_".join(parts)


def load_events(events_tsv: Path) -> pd.DataFrame:
    """Load events and normalize the condition column name."""
    events = pd.read_csv(events_tsv, sep="\t")

    modality = events["modality"].astype(str).str.strip().str.lower()
    complexity = events["complexity"].astype(str).str.strip().str.lower()
    events["trial_type"] = modality + "_" + complexity

    if "trial_type" in events.columns:
        events = events.dropna(subset=["trial_type"])
        events["trial_type"] = events["trial_type"].astype(str)

    return events


def find_gm_mask(fmriprep_dir: Path, subject: str) -> Path:
    """Pick the GM probseg (assuming MNI152NLin2009cAsym space)."""

    anat_dir = fmriprep_dir / f"sub-{subject}" / "ses-01" / "anat"
    candidates = sorted(anat_dir.glob("*space-MNI152NLin2009cAsym*_label-GM_probseg.nii.gz"))
    return candidates[0]


def collect_runs(
    fmriprep_dir: Path, bids_dir: Path, subject: str, max_runs: int | None = None
) -> List[Tuple[Path, Path, Path]]:
    """Pair each preprocessed BOLD run with matching events and confounds."""

    bold_files = sorted((fmriprep_dir / f"sub-{subject}").glob("**/*_desc-preproc_bold.nii.gz"))
    runs: List[Tuple[Path, Path, Path]] = []

    if max_runs is not None:
        logger.info(f"Debugging: limiting to {max_runs} runs for subject {subject}")
        bold_files = bold_files[:max_runs]

    for bold_file in bold_files:
        run_id = derive_run_id(bold_file)
        rel_dir = bold_file.parent.relative_to(fmriprep_dir / f"sub-{subject}")

        events_file = bids_dir / f"sub-{subject}" / rel_dir / f"{run_id}_events.tsv"
        confounds_file = bold_file.parent / f"{run_id}_desc-confounds_timeseries.tsv"

        runs.append((bold_file, events_file, confounds_file))

    return runs


def load_stimulus_confounds(events: pd.DataFrame, n_scans: int, tr: float) -> pd.DataFrame:
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


def conjunction_map(map_a: Path, map_b: Path) -> Path:
    """Compute a simple conjunction (logical AND) of two thresholded maps."""

    img_a = image.math_img("img != 0", img=map_a)
    img_b = image.math_img("img != 0", img=map_b)
    conj = image.math_img("img1 * img2", img1=img_a, img2=img_b)
    out_file = map_a.with_name("conjunction_img_txt.nii.gz")
    save_brain_map(conj, out_file)
    return out_file


def split_runs_by_session(runs: Iterable[Tuple[Path, Path, Path]]) -> Dict[str, List[Tuple[Path, Path, Path]]]:
    """Group runs by session token for downstream localizer logic."""

    grouped: Dict[str, List[Tuple[Path, Path, Path]]] = {}
    for bold, events, confounds in runs:
        run_id = derive_run_id(bold)
        session = run_id.split("_")[1]
        grouped.setdefault(session, []).append((bold, events, confounds))
    return grouped



def get_aspect_ratio(filepath: str):
    """Get the aspect ratio of the image from the local directory.

    :param filepath: The path to the image file
    :type filepath: str
    :return: The aspect ratio of the image
    :rtype: float
    """
    # Load the image
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            return width / height
    except Exception as e:  # noqa
        return 0.0


def get_amr_graph_depth(
    amr_graph: str,
    return_graph=False,
) -> Union[int, Tuple[int, nx.DiGraph]]:
    """Get the depth of the AMR graph for a given example.

    :param amr_graph: The AMR graph to get the depth for (output of a spacy doc._.to_amr()[0] call)
    :type amr_graph: str
    :param return_graph: Whether to return the networkx graph, defaults to False
    :type return_graph: bool, optional
    :return: The maximum "depth" of the AMR graph (longest shortest path)
    :rtype: int
    """
    # Convert to a Penman graph (with de-inverted edges)
    penman_graph = penman.decode(amr_graph)
    # Convert to a nx graph, first initialize the nx graph
    nx_graph = nx.DiGraph()
    # Add edges
    for e in penman_graph.edges():
        nx_graph.add_edge(e.source, e.target)
    # Get the characteristic path length of the graph
    amr_graph_depth = (
        max([max(nx.shortest_path_length(nx_graph, source=n).values()) for n in nx_graph.nodes()])
        if nx.number_of_nodes(nx_graph) > 0
        else 0
    )

    if return_graph:
        return amr_graph_depth, nx_graph
    else:
        return amr_graph_depth


def walk_tree(
    node: Token,
    depth: int,
) -> int:
    """Walk the dependency parse tree and return the maximum depth.

    :param node: The current node in the tree
    :type node: spacy.tokens.Token
    :param depth: The current depth in the tree
    :type depth: int
    :return: The maximum depth in the tree
    :rtype: int
    """
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


def derive_text_depth_features(
    examples: Dict[str, List],
    nlp: Language,
) -> Dict[str, List]:
    """Get the depth of the dep parse tree, number of verbs and "depth" of the AMR graph of an example caption.

    The AMR model needs to be downloaded separately, see https://github.com/bjascob/amrlib-models.

    :param examples: A batch of hf dataset examples
    :type examples: Dict[str, List]
    :param nlp: Spacy pipeline to use, initialized using nlp = spacy.load("en_core_web_trf")
    :type nlp: spacy.language.Language
    :return: The batch with the added features
    :rtype: Dict[str, List]
    """
    result: Dict = {
        "parse_tree_depth": [],
        "n_verbs": [],
        "amr_graph_depth": [],
        "amr_graph": [],
        "amr_n_nodes": [],
        "amr_n_edges": [],
    }
    doc_batched = nlp.pipe(examples["sentences_raw"])

    for doc in doc_batched:
        # Also derive the AMR graph for the caption and derive its depth
        amr_graph = doc._.to_amr()[0]  # type: ignore
        try:
            amr_depth, amr_graph_obj = get_amr_graph_depth(amr_graph, return_graph=True)  # type: ignore
            n_nodes = nx.number_of_nodes(amr_graph_obj)
            n_edges = nx.number_of_edges(amr_graph_obj)
            amr_graph_arr = nx.to_numpy_array(amr_graph_obj)
        except DecodeError:
            amr_depth = 0
            n_nodes = 0
            n_edges = 0
            amr_graph_arr = nx.to_numpy_array(nx.DiGraph())
        # Determine the depth of the dependency parse tree
        result["parse_tree_depth"].append(walk_tree(next(doc.sents).root, 0))
        result["n_verbs"].append(len([token for token in doc if token.pos_ == "VERB"]))
        result["amr_graph_depth"].append(amr_depth)
        result["amr_graph"].append(amr_graph_arr)
        result["amr_n_nodes"].append(n_nodes)
        result["amr_n_edges"].append(n_edges)

    return examples | result


def dependency_parse_to_nx(
    sents: List[Doc],
):
    """Convert spaCy sentence objects into a NetworkX directed graph representing the dependency parse tree.

    :param sents: A list of spaCy sentence objects
    :type sents: List[spacy.tokens.Doc]
    :return: A NetworkX directed graph representing the dependency parse tree
    :rtype: nx.DiGraph
    """
    # Create a directed graph
    graph = nx.DiGraph()

    # Iterate over sentences
    for sent in sents:
        for token in sent:
            # Add node for the token with attributes
            graph.add_node(token.i, text=token.text, pos=token.pos_, tag=token.tag_)
            # Add edge from head to child (if not the root token)
            if token.head != token:  # Avoid self-loop for the root
                graph.add_edge(token.head.i, token.i, dep=token.dep_)

    return graph


def flatten_examples(
    examples: Dict[str, List],
    flatten_col_names: List[str] = ["sentences_raw", "sentids"],
) -> Dict[str, List]:
    """Flattens the examples in the dataset.

    :param examples: The examples to flatten
    :type examples: Dict[str, List]
    :param flatten_col_names: The column names to flatten, defaults to ["sentences_raw", "sentids"]
    :type flatten_col_names: List[str]
    :return: The flattened examples
    :rtype: Dict[str, List]
    """
    flattened_data = {}
    number_of_sentences = [len(sents) for sents in examples[flatten_col_names[0]]]
    for key, value in examples.items():
        if key in flatten_col_names:
            flattened_data[key] = [sent for sent_list in value for sent in sent_list]
        else:
            flattened_data[key] = np.repeat(value, number_of_sentences).tolist()
    return flattened_data


def apply_gamma_correction(
    image: Image.Image,
    target_mean=128.0,
) -> Image.Image:
    """Apply gamma correction to an image.

    :param image: The image to apply gamma correction to
    :type image: PIL.Image.Image
    :param target_mean: The target mean brightness of the image, defaults to 128.0
    :type target_mean: float, optional
    :return: The image with gamma correction applied
    :rtype: PIL.Image.Image
    """
    # Convert the PIL image to a numpy array
    img_array = np.array(image)

    # Calculate the current mean brightness of the image
    current_mean = np.mean(img_array)

    # Calculate the gamma value to adjust the mean to the target mean
    # Avoid division by zero
    if current_mean > 0:
        gamma = np.log(target_mean) / np.log(current_mean)
        # Apply gamma correction to the image
        corrected_image = ImageOps.autocontrast(image, cutoff=gamma) # type: ignore
        return corrected_image
    else:
        return image


def save_brain_map(
    img: Union[nib.nifti1.Nifti1Image, object],
    output_path: Union[str, Path],
    is_z_map: bool = False,
    voxel_p: float = DEFAULT_VOXEL_P,
    cluster_threshold: int = DEFAULT_CLUSTER_THRESHOLD,
    make_surface_plot: bool = True,
    surface_plot_kwargs: Optional[Dict] = None,
    sided: str = "positive",
) -> Optional[nib.nifti1.Nifti1Image]:
    """Save a brain map to disk, generate mosaic plots, and cluster tables.

    For z-score/stat maps (``is_z_map=True``), apply one-sided FPR thresholding
    with ``threshold_stats_img`` (using ``voxel_p`` and ``cluster_threshold``).
    Set ``sided="positive"`` to keep only positive effects, ``sided="negative"``
    for negative effects, or ``sided="both"`` to save both directions (files
    suffixed with ``_pos_thr`` and ``_neg_thr``).

    :param img: The brain map image
    :param output_path: Where to save the image
    :param is_z_map: Whether the image is a z-/stat-map that should be thresholded
    :param voxel_p: Two-sided voxel-wise p-value used for thresholding
    :param make_surface_plot: Whether to save a surface plot
    :param surface_plot_kwargs: Extra args for ``generate_surface_plots``
    :param sided: Direction for one-sided thresholding (positive, negative, both)
    :returns: Thresholded positive image if available, otherwise None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _base_path(path: Path) -> str:
        str_path = str(path)
        if str_path.endswith(".nii.gz"):
            return str_path[:-7]
        if str_path.endswith(".nii"):
            return str_path[:-4]
        return str(path.with_suffix(""))

    def _plot(target_img: object, target_path: Path, cmap: Optional[str] = None) -> None:
        base = _base_path(target_path)
        try:
            display = plotting.plot_stat_map(
                target_img,
                display_mode="mosaic",
                cut_coords=5,
                cmap=cmap,  # type: ignore
            )
            display.savefig(f"{base}_mosaic.png")  # type: ignore
            display.close()  # type: ignore
        except Exception as e:
            logger.error(f"Failed to plot brain map: {e}")
    
    # Save the Nifti image
    if hasattr(img, "to_filename"):
        img.to_filename(output_path) # type: ignore
    else:
        nib.save(img, output_path) # type: ignore

    _plot(img, output_path)

    thresholded_img = None
    if is_z_map:
        try:
            do_pos = sided in ("positive", "both")
            do_neg = sided in ("negative", "both")

            def _save_thr(target_img: object, suffix: str, flip_back: bool = False, cmap: Optional[str] = None) -> object:
                thr_img, _ = threshold_stats_img(
                    target_img,
                    alpha=voxel_p,
                    height_control="fpr",
                    cluster_threshold=cluster_threshold,
                    two_sided=False,
                )
                if flip_back:
                    thr_img = image.math_img("-img", img=thr_img)
                if str(output_path).endswith(".nii.gz"):
                    thr_path = output_path.with_name(output_path.name.replace(".nii.gz", f"_{suffix}.nii.gz"))
                else:
                    thr_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
                thr_img.to_filename(thr_path)  # type: ignore
                _plot(thr_img, thr_path, cmap=cmap)
                return thr_img

            if do_pos:
                thresholded_img = _save_thr(img, "pos_thr", cmap="Reds")
            if do_neg:
                _save_thr(image.math_img("-img", img=img), "neg_thr", flip_back=True, cmap="Blues_r")
        except Exception as e:
            logger.error(f"Failed to threshold brain map: {e}")

    # Save surface plot
    if make_surface_plot:
        from compositionality_study.visualization.visualize_brain_maps import generate_surface_plots
        surface_plot_kwargs = surface_plot_kwargs or {}
        try:
            generate_surface_plots(
                nii_file=output_path, 
                output_dir=output_path.parent, 
                **surface_plot_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to generate surface plot: {e}")

    return thresholded_img  # type: ignore
