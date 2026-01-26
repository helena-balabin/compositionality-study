"""Utils for the compositionality project."""

from typing import Dict, List, Optional, Tuple, Union

import amrlib
import networkx as nx
import numpy as np
import pandas as pd
import penman
from datasets import load_dataset
from penman.exceptions import DecodeError
from PIL import Image, ImageOps
from spacy.language import Language
from spacy.tokens import Doc, Token

from compositionality_study.constants import HF_DATASET_NAME

# Set up the spacy amrlib extension
amrlib.setup_spacy_extension()

_COCO_DF = None


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
