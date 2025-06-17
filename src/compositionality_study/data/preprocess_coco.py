"""Preprocesses the Visual Genome + COCO dataset."""

# Imports
import json
import os
from typing import Dict, List, Optional, Union

import click
import networkx as nx
import pandas as pd
import spacy
from datasets import Dataset, load_from_disk
from loguru import logger
from spacy import Language
from tqdm import tqdm

from compositionality_study.constants import (
    COCO_A_ANNOT_FILE,
    COCO_DIR,
    COCO_IMAGE_DIR,
    COCO_OBJ_SEG_DIR,
    COCO_PREP_TEXT_GRAPH_DIR,
    IC_SCORES_FILE,
)
from compositionality_study.utils import (
    derive_text_depth_features,
    flatten_examples,
    get_aspect_ratio,
)


@click.command()
@click.option("--coco_dir", type=str, default=COCO_DIR)
@click.option("--spacy_model", type=str, default="en_core_web_trf")
@click.option("--save_to_disk", type=bool, default=True)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
def add_text_properties_wrapper(
    coco_dir: str = COCO_DIR,
    spacy_model: str = "en_core_web_trf",
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
):
    """Wrapper for the add_text_properties function.

    :param coco_dir: Path to the directory where the Visual Genome + COCO dataset is stored.
    :type coco_dir: str
    :param spacy_model: Which spacy model to use, defaults to "en_core_web_trf" (transformer model)
    :type spacy_model: str
    :param save_to_disk: Whether to save the dataset to disk, defaults to True
    :type save_to_disk: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    """
    add_text_properties(
        coco=coco_dir,
        spacy_model=spacy_model,
        save_to_disk=save_to_disk,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )


def add_text_properties(
    coco: Union[str, Dataset] = COCO_DIR,
    spacy_model: str = "en_core_web_trf",
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
) -> Dataset:
    """Add text properties to the dataset.

    :param coco: Path to the directory where the Visual Genome + COCO dataset is stored or the
        dataset itself, defaults to COCO_DIR
    :type coco: Union[str, Dataset]
    :param spacy_model: Which spacy model to use, defaults to "en_core_web_trf" (transformer model!)
    :type spacy_model: str
    :param save_to_disk: Whether to save the dataset to disk, defaults to True
    :type save_to_disk: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    :return: The dataset with the text properties added
    :rtype: Dataset
    """
    coco_ds = load_from_disk(coco) if isinstance(coco, str) else coco

    # Flatten the dataset so that there is one caption per example
    # Check if any of the features are lists
    if any([isinstance(coco_ds[0][feature], List) for feature in coco_ds.features]):
        vs_coco_ds_flattened = coco_ds.map(flatten_examples, batched=True, num_proc=24)
    else:
        vs_coco_ds_flattened = coco_ds

    # Add sentence length
    preprocessed_ds = vs_coco_ds_flattened.map(
        lambda example: example | {"sentence_length": len(example["sentences_raw"].split())},
        num_proc=24,
    )

    @Language.component("force_single_sentence")
    def one_sentence_per_doc(
        doc: spacy.tokens.Doc,  # noqa
    ) -> spacy.tokens.Doc:  # noqa
        """Force the document to be one sentence.

        :param doc: The document to force to be one sentence
        :type doc: spacy.tokens.Doc
        :return: The document with one sentence
        :rtype: spacy.tokens.Doc
        """
        doc[0].sent_start = True
        for i in range(1, len(doc)):
            doc[i].sent_start = False
        return doc

    # Add dependency parse tree depth and AMR depth
    # Disable unnecessary components
    nlp = spacy.load(
        spacy_model,
        disable=["tok2vec", "attribute_ruler", "lemmatizer"],
    )
    nlp.add_pipe("force_single_sentence", before="parser")
    # This cannot be parallelized because of the spacy model/GPU issues
    preprocessed_ds = preprocessed_ds.map(
        derive_text_depth_features,
        fn_kwargs={"nlp": nlp},
        num_proc=1,
        batched=True,
        batch_size=16,
    )

    # Save to disk
    if save_to_disk:
        output_dir = os.path.split(coco)[0]
        preprocessed_ds.save_to_disk(os.path.join(output_dir, "coco_preprocessed_text"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            preprocessed_ds.select(list(range(dummy_subset_size))).save_to_disk(
                os.path.join(output_dir, "coco_preprocessed_text_dummy")
            )

    return preprocessed_ds


@click.command()
@click.option("--coco_graph_dir", type=str, default=COCO_PREP_TEXT_GRAPH_DIR)
@click.option("--coco_image_dir", type=str, default=COCO_IMAGE_DIR)
@click.option("--coco_obj_seg_dir", type=str, default=COCO_OBJ_SEG_DIR)
@click.option("--coco_a_annot_file", type=str, default=COCO_A_ANNOT_FILE)
@click.option("--save_to_disk", type=bool, default=True)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
@click.option("--output_dir", type=str, default=os.path.split(COCO_DIR)[0])
def add_graph_properties_wrapper(
    coco_graph_dir: str = COCO_PREP_TEXT_GRAPH_DIR,
    coco_image_dir: str = COCO_IMAGE_DIR,
    coco_obj_seg_dir: str = COCO_OBJ_SEG_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(COCO_DIR)[0],
):
    """Wrapper for the add_graph_properties function.

    :param coco_graph_dir: Path to the directory where the Visual Genome + COCO dataset with
        graph properties is stored.
    :type coco_graph_dir: str
    :param coco_image_dir: Directory where the COCO images are stored, defaults to COCO_IMAGE_DIR
    :type coco_image_dir: str
    :param coco_obj_seg_dir: Directory where the COCO object segmentations are stored (instances_train/val2017.json).
    :type coco_obj_seg_dir: str
    :param coco_a_annot_file: Path to the COCO action annotations file, defaults to COCO_A_ANNOT_FILE
    :type coco_a_annot_file: str
    :param save_to_disk: Whether to save the dataset to disk, defaults to True
    :type save_to_disk: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    :param output_dir: Directory where the output dataset should be stored, defaults to
        os.path.split(COCO_DIR)[0]
    :type output_dir: str
    """
    add_graph_properties(
        coco_graph=coco_graph_dir,
        coco_image_dir=coco_image_dir,
        coco_obj_seg_dir=coco_obj_seg_dir,
        coco_a_annot_file=coco_a_annot_file,
        save_to_disk=save_to_disk,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
        output_dir=output_dir,
    )


def add_graph_properties(
    coco_graph: Union[str, Dataset] = COCO_PREP_TEXT_GRAPH_DIR,
    coco_image_dir: str = COCO_IMAGE_DIR,
    coco_obj_seg_dir: str = COCO_OBJ_SEG_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(COCO_DIR)[0],
) -> Dataset:
    """Add COCO action-based graph properties to the dataset.

    :param coco_graph: Path to the directory where the Visual Genome +
        COCO dataset with text and graph properties is stored or the dataset itself.
    :type coco_graph: Union[str, Dataset]
    :param coco_image_dir: Directory where the COCO images are stored, defaults to COCO_IMAGE_DIR
    :type coco_image_dir: str
    :param coco_obj_seg_dir: Directory where the COCO object segmentations are stored (instances_train/val2017.json).
    :type coco_obj_seg_dir: str
    :param coco_a_annot_file: Path to the COCO action annotations file, defaults to COCO_A_ANNOT_FILE
    :type coco_a_annot_file: str
    :param save_to_disk: Whether to save the dataset to disk, defaults to True
    :type save_to_disk: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    :param output_dir: Directory where the dataset should be saved
    :type output_dir: str
    """
    # Load the dataset
    preprocessed_ds = load_from_disk(coco_graph) if isinstance(coco_graph, str) else coco_graph
    preprocessed_df = preprocessed_ds.to_pandas()

    # Get the number of objects based on the COCO annotations for image segmentation
    coco_obj_seg_df = get_coco_obj_seg_df(
        coco_obj_seg_dir,
        coco_a_annot_file,
        coco_ids=list(preprocessed_df["cocoid"].unique()),
    )

    # Speed up the merge by setting the ids as index
    preprocessed_df = preprocessed_df.set_index("cocoid")
    # Merge the two datasets
    joined_df = preprocessed_df.join(coco_obj_seg_df, how="inner")
    # Convert 2D numpy array adjacency matrix to a list of lists for coco_a_graph and amr_graph
    joined_df["coco_a_graph"] = joined_df["coco_a_graph"].apply(lambda x: [y.tolist() for y in x])
    joined_df["amr_graph"] = joined_df["amr_graph"].apply(lambda x: [y.tolist() for y in x])
    graph_ds = Dataset.from_pandas(joined_df)
    # Rename the index column back to cocoid
    if "__index_level_0__" in graph_ds.column_names:
        graph_ds = graph_ds.rename_column("__index_level_0__", "cocoid")

    # For each image, get the aspect ratio by loading the image from the local directory with
    # the "filepath" column
    graph_ds = graph_ds.map(
        lambda example: example | {"aspect_ratio": get_aspect_ratio(os.path.join(coco_image_dir, example["filepath"]))},
        num_proc=24,
    )

    # Save to disk
    if save_to_disk:
        graph_ds.save_to_disk(os.path.join(output_dir, "coco_preprocessed_text_image_graphs"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            graph_ds.select(list(range(dummy_subset_size))).save_to_disk(
                os.path.join(output_dir, "coco_preprocessed_text_image_graphs_dummy")
            )

    return graph_ds


@click.command()
@click.option("--coco_graph_dir", type=str, default=COCO_PREP_TEXT_GRAPH_DIR)
@click.option("--ic_scores_file", type=str, default=IC_SCORES_FILE)
@click.option("--save_to_disk", type=bool, default=True)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
@click.option("--output_dir", type=str, default=os.path.split(COCO_DIR)[0])
def add_ic_scores_wrapper(
    coco_graph_dir: str = COCO_PREP_TEXT_GRAPH_DIR,
    ic_scores_file: str = IC_SCORES_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(COCO_PREP_TEXT_GRAPH_DIR)[0],
):
    """Add image complexity scores to the dataset.

    :param coco_graph_dir: Path to the directory where the Visual Genome +
        COCO dataset with text and image segmentation properties is stored.
    :type coco_graph_dir: str
    :param ic_scores_file: Path to the file where the image complexity scores are stored.
    :type ic_scores_file: str
    :param save_to_disk: Whether to save the dataset to disk, defaults to True
    :type save_to_disk: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    :param output_dir: Directory where the dataset should be saved
    :type output_dir: str
    """
    add_ic_scores(
        coco_graph=coco_graph_dir,
        ic_scores_file=ic_scores_file,
        save_to_disk=save_to_disk,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
        output_dir=output_dir,
    )


def add_ic_scores(
    coco_graph: Union[str, Dataset] = COCO_PREP_TEXT_GRAPH_DIR,
    ic_scores_file: str = IC_SCORES_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(COCO_PREP_TEXT_GRAPH_DIR)[0],
) -> Dataset:
    """Add image complexity scores to the dataset.

    :param coco_graph: Path to the directory where the Visual Genome +
        COCO dataset with text and image segmentation properties is stored or the dataset itself.
    :type coco_graph: Union[str, Dataset]
    :param ic_scores_file: Path to the file where the image complexity scores are stored.
    :type ic_scores_file: str
    :param save_to_disk: Whether to save the dataset to disk, defaults to True
    :type save_to_disk: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    :param output_dir: Directory where the dataset should be saved
    :type output_dir: str
    :return: The dataset with the image complexity scores added
    :rtype: Dataset
    """
    # Load the dataset
    preprocessed_ds = load_from_disk(coco_graph) if isinstance(coco_graph, str) else coco_graph
    preprocessed_df = preprocessed_ds.to_pandas()

    # Load the image complexity scores, give it a column name
    ic_scores_df = pd.read_json(ic_scores_file, orient="index")
    ic_scores_df.columns = ["ic_score"]
    # In case that the index contains "COCO": Only take the last numbers of the file name without leading zeros
    ic_scores_df.index = ic_scores_df.index.map(lambda x: int(x.split("_")[-1]) if type(x) == str else x)  # noqa
    # Convert the index to int
    ic_scores_df.index = ic_scores_df.index.astype(int)
    # Rename the index column to cocoid
    ic_scores_df.index.name = "cocoid"
    # Merge the two datasets based on the cocoid
    joined_df = preprocessed_df.merge(ic_scores_df, on="cocoid")
    graph_ds = Dataset.from_pandas(joined_df)

    # Save to disk
    if save_to_disk:
        graph_ds.save_to_disk(os.path.join(output_dir, "coco_preprocessed_text_graph_ic"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            graph_ds.select(list(range(dummy_subset_size))).save_to_disk(
                os.path.join(output_dir, "coco_preprocessed_text_graph_ic_dummy")
            )

    return graph_ds


def create_coco_a_sub_graph(
    data: Dict,
) -> nx.DiGraph:
    """Create a directed graph from the COCO-A annotations.

    :param data: A single COCO-A annotation entry
    :type data: Dict
    :return: The directed graph
    :rtype: nx.DiGraph
    """
    # Initialize a directed graph
    g = nx.DiGraph()

    # Create a one-relation graph
    object_id = data["object_id"]
    subject_id = data["subject_id"]

    # Handle the special case of object_id: -1 (treat as a unique node each time)
    if object_id == -1:
        # Create a unique node for each object with object_id -1
        object_node = str(data["id"])
    else:
        object_node = str(object_id)

    subject_node = str(subject_id)

    # Add nodes and edges to the graph
    g.add_node(subject_node)
    g.add_node(object_node)
    g.add_edge(subject_node, object_node)

    return g


def get_coco_a_graphs(coco_a_data: List[Dict], coco_a_ids: List[str]) -> Dict[str, nx.DiGraph]:
    """Create a dictionary with the graphs from the COCO-action annotations.

    :param coco_a_data: List of COCO-A annotations
    :type coco_a_data: List[Dict]
    :param coco_a_ids: List of COCO-A image IDs
    :type coco_a_ids: List[str]
    :return: Dictionary with the graphs for each image ID
    :rtype: Dict[str, nx.DiGraph]
    """
    # Create a dictionary with the number of actions/graph depth from the COCO action annotations
    coco_a_filtered = {k: {"n_coco_a_actions": 0, "coco_a_graph": None} for k in set(coco_a_ids)}

    # Get all the COCO-A features
    for coco_a_entry in tqdm(
        coco_a_data,
        desc="Adding COCO-A features",
        total=len(coco_a_data),
    ):
        coco_a_filtered[coco_a_entry["image_id"]]["n_coco_a_actions"] += len(coco_a_entry["visual_actions"])

        if coco_a_filtered[coco_a_entry["image_id"]]["coco_a_graph"]:
            # If there is already a graph, add the entry to the existing graph
            coco_a_filtered[coco_a_entry["image_id"]]["coco_a_graph"].add_edges_from(
                create_coco_a_sub_graph(coco_a_entry).edges()
            )
        else:
            # If there is no graph data for that image_id yet, create a new graph
            coco_a_filtered[coco_a_entry["image_id"]]["coco_a_graph"] = create_coco_a_sub_graph(coco_a_entry)

    return coco_a_filtered


def get_coco_obj_seg_df(
    coco_obj_seg_dir: str = COCO_OBJ_SEG_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    coco_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Get the number of objects, object categories based on the COCO annotations for image segmentation.

    :param coco_obj_seg_dir: Directory where the COCO object segmentations are stored (instances_train/val2017.json).
    :type coco_obj_seg_dir: str
    :param coco_a_annot_file: Path to the file where the COCO-A action annotations are stored.
    :type coco_a_annot_file: str
    :param coco_ids: List of COCO ids to get the number of objects for, defaults to None
    :type coco_ids: Optional[List[str]]
    :return: Dataframe with the number of objects per coco id
    :rtype: pd.DataFrame
    """
    # Load the COCO annotations with the standard json dataset loader because load_dataset does not work
    with open(os.path.join(coco_obj_seg_dir, "instances_train2017.json"), "r") as f:
        coco_data_tr = json.load(f)
        coco_obj_seg_tr = [(ex["image_id"], ex["category_id"]) for ex in coco_data_tr["annotations"]]
        coco_cats_tr = {ex["id"]: ex["supercategory"] for ex in coco_data_tr["categories"]}
    with open(os.path.join(coco_obj_seg_dir, "instances_val2017.json"), "r") as f:
        coco_data_val = json.load(f)
        coco_obj_seg_val = [(ex["image_id"], ex["category_id"]) for ex in coco_data_val["annotations"]]
        coco_cats_val = {ex["id"]: ex["supercategory"] for ex in coco_data_val["categories"]}
    with open(coco_a_annot_file, "r") as f:
        # Load the version of the dataset with an annotator agreement of 3 persons
        coco_a_data = json.load(f)["annotations"]["3"]
        coco_a_ids = [ex["image_id"] for ex in coco_a_data]

    # Combine the two lists into a hf dataset
    coco_obj_seg = [i[0] for i in coco_obj_seg_tr] + [i[0] for i in coco_obj_seg_val]
    coco_cat = [i[1] for i in coco_obj_seg_tr] + [i[1] for i in coco_obj_seg_val]
    # Combine all the categories
    coco_cat_mappings = {**coco_cats_tr, **coco_cats_val}

    # Create a dictionary counting the number of objects per image id as well as a list of the categories
    coco_obj_seg_filtered = {k: {"n_graph_obj": 0, "coco_person": 0, "coco_categories": []} for k in set(coco_obj_seg)}

    # Get all the COCO features
    for coco_id, coco_cat_id in tqdm(
        zip(coco_obj_seg, coco_cat),
        desc="Adding COCO features",
        total=len(coco_cat),
    ):
        coco_obj_seg_filtered[coco_id]["n_graph_obj"] += 1  # type: ignore
        coco_obj_seg_filtered[coco_id]["coco_categories"].append(coco_cat_mappings[coco_cat_id])  # type: ignore
        # Check if the category is a person
        if coco_cat_mappings[coco_cat_id] in ["person"]:
            coco_obj_seg_filtered[coco_id]["coco_person"] += 1  # type: ignore

    # Create a dictionary with the number of actions/graph depth from the COCO action annotations
    coco_a_filtered = get_coco_a_graphs(coco_a_data, coco_a_ids)

    # Determine the depths of the graphs
    for coco_a_id, coco_a_graph in coco_a_filtered.items():
        max_longest_shortest_path = 0
        # Iterate over all connected components
        for comp_nodes in nx.weakly_connected_components(coco_a_graph["coco_a_graph"]):
            # Get the longest shortest path for each connected component
            subgraph = coco_a_graph["coco_a_graph"].subgraph(comp_nodes)  # type: ignore
            shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
            longest_path = max(max(shortest_path_lengths[source].values()) for source in comp_nodes)  # type: ignore
            max_longest_shortest_path = max(max_longest_shortest_path, longest_path)

        # Add the longest shortest path to the image id
        coco_a_filtered[coco_a_id]["coco_a_graph_depth"] = max_longest_shortest_path
        # Also add the number of edges, nodes and the entire graph
        coco_a_filtered[coco_a_id]["coco_a_edges"] = coco_a_graph["coco_a_graph"].number_of_edges()
        coco_a_filtered[coco_a_id]["coco_a_nodes"] = coco_a_graph["coco_a_graph"].number_of_nodes()
        coco_a_filtered[coco_a_id]["coco_a_graph"] = nx.to_numpy_array(coco_a_graph["coco_a_graph"])

    # Create dataframes from the dictionaries
    coco_obj_seg_df = pd.DataFrame.from_dict(
        coco_obj_seg_filtered,
        orient="index",
        columns=["n_graph_obj", "coco_person", "coco_categories"],
    )
    coco_a_df = pd.DataFrame.from_dict(
        coco_a_filtered,
        orient="index",
        columns=["n_coco_a_actions", "coco_a_graph_depth", "coco_a_edges", "coco_a_nodes", "coco_a_graph"],
    )
    # Merge the two dataframes (rows where there is data for both)
    coco_obj_seg_df = coco_obj_seg_df.join(coco_a_df, how="inner")

    # Select by coco_image_ids
    if coco_ids:
        coco_obj_seg_df = coco_obj_seg_df.loc[coco_obj_seg_df.index.isin(coco_ids)]

    return coco_obj_seg_df


@click.command()
@click.option("--coco_dir", default=COCO_DIR)
@click.option("--coco_image_dir", default=COCO_IMAGE_DIR)
@click.option("--spacy_model", default="en_core_web_trf")
@click.option("--coco_obj_seg_dir", default=COCO_OBJ_SEG_DIR)
@click.option("--coco_a_annot_file", default=COCO_A_ANNOT_FILE)
@click.option("--ic_scores_file", default=IC_SCORES_FILE)
@click.option("--save_intermediate_steps", default=False)
@click.option("--save_dummy_subset", default=True)
@click.option("--dummy_subset_size", default=1000)
@click.option("--output_dir", default=COCO_DIR)
def add_all_properties(
    coco_dir: str = COCO_DIR,
    coco_image_dir: str = COCO_IMAGE_DIR,
    spacy_model: str = "en_core_web_trf",
    coco_obj_seg_dir: str = COCO_OBJ_SEG_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    ic_scores_file: str = IC_SCORES_FILE,
    save_intermediate_steps: bool = False,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = COCO_DIR,
):
    """Add all properties to the COCO dataset.

    :param coco_dir: Directory where the COCO dataset is stored, defaults to COCO_DIR
    :type coco_dir: str
    :param coco_image_dir: Directory where the COCO images are stored, defaults to COCO_IMAGE_DIR
    :type coco_image_dir: str
    :param spacy_model: SpaCy model to use for text processing, defaults to "en_core_web_trf"
    :type spacy_model: str
    :param coco_obj_seg_dir: Directory where the COCO object segmentations are stored (instances_train/val2017.json),
        defaults to COCO_OBJ_SEG_DIR
    :type coco_obj_seg_dir: str
    :param coco_a_annot_file: File with the COCO action annotations, defaults to COCO_A_ANNOT_FILE
    :type coco_a_annot_file: str
    :param ic_scores_file: File with the IC scores, defaults to IC_SCORES_FILE
    :type ic_scores_file: str
    :param save_intermediate_steps: Whether to save the intermediate steps, defaults to False
    :type save_intermediate_steps: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    :param output_dir: Directory where the COCO dataset with all properties is stored, defaults to COCO_DIR
    :type output_dir: str
    """
    # 1. Add text properties
    text_ds = add_text_properties(
        coco=coco_dir,
        spacy_model=spacy_model,
        save_to_disk=save_intermediate_steps,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )
    logger.info(f"Added text properties for {len(text_ds)} entries")
    # 2. Add COCO graph, image segmentation + category properties
    image_segmentation_ds = add_graph_properties(
        coco_graph=text_ds,
        coco_image_dir=coco_image_dir,
        coco_obj_seg_dir=coco_obj_seg_dir,
        coco_a_annot_file=coco_a_annot_file,
        save_to_disk=save_intermediate_steps,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )
    logger.info(f"Added COCO-A properties for {len(text_ds)} entries")
    # 3. Add IC scores
    ic_scores_ds = add_ic_scores(
        coco_graph=image_segmentation_ds,
        ic_scores_file=ic_scores_file,
        save_to_disk=save_intermediate_steps,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )
    logger.info(f"Added image complexity scores for {len(text_ds)} entries")
    # Save the dataset
    ic_scores_ds.save_to_disk(os.path.join(output_dir, "coco_a_preprocessed_all"))
    # Save the dummy subset
    if save_dummy_subset:
        ic_scores_ds.select(range(dummy_subset_size)).save_to_disk(
            os.path.join(output_dir, "coco_a_preprocessed_all_dummy")
        )
    logger.info(f"Saved the preprocessed dataset to {output_dir}")

    return ic_scores_ds


@click.group()
def cli() -> None:
    """Preprocess the COCO dataset, first for text, then for graph properties and image complexity (on top of text)."""


if __name__ == "__main__":
    cli.add_command(add_text_properties_wrapper)
    cli.add_command(add_graph_properties_wrapper)
    cli.add_command(add_ic_scores_wrapper)
    cli.add_command(add_all_properties)
    cli()
