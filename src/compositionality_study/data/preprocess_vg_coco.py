"""Preprocesses the Visual Genome + COCO overlap dataset."""

# Imports
import json
import os
from typing import Any, Dict, List, Optional, Union

import click
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from spacy import Language
from tqdm import tqdm

from compositionality_study.constants import (
    COCO_A_ANNOT_FILE,
    IC_SCORES_FILE,
    VG_COCO_OBJ_SEG_DIR,
    VG_COCO_OVERLAP_DIR,
    VG_COCO_PREP_TEXT_GRAPH_DIR,
    VG_COCO_PREP_TEXT_IMG_SEG_DIR,
    VG_COCO_PREPROCESSED_TEXT_DIR,
    VG_DIR,
    VG_OBJECTS_FILE,
    VG_RELATIONSHIPS_FILE,
)
from compositionality_study.utils import (
    check_if_living_being,
    derive_text_depth_features,
    flatten_examples,
)


@click.command()
@click.option("--vg_coco_overlap_dir", type=str, default=VG_COCO_OVERLAP_DIR)
@click.option("--spacy_model", type=str, default="en_core_web_trf")
@click.option("--save_to_disk", type=bool, default=True)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
def add_text_properties_wrapper(
    vg_coco_overlap_dir: str = VG_COCO_OVERLAP_DIR,
    spacy_model: str = "en_core_web_trf",
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
):
    """Wrapper for the add_text_properties function.

    :param vg_coco_overlap_dir: Path to the directory where the Visual Genome + COCO overlap dataset is stored.
    :type vg_coco_overlap_dir: str
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
        vg_coco_overlap=vg_coco_overlap_dir,
        spacy_model=spacy_model,
        save_to_disk=save_to_disk,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )


def add_text_properties(
    vg_coco_overlap: Union[str, Dataset] = VG_COCO_OVERLAP_DIR,
    spacy_model: str = "en_core_web_trf",
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
) -> Dataset:
    """Add text properties to the dataset.

    :param vg_coco_overlap: Path to the directory where the Visual Genome + COCO overlap dataset is stored or the
        dataset itself, defaults to VG_COCO_OVERLAP_DIR
    :type vg_coco_overlap: Union[str, Dataset]
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
    if isinstance(vg_coco_overlap, str) and "vg_" in vg_coco_overlap:
        vg_in_file_name = "vg_"
    elif isinstance(vg_coco_overlap, Dataset) and "vg_image_id" in vg_coco_overlap.column_names:
        vg_in_file_name = "vg_"
    else:
        vg_in_file_name = ""
    vg_coco_ds = load_from_disk(vg_coco_overlap) if isinstance(vg_coco_overlap, str) else vg_coco_overlap

    # Flatten the dataset so that there is one caption per example
    # Check if any of the features are lists
    if any([isinstance(vg_coco_ds[0][feature], List) for feature in vg_coco_ds.features]):
        vs_coco_ds_flattened = vg_coco_ds.map(flatten_examples, batched=True, num_proc=24)
    else:
        vs_coco_ds_flattened = vg_coco_ds

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
    # Prefer GPU if available
    spacy.prefer_gpu()
    # Disable unnecessary components
    nlp = spacy.load(spacy_model, disable=["tok2vec", "attribute_ruler", "lemmatizer"])
    nlp.add_pipe("force_single_sentence", before="parser")
    # This cannot be parallelized because of the spacy model/GPU issues
    preprocessed_ds = preprocessed_ds.map(
        derive_text_depth_features,
        fn_kwargs={"nlp": nlp},
        num_proc=1,
        batched=True,
        batch_size=64,
    )

    # Save to disk
    if save_to_disk:
        output_dir = os.path.split(vg_coco_overlap)[0]
        preprocessed_ds.save_to_disk(os.path.join(output_dir, f"{vg_in_file_name}coco_preprocessed_text"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            preprocessed_ds.select(list(range(dummy_subset_size))).save_to_disk(
                os.path.join(output_dir, f"{vg_in_file_name}coco_preprocessed_text_dummy")
            )

    return preprocessed_ds


@click.command()
@click.option("--vg_coco_overlap_text_dir", type=str, default=VG_COCO_PREPROCESSED_TEXT_DIR)
@click.option("--vg_objects_file", type=str, default=VG_OBJECTS_FILE)
@click.option("--vg_relationships_file", type=str, default=VG_RELATIONSHIPS_FILE)
@click.option("--save_to_disk", type=bool, default=True)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
def add_graph_properties_wrapper(
    vg_coco_overlap_text_dir: str = VG_COCO_PREPROCESSED_TEXT_DIR,
    vg_objects_file: str = VG_OBJECTS_FILE,
    vg_relationships_file: str = VG_RELATIONSHIPS_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
):
    """Wrapper for the add_graph_properties function.

    :param vg_coco_overlap_text_dir: Path to the directory where the Visual Genome + COCO overlap dataset with text
        properties is stored.
    :type vg_coco_overlap_text_dir: str
    :param vg_objects_file: Path to the file where the Visual Genome objects json is stored.
    :type vg_objects_file: str
    :param vg_relationships_file: Path to the file where the Visual Genome relationship json is stored.
    :type vg_relationships_file: str
    :param save_to_disk: Whether to save the dataset to disk, defaults to True
    :type save_to_disk: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    """
    add_graph_properties(
        vg_coco_overlap_text=vg_coco_overlap_text_dir,
        vg_objects_file=vg_objects_file,
        vg_relationships_file=vg_relationships_file,
        save_to_disk=save_to_disk,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )


def add_graph_properties(
    vg_coco_overlap_text: Union[str, Dataset] = VG_COCO_PREPROCESSED_TEXT_DIR,
    vg_objects_file: str = VG_OBJECTS_FILE,
    vg_relationships_file: str = VG_RELATIONSHIPS_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
) -> Dataset:
    """Add graph properties to the dataset.

    :param vg_coco_overlap_text: Path to the directory where the Visual Genome +
        COCO overlap dataset with text properties is stored or the dataset itself.
    :type vg_coco_overlap_text: Union[str, Dataset]
    :param vg_objects_file: Path to the file where the Visual Genome objects json is stored.
    :type vg_objects_file: str
    :param vg_relationships_file: Path to the file where the Visual Genome relationship json is stored.
    :type vg_relationships_file: str
    :param save_to_disk: Whether to save the dataset to disk, defaults to True
    :type save_to_disk: bool
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    :return: The dataset with the graph properties added
    :rtype: Dataset
    """
    # Load the dataset
    preprocessed_ds = (
        load_from_disk(vg_coco_overlap_text) if isinstance(vg_coco_overlap_text, str) else vg_coco_overlap_text
    )
    image_ids = list(set(preprocessed_ds["vg_image_id"]))
    preprocessed_df = preprocessed_ds.to_pandas()

    # Get the graph characteristics
    vg_graph_dict = determine_graph_complexity_measures(vg_objects_file, vg_relationships_file, image_ids)
    vg_graph_df = pd.DataFrame(vg_graph_dict).rename(columns={"image_id": "vg_image_id"})

    # Speed up the merge by setting the ids as index
    preprocessed_df = preprocessed_df.set_index("vg_image_id")
    vg_graph_df = vg_graph_df.set_index("vg_image_id")
    # Merge the two datasets
    vg_graph_merged_ds = Dataset.from_pandas(preprocessed_df.join(vg_graph_df))

    # Save to disk
    if save_to_disk:
        output_dir = os.path.split(vg_coco_overlap_text)[0] if isinstance(vg_coco_overlap_text, str) else VG_DIR
        vg_graph_merged_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_graph"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            vg_graph_merged_ds.select(list(range(dummy_subset_size))).save_to_disk(
                os.path.join(output_dir, "vg_coco_preprocessed_graph_dummy")
            )

    return vg_graph_merged_ds


@click.command()
@click.option("--vg_coco_overlap_graph_dir", type=str, default=VG_COCO_PREP_TEXT_GRAPH_DIR)
@click.option("--coco_obj_seg_dir", type=str, default=VG_COCO_OBJ_SEG_DIR)
@click.option("--coco_a_annot_file", type=str, default=COCO_A_ANNOT_FILE)
@click.option("--save_to_disk", type=bool, default=True)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
@click.option("--output_dir", type=str, default=os.path.split(VG_COCO_OVERLAP_DIR)[0])
def add_coco_properties_wrapper(
    vg_coco_overlap_graph_dir: str = VG_COCO_PREP_TEXT_GRAPH_DIR,
    coco_obj_seg_dir: str = VG_COCO_OBJ_SEG_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(VG_COCO_OVERLAP_DIR)[0],
):
    """Wrapper for the add_coco_properties_properties function.

    :param vg_coco_overlap_graph_dir: Path to the directory where the Visual Genome + COCO overlap dataset with
        graph properties is stored.
    :type vg_coco_overlap_graph_dir: str
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
        os.path.split(VG_COCO_OVERLAP_DIR)[0]
    :type output_dir: str
    """
    add_coco_properties(
        vg_coco_overlap_graph=vg_coco_overlap_graph_dir,
        coco_obj_seg_dir=coco_obj_seg_dir,
        coco_a_annot_file=coco_a_annot_file,
        save_to_disk=save_to_disk,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
        output_dir=output_dir,
    )


def add_coco_properties(
    vg_coco_overlap_graph: Union[str, Dataset] = VG_COCO_PREP_TEXT_GRAPH_DIR,
    coco_obj_seg_dir: str = VG_COCO_OBJ_SEG_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(VG_COCO_OVERLAP_DIR)[0],
) -> Dataset:
    """Add image segmentation properties to the dataset.

    :param vg_coco_overlap_graph: Path to the directory where the Visual Genome +
        COCO overlap dataset with text and graph properties is stored or the dataset itself.
    :type vg_coco_overlap_graph: Union[str, Dataset]
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
    # Set the output name
    if isinstance(vg_coco_overlap_graph, str) and "vg_" in vg_coco_overlap_graph:
        vg_in_file_name = "vg_"
    elif isinstance(vg_coco_overlap_graph, Dataset) and "vg_image_id" in vg_coco_overlap_graph.column_names:
        vg_in_file_name = "vg_"
    else:
        vg_in_file_name = ""

    # Load the dataset
    preprocessed_ds = (
        load_from_disk(vg_coco_overlap_graph) if isinstance(vg_coco_overlap_graph, str) else vg_coco_overlap_graph
    )
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
    vg_img_seg_ds = Dataset.from_pandas(joined_df)
    # Rename the index column back to cocoid
    if "__index_level_0__" in vg_img_seg_ds.column_names:
        vg_img_seg_ds = vg_img_seg_ds.rename_column("__index_level_0__", "cocoid")

    # Save to disk
    if save_to_disk:
        vg_img_seg_ds.save_to_disk(os.path.join(output_dir, f"{vg_in_file_name}coco_preprocessed_img_seg"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            vg_img_seg_ds.select(list(range(dummy_subset_size))).save_to_disk(
                os.path.join(output_dir, f"{vg_in_file_name}coco_preprocessed_img_seg_dummy")
            )

    return vg_img_seg_ds


@click.command()
@click.option("--vg_coco_overlap_graph_dir", type=str, default=VG_COCO_PREP_TEXT_IMG_SEG_DIR)
@click.option("--ic_scores_file", type=str, default=IC_SCORES_FILE)
@click.option("--save_to_disk", type=bool, default=True)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
@click.option("--output_dir", type=str, default=os.path.split(VG_COCO_OVERLAP_DIR)[0])
def add_ic_scores_wrapper(
    vg_coco_overlap_graph_dir: str = VG_COCO_PREP_TEXT_IMG_SEG_DIR,
    ic_scores_file: str = IC_SCORES_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(VG_COCO_PREP_TEXT_IMG_SEG_DIR)[0],
):
    """Add image complexity scores to the dataset.

    :param vg_coco_overlap_graph_dir: Path to the directory where the Visual Genome +
        COCO overlap dataset with text and image segmentation properties is stored.
    :type vg_coco_overlap_graph_dir: str
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
        vg_coco_overlap_graph=vg_coco_overlap_graph_dir,
        ic_scores_file=ic_scores_file,
        save_to_disk=save_to_disk,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
        output_dir=output_dir,
    )


def add_ic_scores(
    vg_coco_overlap_graph: Union[str, Dataset] = VG_COCO_PREP_TEXT_IMG_SEG_DIR,
    ic_scores_file: str = IC_SCORES_FILE,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(VG_COCO_PREP_TEXT_IMG_SEG_DIR)[0],
) -> Dataset:
    """Add image complexity scores to the dataset.

    :param vg_coco_overlap_graph: Path to the directory where the Visual Genome +
        COCO overlap dataset with text and image segmentation properties is stored or the dataset itself.
    :type vg_coco_overlap_graph: Union[str, Dataset]
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
    preprocessed_ds = (
        load_from_disk(vg_coco_overlap_graph) if isinstance(vg_coco_overlap_graph, str) else vg_coco_overlap_graph
    )
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
    joined_df["ic_score"] = joined_df["ic_score_x"].combine_first(joined_df["ic_score_y"])
    joined_df.drop(columns=["ic_score_x", "ic_score_y"], inplace=True)
    # If present, remove the "__index_level_0__", "id" and "split" columns
    if "__index_level_0__" in joined_df.columns:
        joined_df = joined_df.drop(columns=["__index_level_0__"])
    if "id" in joined_df.columns:
        joined_df = joined_df.drop(columns=["id"])
    if "split" in joined_df.columns:
        joined_df = joined_df.drop(columns=["split"])
    vg_img_seg_ds = Dataset.from_pandas(joined_df)

    # Save to disk
    if save_to_disk:
        vg_img_seg_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_img_seg_ic"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            vg_img_seg_ds.select(list(range(dummy_subset_size))).save_to_disk(
                os.path.join(output_dir, "vg_coco_preprocessed_img_seg_ic_dummy")
            )

    return vg_img_seg_ds


def determine_graph_complexity_measures(
    vg_objects_file: str = VG_OBJECTS_FILE,
    vg_relationships_file: str = VG_RELATIONSHIPS_FILE,
    image_ids: Optional[List[str]] = None,
    return_graphs: bool = False,
) -> List[Dict[str, Any]]:
    """Characterize the graph complexity of the VG + COCO overlap dataset for the given image ids.

    :param vg_objects_file: Path to the file where the Visual Genome objects json is stored.
    :type vg_objects_file: str
    :param vg_relationships_file: Path to the file where the Visual Genome relationship json is stored.
    :type vg_relationships_file: str
    :param image_ids: Optional list of image ids to characterize the graph complexity for, defaults to None
    :type image_ids: Optional[List[str]]
    :param return_graphs: Whether to return the graphs as well, defaults to False
    :type return_graphs: bool
    :return: List of dictionaries with the graph complexity measures and image id
    :rtype: List[Dict[str, Any]]
    """
    # Load the object and relationship files from json
    vg_objects = load_dataset("json", data_files=vg_objects_file, split="train")
    vg_relationships = load_dataset("json", data_files=vg_relationships_file, split="train")
    # Filter by image ids if given
    if image_ids:
        vg_objects = vg_objects.filter(lambda x: x["image_id"] in image_ids, num_proc=4)
        vg_relationships = vg_relationships.filter(
            lambda x: x["image_id"] in image_ids,
            num_proc=4,
        )

    # Process each VG image/graph into a networkx graph
    graph_measures = []
    graphs = {}
    for obj, rel in tqdm(
        zip(vg_objects, vg_relationships),
        desc="Processing rels/objs as networkx graphs",
        total=len(vg_objects),
    ):
        # Create the graph based on objects and relationships
        graph = nx.Graph()
        for o in obj["objects"]:
            graph.add_node(o["object_id"])
        for r in rel["relationships"]:
            graph.add_edge(
                r["object"]["object_id"],
                r["subject"]["object_id"],
                rel_id=r["relationship_id"],
            )

        # Append the graph to the dict
        graphs[obj["image_id"]] = graph
        # Filter for relationships that have at least one living being as subject/object
        filtered_rels = [
            r
            for r in rel["relationships"]
            if len(r["object"]["synsets"]) > 0
            and len(r["subject"]["synsets"]) > 0
            and (check_if_living_being(r["object"]["synsets"][0]) or check_if_living_being(r["subject"]["synsets"][0]))
        ]
        filtered_rel_ids = [r["relationship_id"] for r in filtered_rels]
        filtered_edges = [
            (u, v, data) for u, v, data in graph.edges(data=True) if data.get("rel_id") in filtered_rel_ids
        ]
        # Create a new graph with the filtered edges
        filtered_graph = nx.Graph(filtered_edges)

        # Determine the characteristics + add the image id as well
        measures = {
            "image_id": obj["image_id"],
            "avg_node_degree": (np.mean([d for _, d in graph.degree()]) if nx.number_of_nodes(graph) > 0 else 0),
            "avg_node_connectivity": (
                nx.algorithms.connectivity.average_node_connectivity(graph) if nx.number_of_nodes(graph) > 0 else 0
            ),
            "avg_clustering_coefficient": (
                nx.algorithms.approximation.average_clustering(
                    graph,
                    trials=1000,
                    seed=42,
                )
                if nx.number_of_nodes(graph) > 0
                else 0
            ),
            "density": nx.density(graph),
            "sg_depth": (
                max([max(nx.shortest_path_length(graph, source=n).values()) for n in graph.nodes()])
                if nx.number_of_nodes(graph) > 0
                else 0
            ),
            "sg_filtered_depth": (
                max([max(nx.shortest_path_length(filtered_graph, source=n).values()) for n in filtered_graph.nodes()])
                if nx.number_of_nodes(filtered_graph) > 0
                else 0
            ),
            "n_connected_components": nx.number_connected_components(graph),
            "n_obj": len(obj["objects"]),
            "n_rel": len(rel["relationships"]),
            "n_filtered_rel": len(filtered_rels),
        }
        # Append them to the list
        graph_measures.append(measures)

    if return_graphs:
        return graphs, graph_measures
    else:
        return graph_measures


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
    coco_obj_seg_dir: str = VG_COCO_OBJ_SEG_DIR,
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
    coco_obj_seg_filtered = {
        k: {"n_img_seg_obj": 0, "coco_person": 0, "coco_categories": []} for k in set(coco_obj_seg)
    }

    # Get all the COCO features
    for coco_id, coco_cat_id in tqdm(
        zip(coco_obj_seg, coco_cat),
        desc="Adding COCO features",
        total=len(coco_cat),
    ):
        coco_obj_seg_filtered[coco_id]["n_img_seg_obj"] += 1  # type: ignore
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
        # Remove the graph from the dictionary
        coco_a_filtered[coco_a_id].pop("coco_a_graph")

    # Create dataframes from the dictionaries
    coco_obj_seg_df = pd.DataFrame.from_dict(
        coco_obj_seg_filtered,
        orient="index",
        columns=["n_img_seg_obj", "coco_person", "coco_categories"],
    )
    coco_a_df = pd.DataFrame.from_dict(
        coco_a_filtered,
        orient="index",
        columns=["n_coco_a_actions", "coco_a_graph_depth"],
    )
    # Merge the two dataframes (rows where there is data for both)
    coco_obj_seg_df = coco_obj_seg_df.join(coco_a_df, how="inner")

    # Select by coco_image_ids
    if coco_ids:
        coco_obj_seg_df = coco_obj_seg_df.loc[coco_obj_seg_df.index.isin(coco_ids)]

    return coco_obj_seg_df


@click.command()
@click.option("--vg_coco_overlap_dir", default=VG_COCO_OVERLAP_DIR)
@click.option("--spacy_model", default="en_core_web_trf")
@click.option("--vg_objects_file", default=VG_OBJECTS_FILE)
@click.option("--vg_relationships_file", default=VG_RELATIONSHIPS_FILE)
@click.option("--coco_obj_seg_dir", default=VG_COCO_OBJ_SEG_DIR)
@click.option("--coco_a_annot_file", default=COCO_A_ANNOT_FILE)
@click.option("--ic_scores_file", default=IC_SCORES_FILE)
@click.option("--save_intermediate_steps", default=False)
@click.option("--save_dummy_subset", default=True)
@click.option("--dummy_subset_size", default=1000)
@click.option("--output_dir", default=VG_DIR)
def add_all_properties(
    vg_coco_overlap_dir: str = VG_COCO_OVERLAP_DIR,
    spacy_model: str = "en_core_web_trf",
    vg_objects_file: str = VG_OBJECTS_FILE,
    vg_relationships_file: str = VG_RELATIONSHIPS_FILE,
    coco_obj_seg_dir: str = VG_COCO_OBJ_SEG_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    ic_scores_file: str = IC_SCORES_FILE,
    save_intermediate_steps: bool = False,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = VG_DIR,
):
    """Add all properties to the VG + COCO overlap dataset.

    :param vg_coco_overlap_dir: Directory where the VG + COCO overlap dataset is stored, defaults to VG_COCO_OVERLAP_DIR
    :type vg_coco_overlap_dir: str
    :param spacy_model: SpaCy model to use for text processing, defaults to "en_core_web_trf"
    :type spacy_model: str
    :param vg_objects_file: File with the VG objects, defaults to VG_OBJECTS_FILE
    :type vg_objects_file: str
    :param vg_relationships_file: File with the VG relationships, defaults to VG_RELATIONSHIPS_FILE
    :type vg_relationships_file: str
    :param coco_obj_seg_dir: Directory where the COCO object segmentations are stored (instances_train/val2017.json),
        defaults to VG_COCO_OBJ_SEG_DIR
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
    :param output_dir: Directory where the VG + COCO overlap dataset with all properties is stored, defaults to VG_DIR
    :type output_dir: str
    """
    # 1. Add text properties
    text_ds = add_text_properties(
        vg_coco_overlap=vg_coco_overlap_dir,
        spacy_model=spacy_model,
        save_to_disk=save_intermediate_steps,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )
    logger.info(f"Added text properties for {len(text_ds)} entries")
    # 2. Add graph properties
    graph_ds = add_graph_properties(
        vg_coco_overlap_text=text_ds,
        vg_objects_file=vg_objects_file,
        vg_relationships_file=vg_relationships_file,
        save_to_disk=save_intermediate_steps,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )
    logger.info(f"Added graph properties for {len(text_ds)} entries")
    # 3. Add COCO image segmentation + category properties
    image_segmentation_ds = add_coco_properties(
        vg_coco_overlap_graph=graph_ds,
        coco_obj_seg_dir=coco_obj_seg_dir,
        coco_a_annot_file=coco_a_annot_file,
        save_to_disk=save_intermediate_steps,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )
    logger.info(f"Added image segmentation and category properties for {len(text_ds)} entries")
    # 4. Add IC scores
    ic_scores_ds = add_ic_scores(
        vg_coco_overlap_graph=image_segmentation_ds,
        ic_scores_file=ic_scores_file,
        save_to_disk=save_intermediate_steps,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )
    logger.info(f"Added image complexity scores for {len(text_ds)} entries")
    # Save the dataset
    ic_scores_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_all"))
    # Save the dummy subset
    if save_dummy_subset:
        ic_scores_ds.select(range(dummy_subset_size)).save_to_disk(
            os.path.join(output_dir, "vg_coco_preprocessed_all_dummy")
        )
    logger.info(f"Saved the preprocessed dataset to {output_dir}")

    return ic_scores_ds


@click.command()
@click.option("--vg_objs_file", type=str, default=VG_OBJECTS_FILE)
@click.option("--vg_rels_file", type=str, default=VG_RELATIONSHIPS_FILE)
def create_searchable_vg_rel_obj_idx(
    vg_objs_file: str = VG_OBJECTS_FILE,
    vg_rels_file: str = VG_RELATIONSHIPS_FILE,
) -> Dict[str, Dict[str, List]]:
    """Create a dictionary that maps the vg_image_id to the indices in the relationships and objects file.

    :param vg_objs_file: Path to the json file with the VG objects
    :type vg_objs_file: str
    :param vg_rels_file: Path to the json file with the VG relationships
    :type vg_rels_file: str
    :return: Dictionary with vg_image_id (str) mapped a Dict containing "rels" and "objs" as keys, and the List of
        indices from the vg_objs_file and vg_rels_file matching the vg_image_id, respectively
    :rtype: Dict[str, Dict[str, List]]
    """
    # Load the files
    output_dir = os.path.dirname(vg_objs_file)
    with open(vg_objs_file, "r") as f:
        vg_objs = json.load(f)
    with open(vg_rels_file, "r") as f:
        vg_rels = json.load(f)
    # Initialize the result
    all_ids = set([x["image_id"] for x in vg_objs]).intersection(set([x["image_id"] for x in vg_rels]))
    res_dict: Dict[str, Dict[str, List]] = {vg_image_id: {"rels": [], "objs": []} for vg_image_id in all_ids}

    # Iterate over all objects
    for i, obj in tqdm(enumerate(vg_objs), desc="Iterating through all objects"):
        res_dict[obj["image_id"]]["objs"].append(i)
    # Iterate over all relationships
    for i, obj in tqdm(enumerate(vg_rels), desc="Iterating through all relationships"):
        res_dict[obj["image_id"]]["rels"].append(i)

    # Save the result
    with open(os.path.join(output_dir, "vg_objs_rels_idx.json"), "w") as f:
        json.dump(res_dict, f)

    return res_dict


@click.group()
def cli() -> None:
    """Preprocess the VG + COCO overlap dataset, first for text, then for graph properties (on top of text)."""


if __name__ == "__main__":
    cli.add_command(add_text_properties_wrapper)
    cli.add_command(add_graph_properties_wrapper)
    cli.add_command(add_coco_properties_wrapper)
    cli.add_command(add_ic_scores_wrapper)
    cli.add_command(add_all_properties)
    cli.add_command(create_searchable_vg_rel_obj_idx)
    cli()
