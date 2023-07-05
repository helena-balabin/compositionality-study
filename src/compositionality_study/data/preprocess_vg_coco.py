"""Preprocesses the Visual Genome + COCO overlap dataset."""
# Imports
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import click
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from nltk.corpus import wordnet as wn
from spacy import Language
from tqdm import tqdm

from compositionality_study.constants import (
    IC_SCORES_FILE,
    VG_COCO_OBJ_SEG_DIR,
    VG_COCO_OVERLAP_DIR,
    VG_COCO_PREPROCESSED_TEXT_DIR,
    VG_COCO_PREP_TEXT_GRAPH_DIR,
    VG_COCO_PREP_TEXT_IMG_SEG_DIR,
    VG_DIR, VG_OBJECTS_FILE,
    VG_RELATIONSHIPS_FILE,
    WN_EXCLUDED_CATEGORIES,
)
from compositionality_study.utils import flatten_examples, walk_tree_hf_ds


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
    vg_coco_ds = load_from_disk(vg_coco_overlap) if isinstance(vg_coco_overlap, str) else vg_coco_overlap
    # Remove unnnecessary columns
    vg_coco_ds = vg_coco_ds.remove_columns(["sentences_tokens", "sentences_sentid", "__index_level_0__"])

    # Flatten the dataset so that there is one caption per example
    vs_coco_ds_flattened = vg_coco_ds.map(flatten_examples, batched=True, num_proc=4)

    # Add sentence length
    preprocessed_ds = vs_coco_ds_flattened.map(
        lambda example: example | {"sentence_length": len(example["sentences_raw"].split())}, num_proc=4,
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

    # Add dependency parse tree depth
    nlp = spacy.load(spacy_model)
    nlp.add_pipe("force_single_sentence", before="parser")
    preprocessed_ds = preprocessed_ds.map(walk_tree_hf_ds, fn_kwargs={"nlp": nlp}, num_proc=4)

    # Save to disk
    if save_to_disk:
        output_dir = os.path.split(vg_coco_overlap)[0]
        preprocessed_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_text"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            preprocessed_ds.select(
                list(range(dummy_subset_size))
            ).save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_text_dummy"))

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
    preprocessed_ds = load_from_disk(
        vg_coco_overlap_text
    ) if isinstance(vg_coco_overlap_text, str) else vg_coco_overlap_text
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
        output_dir = os.path.split(vg_coco_overlap_text)[0]
        vg_graph_merged_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_graph"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            vg_graph_merged_ds.select(
                list(range(dummy_subset_size))
            ).save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_graph_dummy"))

    return vg_graph_merged_ds


@click.command()
@click.option("--vg_coco_overlap_graph_dir", type=str, default=VG_COCO_PREP_TEXT_GRAPH_DIR)
@click.option("--coco_obj_seg_dir", type=str, default=VG_COCO_OBJ_SEG_DIR)
@click.option("--save_to_disk", type=bool, default=True)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
@click.option("--output_dir", type=str, default=os.path.split(VG_COCO_OVERLAP_DIR)[0])
def add_image_segmentation_properties_wrapper(
    vg_coco_overlap_graph_dir: str = VG_COCO_PREP_TEXT_GRAPH_DIR,
    coco_obj_seg_dir: str = VG_COCO_OBJ_SEG_DIR,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(VG_COCO_OVERLAP_DIR)[0],
):
    """Wrapper for the add_image_segmentation_properties function.

    :param vg_coco_overlap_graph_dir: Path to the directory where the Visual Genome + COCO overlap dataset with
        graph properties is stored.
    :type vg_coco_overlap_graph_dir: str
    :param coco_obj_seg_dir: Directory where the COCO object segmentations are stored (instances_train/val2017.json).
    :type coco_obj_seg_dir: str
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
    add_image_segmentation_properties(
        vg_coco_overlap_graph=vg_coco_overlap_graph_dir,
        coco_obj_seg_dir=coco_obj_seg_dir,
        save_to_disk=save_to_disk,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
        output_dir=output_dir,
    )


def add_image_segmentation_properties(
    vg_coco_overlap_graph: Union[str, Dataset] = VG_COCO_PREP_TEXT_GRAPH_DIR,
    coco_obj_seg_dir: str = VG_COCO_OBJ_SEG_DIR,
    save_to_disk: bool = True,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
    output_dir: str = os.path.split(VG_COCO_OVERLAP_DIR)[0]
) -> Dataset:
    """Add image segmentation properties to the dataset.

    :param vg_coco_overlap_graph: Path to the directory where the Visual Genome +
        COCO overlap dataset with text and graph properties is stored or the dataset itself.
    :type vg_coco_overlap_graph: Union[str, Dataset]
    :param coco_obj_seg_dir: Directory where the COCO object segmentations are stored (instances_train/val2017.json).
    :type coco_obj_seg_dir: str
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
    preprocessed_ds = load_from_disk(
        vg_coco_overlap_graph
    ) if isinstance(vg_coco_overlap_graph, str) else vg_coco_overlap_graph
    preprocessed_df = preprocessed_ds.to_pandas()

    # Get the number of objects based on the COCO annotations for image segmentation
    coco_obj_seg_df = get_coco_obj_seg_df(coco_obj_seg_dir, coco_ids=list(preprocessed_df["cocoid"].unique()))

    # Speed up the merge by setting the ids as index
    preprocessed_df = preprocessed_df.set_index("cocoid")
    # Merge the two datasets
    joined_df = preprocessed_df.join(coco_obj_seg_df)
    vg_img_seg_ds = Dataset.from_pandas(joined_df)
    # Rename the index column back to cocoid
    vg_img_seg_ds = vg_img_seg_ds.rename_column("__index_level_0__", "cocoid")

    # Save to disk
    if save_to_disk:
        vg_img_seg_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_img_seg"))
        # Also save a small dummy subset of dummy_subset_size many entries
        if save_dummy_subset:
            vg_img_seg_ds.select(
                list(range(dummy_subset_size))
            ).save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_img_seg_dummy"))

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
    preprocessed_ds = load_from_disk(vg_coco_overlap_graph) if isinstance(
        vg_coco_overlap_graph, str
    ) else vg_coco_overlap_graph
    preprocessed_df = preprocessed_ds.to_pandas()

    # Load the image complexity scores, give it a column name
    ic_scores_df = pd.read_json(ic_scores_file, orient="index")
    ic_scores_df.columns = ["ic_score"]
    # Rename the index column to cocoid
    ic_scores_df.index.name = "cocoid"
    # Merge the two datasets based on the cocoid
    joined_df = preprocessed_df.join(ic_scores_df, on="cocoid")
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
            vg_img_seg_ds.select(
                list(range(dummy_subset_size))
            ).save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_img_seg_ic_dummy"))

    return vg_img_seg_ds


def determine_graph_complexity_measures(
    vg_objects_file: str = VG_OBJECTS_FILE,
    vg_relationships_file: str = VG_RELATIONSHIPS_FILE,
    image_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Characterize the graph complexity of the VG + COCO overlap dataset for the given image ids.

    :param vg_objects_file: Path to the file where the Visual Genome objects json is stored.
    :type vg_objects_file: str
    :param vg_relationships_file: Path to the file where the Visual Genome relationship json is stored.
    :type vg_relationships_file: str
    :param image_ids: Optional list of image ids to characterize the graph complexity for, defaults to None
    :type image_ids: Optional[List[str]]
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
            graph.add_edge(r["object"]["object_id"], r["subject"]["object_id"])

        # Count the number of (action) verbs in the image by check for predicate synsets that are (action) verbs
        # One might need to install wordnet via nltk for this: nltk.download('wordnet')
        all_verbs = [
            r for r in rel["relationships"] if len(r["synsets"]) > 0 and ".v." in r["synsets"][0]
        ]
        # Exlcude static verbs (e.g. be, have, ...)
        action_verbs = [v for v in all_verbs if wn.synset(v["synsets"][0]).lexname() not in WN_EXCLUDED_CATEGORIES]

        # Determine the characteristics + add the image id as well
        measures = {
            "image_id": obj["image_id"],
            "avg_node_degree": np.mean([d for _, d in graph.degree()]) if nx.number_of_nodes(graph) > 0 else 0,
            "avg_node_connectivity": nx.algorithms.connectivity.average_node_connectivity(
                graph
            ) if nx.number_of_nodes(graph) > 0 else 0,
            "avg_clustering_coefficient": nx.algorithms.approximation.average_clustering(
                graph,
                trials=1000,
                seed=42,
            ) if nx.number_of_nodes(graph) > 0 else 0,
            "density": nx.density(graph),
            "n_connected_components": nx.number_connected_components(graph),
            "n_obj": len(obj["objects"]),
            "n_rel": len(rel["relationships"]),
            "n_rel_verbs": len(all_verbs),
            "n_rel_action_verbs": len(action_verbs),
        }
        # Append them to the list
        graph_measures.append(measures)

    return graph_measures


def get_coco_obj_seg_df(
    coco_obj_seg_dir: str = VG_COCO_OBJ_SEG_DIR,
    coco_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Get the number of objects based on the COCO annotations for image segmentation.

    :param coco_obj_seg_dir: Directory where the COCO object segmentations are stored (instances_train/val2017.json).
    :type coco_obj_seg_dir: str
    :param coco_ids: List of COCO ids to get the number of objects for, defaults to None
    :type coco_ids: Optional[List[str]]
    :return: Dataframe with the number of objects per coco id
    :rtype: pd.DataFrame
    """
    # Load the COCO annotations with the standard json dataset loader because load_dataset does not work
    with open(os.path.join(coco_obj_seg_dir, "instances_train2017.json"), "r") as f:
        coco_obj_seg_tr = [ex["image_id"] for ex in json.load(f)["annotations"]]
    with open(os.path.join(coco_obj_seg_dir, "instances_val2017.json"), "r") as f:
        coco_obj_seg_val = [ex["image_id"] for ex in json.load(f)["annotations"]]

    # Combine the two lists into a hf dataset
    coco_obj_seg = coco_obj_seg_tr + coco_obj_seg_val
    # Get all the unique image_ids
    coco_image_ids = coco_ids or set(coco_obj_seg)

    # Create a dictionary with the number of objects per image id
    coco_obj_seg_counter = Counter(coco_obj_seg)
    coco_obj_seg_filtered = {k: coco_obj_seg_counter.get(k, 0) for k in coco_image_ids}

    # Create a dataframe from the dictionary
    coco_obj_seg_df = pd.DataFrame.from_dict(coco_obj_seg_filtered, orient="index", columns=["n_img_seg_obj"])

    return coco_obj_seg_df


@click.command()
@click.option("--vg_coco_overlap_dir", default=VG_COCO_OVERLAP_DIR)
@click.option("--spacy_model", default="en_core_web_trf")
@click.option("--vg_objects_file", default=VG_OBJECTS_FILE)
@click.option("--vg_relationships_file", default=VG_RELATIONSHIPS_FILE)
@click.option("--coco_obj_seg_dir", default=VG_COCO_OBJ_SEG_DIR)
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
    # 3. Add image segmentation properties
    image_segmentation_ds = add_image_segmentation_properties(
        vg_coco_overlap_graph=graph_ds,
        coco_obj_seg_dir=coco_obj_seg_dir,
        save_to_disk=save_intermediate_steps,
        save_dummy_subset=save_dummy_subset,
        dummy_subset_size=dummy_subset_size,
    )
    logger.info(f"Added image segmentation properties for {len(text_ds)} entries")
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


@click.group()
def cli() -> None:
    """Preprocess the VG + COCO overlap dataset, first for text, then for graph properties (on top of text)."""


if __name__ == "__main__":
    cli.add_command(add_text_properties_wrapper)
    cli.add_command(add_graph_properties_wrapper)
    cli.add_command(add_image_segmentation_properties_wrapper)
    cli.add_command(add_ic_scores_wrapper)
    cli.add_command(add_all_properties)
    cli()
