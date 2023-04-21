"""Preprocesses the Visual Genome + COCO overlap dataset."""
# Imports
import os
from typing import Any, Dict, List, Optional

import click
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm

from compositionality_study.constants import (
    VG_COCO_OVERLAP_DIR,
    VG_COCO_PREPROCESSED_TEXT_DIR,
    VG_OBJECTS_FILE,
    VG_RELATIONSHIPS_FILE,
)
from compositionality_study.utils import flatten_examples, walk_tree_hf_ds


@click.command()
@click.option("--vg_coco_overlap_dir", type=str, default=VG_COCO_OVERLAP_DIR)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
def add_text_properties(
    vg_coco_overlap_dir: str = VG_COCO_OVERLAP_DIR,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
):
    """Add text properties to the dataset.

    :param vg_coco_overlap_dir: Path to the directory where the Visual Genome + COCO overlap dataset is stored.
    :type vg_coco_overlap_dir: str
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    """
    vg_coco_ds = load_from_disk(vg_coco_overlap_dir)
    # Remove unnnecessary columns
    vg_coco_ds = vg_coco_ds.remove_columns(["sentences_tokens", "sentences_sentid"])

    # Flatten the dataset so that there is one caption per example
    vs_coco_ds_flattened = vg_coco_ds.map(flatten_examples, batched=True, num_proc=4)

    # Add sentence length
    preprocessed_ds = vs_coco_ds_flattened.map(
        lambda example: example | {"sentence_length": len(example["sentences_raw"].split())}, num_proc=4,
    )

    # Add dependency parse tree depth
    nlp = spacy.load("en_core_web_sm")
    preprocessed_ds = preprocessed_ds.map(walk_tree_hf_ds, fn_kwargs={"nlp": nlp}, num_proc=4)

    # Save to disk
    output_dir = os.path.split(vg_coco_overlap_dir)[0]
    preprocessed_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_text"))
    # Also save a small dummy subset of dummy_subset_size many entries
    if save_dummy_subset:
        preprocessed_ds.select(
            list(range(dummy_subset_size))
        ).save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_text_dummy"))


@click.command()
@click.option("--vg_coco_overlap_text_dir", type=str, default=VG_COCO_PREPROCESSED_TEXT_DIR)
@click.option("--vg_objects_file", type=str, default=VG_OBJECTS_FILE)
@click.option("--vg_relationships_file", type=str, default=VG_RELATIONSHIPS_FILE)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
def add_graph_properties(
    vg_coco_overlap_text_dir: str = VG_COCO_PREPROCESSED_TEXT_DIR,
    vg_objects_file: str = VG_OBJECTS_FILE,
    vg_relationships_file: str = VG_RELATIONSHIPS_FILE,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 1000,
):
    """Add graph properties to the dataset.

    :param vg_coco_overlap_text_dir: Path to the directory where the Visual Genome +
        COCO overlap dataset with text properties is stored.
    :type vg_coco_overlap_text_dir: str
    :param vg_objects_file: Path to the file where the Visual Genome objects json is stored.
    :type vg_objects_file: str
    :param vg_relationships_file: Path to the file where the Visual Genome relationship json is stored.
    :type vg_relationships_file: str
    :param save_dummy_subset: Whether to save a dummy subset of the dataset, defaults to True
    :type save_dummy_subset: bool
    :param dummy_subset_size: Size of the dummy subset, defaults to 1000
    :type dummy_subset_size: int
    """
    # Load the dataset
    preprocessed_ds = load_from_disk(vg_coco_overlap_text_dir)
    image_ids = list(set(preprocessed_ds["vg_image_id"]))
    preprocessed_df = preprocessed_ds.to_pandas()

    # Get the graph characteristics
    vg_graph_dict = determine_graph_complexity_measures(vg_objects_file, vg_relationships_file, image_ids)
    vg_graph_df = pd.DataFrame(vg_graph_dict).rename(columns={"image_id": "vg_image_id"})

    vg_graph_merged_ds = Dataset.from_pandas(pd.merge(preprocessed_df, vg_graph_df, on="vg_image_id"))

    # Save to disk
    output_dir = os.path.split(vg_coco_overlap_text_dir)[0]
    vg_graph_merged_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_graph"))
    # Also save a small dummy subset of dummy_subset_size many entries
    if save_dummy_subset:
        vg_graph_merged_ds.select(
            list(range(dummy_subset_size))
        ).save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_graph_dummy"))


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

        # Determine the characteristics + add the image id as well
        measures = {
            "image_id": obj["image_id"],
            "avg_node_degree": np.mean([d for _, d in graph.degree()]) if nx.number_of_nodes(graph) > 0 else 0,
            "avg_clustering_coefficient": nx.algorithms.approximation.average_clustering(
                graph,
                trials=1000,
                seed=42,
            ) if nx.number_of_nodes(graph) > 0 else 0,
            "density": nx.density(graph),
            "n_connected_components": nx.number_connected_components(graph),
            "n_obj": len(obj["objects"]),
            "n_rel": len(rel["relationships"]),
        }
        # Append them to the list
        graph_measures.append(measures)

    return graph_measures


@click.group()
def cli() -> None:
    """Preprocess the VG + COCO overlap dataset, first for text, then for graph properties (on top of text)."""


if __name__ == "__main__":
    cli.add_command(add_text_properties)
    cli.add_command(add_graph_properties)
    cli()
