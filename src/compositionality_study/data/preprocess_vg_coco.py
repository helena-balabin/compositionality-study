"""Preprocesses the Visual Genome + COCO overlap dataset."""
# Imports
import os

import click
import pandas as pd
import spacy
from datasets import Dataset, load_dataset, load_from_disk

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
    image_ids = set(preprocessed_ds["vg_image_id"])
    preprocessed_df = preprocessed_ds.to_pandas()

    # Load the object and relationship files as hf datasets from json
    vg_objects = load_dataset("json", data_files=vg_objects_file)
    vg_relationships = load_dataset("json", data_files=vg_relationships_file)

    # Get the number of objects
    # Filter for those objects and relationships that are in the VG + COCO overlap dataset
    vg_objects_filtered = vg_objects.filter(
        lambda example: "image_id" in example.keys() and example["image_id"] in image_ids,
        num_proc=4,
    )
    vg_n_obj = vg_objects_filtered.map(
        lambda example: example | {"n_obj": len(example["objects"])},
        num_proc=4,
    )
    vg_n_obj = vg_n_obj.rename_column("image_id", "vg_image_id")
    vg_n_obj = vg_n_obj.remove_columns(["objects", "image_url"])
    vg_n_obj_df = pd.DataFrame(vg_n_obj["train"])

    # Get the number of relationships
    vg_relationships_filtered = vg_relationships.filter(
        lambda example: "image_id" in example.keys() and example["image_id"] in image_ids,
        num_proc=4,
    )
    vg_n_rel = vg_relationships_filtered.map(
        lambda example: example | {"n_rel": len(example["relationships"])},
        num_proc=4,
    )
    vg_n_rel = vg_n_rel.rename_column("image_id", "vg_image_id")
    vg_n_rel = vg_n_rel.remove_columns(["relationships"])
    vg_n_rel_df = pd.DataFrame(vg_n_rel["train"])

    # Add the number of obj and rels by merging everything nicely with to_pandas and then back to hf datasets
    vg_graph_df = pd.merge(vg_n_obj_df, vg_n_rel_df, on="vg_image_id")
    vg_graph_merged_ds = Dataset.from_pandas(pd.merge(preprocessed_df, vg_graph_df, on="vg_image_id"))

    # Save to disk
    output_dir = os.path.split(vg_coco_overlap_text_dir)[0]
    vg_graph_merged_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_graph"))
    # Also save a small dummy subset of dummy_subset_size many entries
    if save_dummy_subset:
        vg_graph_merged_ds.select(
            list(range(dummy_subset_size))
        ).save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_graph_dummy"))


@click.group()
def cli() -> None:
    """Preprocess the VG + COCO overlap dataset, first for text, then for graph properties (on top of text)."""


if __name__ == "__main__":
    cli.add_command(add_text_properties)
    cli.add_command(add_graph_properties)
    cli()
