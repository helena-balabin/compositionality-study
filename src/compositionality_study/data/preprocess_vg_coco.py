"""Preprocesses the Visual Genome + COCO overlap dataset."""
# Imports
import os

import click
import spacy
from datasets import load_from_disk

from compositionality_study.constants import VG_COCO_OVERLAP_DIR
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
    """Add text properties to the dataset."""
    vg_coco_ds = load_from_disk(vg_coco_overlap_dir)
    # Remove unnnecessary columns
    vg_coco_ds = vg_coco_ds.remove_columns(["sentences_tokens", "sentences_sentid"])

    # Flatten the dataset so that there is one caption per example
    vs_coco_ds_flattened = vg_coco_ds.map(flatten_examples, batched=True)

    # Add sentence length
    preprocessed_ds = vs_coco_ds_flattened.map(
        lambda example: example | {"sentence_length": len(example["sentences_raw"].split())}
    )

    # Add dependency parse tree depth
    nlp = spacy.load("en_core_web_sm")
    preprocessed_ds = preprocessed_ds.map(walk_tree_hf_ds, fn_kwargs={"nlp": nlp})

    # Save to disk
    output_dir = os.path.split(vg_coco_overlap_dir)[0]
    preprocessed_ds.save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_text"))
    # Also save a small dummy subset of dummy_subset_size many entries
    if save_dummy_subset:
        preprocessed_ds.select(
            list(range(dummy_subset_size))
        ).save_to_disk(os.path.join(output_dir, "vg_coco_preprocessed_dummy"))


@click.group()
def cli() -> None:
    """Preprocess the VG + COCO overlap dataset."""


if __name__ == "__main__":
    cli.add_command(add_text_properties)
    cli()
