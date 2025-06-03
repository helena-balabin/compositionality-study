"""Download the Visual Genome dataset."""

import json
import os

# Imports
import click
import datasets
from datasets import load_dataset

from compositionality_study.constants import COCO_A_ANNOT_FILE, COCO_DIR, LARGE_DATASET_STORAGE_PATH


@click.command()
@click.option("--hf_coco_name", type=str, default="Multimodal-Fatima/COCO_captions_train")
@click.option("--coco_a_annot_file", type=str, default=COCO_A_ANNOT_FILE)
@click.option("--coco_cache_dir", type=str, default=LARGE_DATASET_STORAGE_PATH)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=50)
@click.option("--output_dir", type=str, default=COCO_DIR)
def download_coco(
    hf_coco_name: str = "Multimodal-Fatima/COCO_captions_train",
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    coco_cache_dir: str = LARGE_DATASET_STORAGE_PATH,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 50,
    output_dir: str = COCO_DIR,
):
    """Download the COCO dataset.

    :param hf_coco_name: Name of the COCO dataset on huggingface's datasets (to include the captions)
    :type hf_coco_name: str
    :param coco_a_annot_file: Path to the COCO actions annotations file
    :type coco_a_annot_file: str
    :param coco_cache_dir: Where to store the COCO dataset
    :type coco_cache_dir: str
    :param save_dummy_subset: Whether to save a dummy subset of the COCO dataset
    :type save_dummy_subset: bool
    :param dummy_subset_size: How many entries to save in the dummy subset
    :type dummy_subset_size: int
    :param output_dir: Where to save the COCO dataset to, defaults to COCO_DIR
    :type output_dir: str
    :return: None
    :rtype: None
    """
    # Download the COCO dataset from Huggingface
    coco_ds = load_dataset(hf_coco_name, cache_dir=coco_cache_dir)["train"]

    if "Multimodal-Fatima" in hf_coco_name:
        # Also add the separate validation splits in this case
        coco_ds_val_splits = load_dataset(
            hf_coco_name.replace("_train", "_validation"),
            cache_dir=coco_cache_dir,
        )["validation"]
        # As well as the test splits
        coco_ds_test_splits = load_dataset(
            hf_coco_name.replace("_train", "_test"),
            cache_dir=coco_cache_dir,
        )["test"]
        coco_ds = datasets.concatenate_datasets([coco_ds, coco_ds_val_splits, coco_ds_test_splits])

    # Subset relevant columns
    coco_ds = coco_ds.select_columns(["filepath", "sentids", "imgid", "cocoid", "sentences_raw", "id"])
    # Flatten the sentences_raw column
    # Save the COCO dataset to disk
    coco_ds = coco_ds.filter(lambda example: len(example["sentences_raw"]) > 0, num_proc=4)
    coco_ds.save_to_disk(os.path.join(output_dir, "coco_complete"))

    # Subset the dataset to those entries that are present in COCO-A
    with open(coco_a_annot_file, "r") as f:
        coco_a_annot = json.load(f)["annotations"]["3"]
    coco_a_ids = set([annot["image_id"] for annot in coco_a_annot])
    coco_a_ds = coco_ds.filter(lambda example: example["cocoid"] in coco_a_ids, num_proc=4)
    # Save the COCO-A dataset to disk
    coco_a_ds.save_to_disk(os.path.join(output_dir, "coco_a"))

    # Also save a small dummy subset of dummy_subset_size many entries
    if save_dummy_subset:
        coco_ds.select(list(range(dummy_subset_size))).save_to_disk(os.path.join(output_dir, "coco_dummy"))
    return


@click.group()
def cli() -> None:
    """Download the COCO dataset."""


if __name__ == "__main__":
    cli.add_command(download_coco)
    cli()
