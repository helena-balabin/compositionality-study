"""Download the Visual Genome dataset."""
import json
import os

# Imports
import click
import datasets
import pandas as pd
from datasets import Value, load_dataset

from compositionality_study.constants import LARGE_DATASET_STORAGE_PATH, VG_DIR, VG_METADATA_FILE


@click.command()
@click.option("--cache_dir", type=str, default=LARGE_DATASET_STORAGE_PATH)
@click.option("--full_split", type=bool, default=False)
@click.option("--subset", type=str, default="objects_v1.2.0")
def hf_download_vg_coco_overlap(
    cache_dir: str = LARGE_DATASET_STORAGE_PATH,
    full_split: bool = False,
    subset: str = "objects_v1.2.0",
):
    """Download the Visual Genome dataset from huggingface's datasets and filter for COCO overlap.

    :param cache_dir: Where to store the dataset
    :type cache_dir: str
    :param full_split: Whether to download the full split or a small subset, defaults to False
    :type full_split: bool
    :param subset: Which subset of the dataset to download, defaults to "region_description_v1.2.0"
    :type subset: str
    :return: None
    :rtype: None
    """
    vg_ds = load_dataset(
        "visual_genome",
        name=subset,
        cache_dir=cache_dir,
        split="train" if full_split else "train[:1%]",
    )
    vg_ds.filter(lambda example: example["coco_id"] != "null")


@click.command()
@click.option("--vg_metadata_file", type=str, default=VG_METADATA_FILE)
@click.option("--hf_coco_name", type=str, default="Multimodal-Fatima/COCO_captions_train")
@click.option("--coco_cache_dir", type=str, default=LARGE_DATASET_STORAGE_PATH)
@click.option("--save_dummy_subset", type=bool, default=True)
@click.option("--dummy_subset_size", type=int, default=1000)
@click.option("--output_dir", type=str, default=VG_DIR)
def preprocess_local_vg_files_coco_overlap(
    vg_metadata_file: str = VG_METADATA_FILE,
    hf_coco_name: str = "Multimodal-Fatima/COCO_captions_train",
    coco_cache_dir: str = LARGE_DATASET_STORAGE_PATH,
    save_dummy_subset: bool = True,
    dummy_subset_size: int = 5000,
    output_dir: str = VG_DIR,
):
    """Preprocess the local VG files to filter for COCO overlap to get the COCO captions.

    :param vg_metadata_file: Path to the VG metadata file (image_data.json)
    :type vg_metadata_file: str
    :param hf_coco_name: Name of the COCO dataset on huggingface's datasets (to include the captions)
    :type hf_coco_name: str
    :param coco_cache_dir: Where to store the COCO dataset
    :type coco_cache_dir: str
    :param save_dummy_subset: Whether to save a dummy subset of the VG + COCO overlap dataset
    :type save_dummy_subset: bool
    :param dummy_subset_size: How many entries to save in the dummy subset
    :type dummy_subset_size: int
    :param output_dir: Where to save the VG + COCO overlap dataset to, defaults to VG_DIR
    :type output_dir: str
    :return: None
    :rtype: None
    """
    # Get the VG images that have COCO overlap
    with open(vg_metadata_file, "r") as f:
        vg_metadata = json.load(f)
    vg = [
        {
            "cocoid": m["coco_id"],
            "vg_image_id": m["image_id"],
            "vg_url": m["url"],
            "aspect_ratio": m["width"] / m["height"],
        }
        for m in vg_metadata if m["coco_id"] is not None
    ]
    vg_ds = datasets.Dataset.from_pandas(pd.DataFrame(data=vg))
    # Change the datatype of the cocoid column to int32
    new_features = vg_ds.features.copy()
    new_features["cocoid"] = Value("int32")
    vg_ds = vg_ds.cast(new_features)

    # Get the COCO images that are in VG and have captions
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
        coco_ds = datasets.concatenate_datasets(
            [coco_ds, coco_ds_val_splits, coco_ds_test_splits]
        )

    # Subset relevant columns
    coco_ds = coco_ds.select_columns(["filepath", "sentids", "imgid", "cocoid", "sentences_raw", "id"])
    # Flatten the sentences_raw column
    # Save the COCO dataset to disk
    coco_ds.save_to_disk(os.path.join(output_dir, "coco_complete"))

    coco_ids = set(coco_ds["cocoid"]).intersection(set(vg_ds["cocoid"]))
    coco_ds = coco_ds.filter(lambda example: example["cocoid"] in coco_ids, num_proc=4)
    vg_ds = vg_ds.filter(lambda example: example["cocoid"] in coco_ids, num_proc=4)

    # Merge the two datasets by converting them into dataframes and then merging them
    coco_df = coco_ds.to_pandas()
    vg_df = vg_ds.to_pandas()
    merged_df = pd.merge(coco_df, vg_df, on="cocoid")
    merged_ds = datasets.Dataset.from_pandas(merged_df)
    merged_ds = merged_ds.filter(lambda example: len(example["sentences_raw"]) > 0, num_proc=4)

    # Save the dataset to disk
    merged_ds.save_to_disk(os.path.join(output_dir, "vg_coco_overlap"))
    # Also save a small dummy subset of dummy_subset_size many entries
    if save_dummy_subset:
        merged_ds.select(
            list(range(dummy_subset_size))
        ).save_to_disk(os.path.join(output_dir, "vg_coco_overlap_dummy"))
    return


@click.group()
def cli() -> None:
    """Download the Visual Genome dataset."""


if __name__ == "__main__":
    cli.add_command(hf_download_vg_coco_overlap)
    cli.add_command(preprocess_local_vg_files_coco_overlap)
    cli()
