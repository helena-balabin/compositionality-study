"""Download the Visual Genome dataset."""
# Imports
import click
from datasets import load_dataset

from compositionality_study.constants import LARGE_DATASET_STORAGE_PATH


@click.command()
@click.option("--cache_dir", type=str, default=LARGE_DATASET_STORAGE_PATH)
@click.option("--full_split", type=bool, default=False)
@click.option("--subset", type=str, default="objects_v1.2.0")
def download_vg_coco_overlap(
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
        subset=subset,
        cache_dir=cache_dir,
        split="train" if full_split else "train[:1%]",
    )
    vg_ds.filter(lambda example: example["coco_id"] != "null")


@click.group()
def cli() -> None:
    """Download the Visual Genome dataset."""


if __name__ == "__main__":
    cli.add_command(download_vg_coco_overlap)
    cli()
