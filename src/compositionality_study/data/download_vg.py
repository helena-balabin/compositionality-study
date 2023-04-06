"""Download the Visual Genome dataset."""
# Imports
import click
from datasets import load_dataset

from compositionality_study.constants import LARGE_DATASET_STORAGE_PATH


@click.command()
@click.option("--cache_dir", type=str, default=LARGE_DATASET_STORAGE_PATH)
@click.option("--subset", type=str, default="region_description_v1.2.0")
def download_vg(
    cache_dir: str = LARGE_DATASET_STORAGE_PATH,
    subset: str = "region_description_v1.2.0",
):
    """Download the Visual Genome dataset.

    :param cache_dir: Where to store the dataset
    :type cache_dir: str
    :param subset: Which subset of the dataset to download, defaults to "region_description_v1.2.0"
    :type subset: str
    :return: None
    :rtype: None
    """
    load_dataset("visual_genome", subset=subset, cache_dir=cache_dir)


@click.group()
def cli() -> None:
    """Download the Visual Genome dataset."""


if __name__ == "__main__":
    cli.add_command(download_vg)
    cli()
