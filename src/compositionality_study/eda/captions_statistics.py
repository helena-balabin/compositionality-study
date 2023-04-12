"""Run some basic statistics on the VG + COCO overlap captions."""
# Imports
import os

import click
import datasets
import matplotlib.pyplot as plt
import seaborn as sns

from compositionality_study.constants import VG_COCO_OVERLAP_DIR, VISUALIZATIONS_DIR


@click.command()
@click.option("--coco_vg_ds_dir", type=str, default=VG_COCO_OVERLAP_DIR)
@click.option("--vis_output_dir", type=str, default=VISUALIZATIONS_DIR)
@click.option("--color", type=str, default="#66C2A5")
def sent_len_histogram(
    coco_vg_ds_dir: str = VG_COCO_OVERLAP_DIR,
    vis_output_dir: str = VISUALIZATIONS_DIR,
    color: str = "#66C2A5",
):
    """Plot a histogram of the sentence lengths.

    :param coco_vg_ds_dir: The VG + COCO overlap dataset with the captions to plot the histogram for
    :type coco_vg_ds_dir: str
    :param vis_output_dir: Where to save the figures of the plots to, defaults to VISUALIZATIONS_DIR
    :type vis_output_dir: str
    :param color: Color of the bars in the histogram, defaults to "#66C2A5"
    :type color: str
    :return: None
    :rtype: None
    """
    # Get the sentences and their lengths
    coco_vg_ds = datasets.load_from_disk(coco_vg_ds_dir)
    captions_nested = coco_vg_ds["sentences_raw"]
    captions = [sent for sent_list in captions_nested for sent in sent_list]
    sent_lens = [len(sent.split()) for sent in captions]

    # Create the histogram
    sns.histplot(sent_lens, color=color)
    plt.title("VG + COCO overlap: Sentence length histogram")
    plt.xlabel("Sentence length")
    plt.ylabel("Frequency")

    # Create the output directory
    vis_statistics_dir = os.path.join(vis_output_dir, "statistics")
    os.makedirs(vis_statistics_dir, exist_ok=True)

    # Save the figure
    plt.savefig(os.path.join(vis_statistics_dir, "sentence_length_histogram.png"), dpi=300)


@click.group()
def cli() -> None:
    """Generate basic statistics on the VG + COCO overlap captions dataset."""


if __name__ == "__main__":
    cli.add_command(sent_len_histogram)
    cli()
