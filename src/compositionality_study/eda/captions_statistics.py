"""Run some basic statistics on the VG + COCO overlap captions."""
# Imports
import os

import click
import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from tqdm import tqdm

from compositionality_study.constants import VG_COCO_OVERLAP_DIR, VISUALIZATIONS_DIR
from compositionality_study.utils import walk_tree


@click.command()
@click.option("--coco_vg_ds_dir", type=str, default=VG_COCO_OVERLAP_DIR)
@click.option("--vis_output_dir", type=str, default=VISUALIZATIONS_DIR)
@click.option("--color", type=str, default="#66C2A5")
def sent_len_histogram(
    coco_vg_ds_dir: str = VG_COCO_OVERLAP_DIR,
    vis_output_dir: str = VISUALIZATIONS_DIR,
    color: str = "#66C2A5",
):
    """Plot a histogram of the sentence lengths for all COCO captions.

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

    # Set seaborn theme
    plt.rcParams['font.family'] = ['sans-serif']
    sns.set_style("ticks")

    # Create the histogram
    sns.displot(sent_lens, color=color, kind="hist", bins=50)
    plt.title("VG + COCO overlap: Sentence length histogram")
    plt.xlabel("Sentence length")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Create the output directory
    vis_statistics_dir = os.path.join(vis_output_dir, "statistics")
    os.makedirs(vis_statistics_dir, exist_ok=True)

    # Save the figure
    plt.savefig(os.path.join(vis_statistics_dir, "sentence_length_histogram.png"), dpi=300)


@click.command()
@click.option("--coco_vg_ds_dir", type=str, default=VG_COCO_OVERLAP_DIR)
@click.option("--vis_output_dir", type=str, default=VISUALIZATIONS_DIR)
@click.option("--color", type=str, default="#FFD966")
def dep_parse_tree_depth_histogram(
    coco_vg_ds_dir: str = VG_COCO_OVERLAP_DIR,
    vis_output_dir: str = VISUALIZATIONS_DIR,
    color: str = "#FFD966",
):
    """Plot a histogram of the dependency parse tree depths for all COCO captions.

    :param coco_vg_ds_dir: The VG + COCO overlap dataset with the captions to plot the histogram for
    :type coco_vg_ds_dir: str
    :param vis_output_dir: Where to save the figures of the plots to, defaults to VISUALIZATIONS_DIR
    :type vis_output_dir: str
    :param color: Color of the bars in the histogram, defaults to "#66C2A5"
    :type color: str
    :return: None
    :rtype: None
    """
    # Get the sentences
    coco_vg_ds = datasets.load_from_disk(coco_vg_ds_dir)
    captions_nested = coco_vg_ds["sentences_raw"]
    captions = [sent for sent_list in captions_nested for sent in sent_list]

    # Get the dependency parse tree depths
    # You might need to run `python -m spacy download en_core_web_sm` to download the model before
    nlp = spacy.load("en_core_web_sm")
    depths = []
    for sentence in tqdm(captions, desc="Determing dependency parse tree depths"):
        doc = nlp(sentence)
        depths.append(walk_tree(next(doc.sents).root, 0))

    # Set seaborn theme
    plt.rcParams["font.family"] = ["sans-serif"]
    sns.set_style("ticks")

    # Create the histogram
    sns.displot(depths, color=color, kind="hist", bins=50)
    plt.title("VG + COCO overlap: Depedency parse tree depth histogram")
    plt.xlabel("Dependency parse tree depth")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Create the output directory
    vis_statistics_dir = os.path.join(vis_output_dir, "statistics")
    os.makedirs(vis_statistics_dir, exist_ok=True)

    # Save the figure
    plt.savefig(os.path.join(vis_statistics_dir, "dep_parse_tree_depth_histogram.png"), dpi=300)


@click.command()
@click.option("--coco_vg_ds_dir", type=str, default=VG_COCO_OVERLAP_DIR)
@click.option("--vis_output_dir", type=str, default=VISUALIZATIONS_DIR)
@click.option("--color", type=str, default="#9B9ED4")
def sent_len_dep_depth_correlation(
    coco_vg_ds_dir: str = VG_COCO_OVERLAP_DIR,
    vis_output_dir: str = VISUALIZATIONS_DIR,
    color: str = "#9B9ED4",
):
    """Plot a scatter plot of the sentence lengths and dependency parse tree depths for all COCO captions.

    :param coco_vg_ds_dir: The VG + COCO overlap dataset with the captions to plot the histogram for
    :type coco_vg_ds_dir: str
    :param vis_output_dir: Where to save the figures of the plots to, defaults to VISUALIZATIONS_DIR
    :type vis_output_dir: str
    :param color: Color of the bars in the histogram, defaults to "#66C2A5"
    :type color: str
    :return: None
    :rtype: None
    """
    # Get the sentences
    coco_vg_ds = datasets.load_from_disk(coco_vg_ds_dir)
    captions_nested = coco_vg_ds["sentences_raw"]
    captions = [sent for sent_list in captions_nested for sent in sent_list]

    # Get their sentence lengths and dep parse tree depths
    sent_lens = [len(sent.split()) for sent in captions]
    # You might need to run `python -m spacy download en_core_web_sm` to download the model before
    nlp = spacy.load("en_core_web_sm")
    depths = []
    for sentence in tqdm(captions, desc="Determing dependency parse tree depths"):
        doc = nlp(sentence)
        depths.append(walk_tree(next(doc.sents).root, 0))

    # Set seaborn theme
    plt.rcParams["font.family"] = ["sans-serif"]
    sns.set_style("ticks")

    # Create a scatter plot
    sns.scatterplot(x=depths, y=sent_lens, color=color)
    plt.title("VG + COCO overlap: \n Sentence length vs. dependency parse tree depth")
    plt.xlabel("Dependency parse tree depth")
    plt.ylabel("Sentence length")
    plt.tight_layout()

    # Create the output directory
    vis_statistics_dir = os.path.join(vis_output_dir, "statistics")
    os.makedirs(vis_statistics_dir, exist_ok=True)

    # Save the figure
    plt.savefig(os.path.join(vis_statistics_dir, "len_dep_correlation_plot.png"), dpi=300)


@click.group()
def cli() -> None:
    """Generate basic statistics on the VG + COCO overlap captions dataset."""


if __name__ == "__main__":
    cli.add_command(sent_len_histogram)
    cli.add_command(dep_parse_tree_depth_histogram)
    cli.add_command(sent_len_dep_depth_correlation)
    cli()
