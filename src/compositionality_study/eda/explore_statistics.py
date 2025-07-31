"""Run some basic statistics on the VG + COCO overlap captions."""

# Imports
import os
from typing import List

import click
import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from loguru import logger
from tqdm import tqdm

from compositionality_study.constants import (
    COCO_PREP_ALL,
    COCO_PREP_TEXT_GRAPH_DIR,
    VISUALIZATIONS_DIR,
)
from compositionality_study.utils import walk_tree


@click.command()
@click.option("--coco_ds_dir", type=str, default=COCO_PREP_ALL)
@click.option("--vis_output_dir", type=str, default=VISUALIZATIONS_DIR)
@click.option("--color", type=str, default="#66C2A5")
def sent_len_histogram(
    coco_ds_dir: str = COCO_PREP_ALL,
    vis_output_dir: str = VISUALIZATIONS_DIR,
    color: str = "#66C2A5",
):
    """Plot a histogram of the sentence lengths for all COCO captions.

    :param coco_ds_dir: The VG + COCO overlap dataset with the captions to plot the histogram for
    :type coco_ds_dir: str
    :param vis_output_dir: Where to save the figures of the plots to, defaults to VISUALIZATIONS_DIR
    :type vis_output_dir: str
    :param color: Color of the bars in the histogram, defaults to "#66C2A5"
    :type color: str
    """
    # Get the sentences and their lengths
    coco_ds = datasets.load_from_disk(coco_ds_dir)
    captions_nested = coco_ds["sentences_raw"]
    captions = [sent for sent_list in captions_nested for sent in sent_list]
    sent_lens = [len(sent.split()) for sent in captions]

    # Set seaborn theme
    plt.rcParams["font.family"] = ["sans-serif"]  # noqa
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
@click.option("--coco_ds_dir", type=str, default=COCO_PREP_ALL)
@click.option("--vis_output_dir", type=str, default=VISUALIZATIONS_DIR)
@click.option("--color", type=str, default="#FFD966")
def dep_parse_tree_depth_histogram(
    coco_ds_dir: str = COCO_PREP_ALL,
    vis_output_dir: str = VISUALIZATIONS_DIR,
    color: str = "#FFD966",
):
    """Plot a histogram of the dependency parse tree depths for all COCO captions.

    :param coco_ds_dir: The VG + COCO overlap dataset with the captions to plot the histogram for
    :type coco_ds_dir: str
    :param vis_output_dir: Where to save the figures of the plots to, defaults to VISUALIZATIONS_DIR
    :type vis_output_dir: str
    :param color: Color of the bars in the histogram, defaults to "#66C2A5"
    :type color: str
    """
    # Get the sentences
    coco_ds = datasets.load_from_disk(coco_ds_dir)
    captions_nested = coco_ds["sentences_raw"]
    captions = [sent for sent_list in captions_nested for sent in sent_list]

    # Get the dependency parse tree depths
    # You might need to run `python -m spacy download en_core_web_trf` to download the model before
    nlp = spacy.load("en_core_web_trf")
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
@click.option("--coco_ds_dir", type=str, default=COCO_PREP_ALL)
@click.option("--vis_output_dir", type=str, default=VISUALIZATIONS_DIR)
@click.option("--color", type=str, default="#9B9ED4")
def sent_len_dep_depth_correlation(
    coco_ds_dir: str = COCO_PREP_ALL,
    vis_output_dir: str = VISUALIZATIONS_DIR,
    color: str = "#9B9ED4",
):
    """Plot a scatter plot of the sentence lengths and dependency parse tree depths for all COCO captions.

    :param coco_ds_dir: The VG + COCO overlap dataset with the captions to plot the histogram for
    :type coco_ds_dir: str
    :param vis_output_dir: Where to save the figures of the plots to, defaults to VISUALIZATIONS_DIR
    :type vis_output_dir: str
    :param color: Color of the bars in the histogram, defaults to "#66C2A5"
    :type color: str
    """
    # Get the sentences
    coco_ds = datasets.load_from_disk(coco_ds_dir)
    captions_nested = coco_ds["sentences_raw"]
    captions = [sent for sent_list in captions_nested for sent in sent_list]

    # Get their sentence lengths and dep parse tree depths
    sent_lens = [len(sent.split()) for sent in captions]
    # You might need to run `python -m spacy download en_core_web_trf` to download the model before
    nlp = spacy.load("en_core_web_trf")
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


@click.command()
@click.option("--coco_text_graph_dir", type=str, default=COCO_PREP_TEXT_GRAPH_DIR)
@click.option(
    "--columns",
    type=str,
    multiple=True,
    default=["sentence_length", "parse_tree_depth", "n_obj", "n_rel"],
)
@click.option("--vis_output_dir", type=str, default=VISUALIZATIONS_DIR)
@click.option("--color", type=str, default="#6C8EBF")
def all_properties_corr(
    coco_text_graph_dir: str = COCO_PREP_TEXT_GRAPH_DIR,
    columns: List[str] = ["sentence_length", "parse_tree_depth", "n_obj", "n_rel"],  # noqa
    vis_output_dir: str = VISUALIZATIONS_DIR,
    color: str = "#6C8EBF",
):
    """Plot a pairplot of the text and graph properties of the VG + COCO overlap dataset.

    :param coco_text_graph_dir: The VG + COCO overlap dataset to plot the pairplot for
    :type coco_text_graph_dir: str
    :param columns: The columns to plot the pairplot for,
        defaults to ["sentence_length", "parse_tree_depth", "n_obj", "n_rel"]
    :type columns: List[str]
    :param vis_output_dir: Where to save the figures of the plots to, defaults to VISUALIZATIONS_DIR
    :type vis_output_dir: str
    :param color: Color of the points in the pairplot, defaults to "#6BAED6"
    :type color: str
    """
    # Load the dataset and convert to pandas
    ds = datasets.load_from_disk(coco_text_graph_dir)
    df = ds.to_pandas()

    # Set seaborn theme
    plt.rcParams["font.family"] = ["sans-serif"]
    sns.set_style("ticks")

    # Create a pairplot
    pairplot = sns.pairplot(
        df[list(columns)],
        kind="reg",
        diag_kind="kde",
        plot_kws={"color": color, "scatter_kws": {"s": 1}},
        diag_kws={"color": color},
    )
    plt.tight_layout()

    # Calculate correlation coefficients
    correlation_matrix = df[list(columns)].corr()

    # Add correlation coefficients as annotations
    for i, (ax_row, correlation_row) in enumerate(zip(pairplot.axes, correlation_matrix.values)):
        for j, (ax, correlation) in enumerate(zip(ax_row, correlation_row)):
            if i != j:
                ax.annotate(
                    f"r = {correlation:.2f}",
                    (0.5, 0.9),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=10,
                )

    # Create the output directory
    vis_statistics_dir = os.path.join(vis_output_dir, "statistics")
    os.makedirs(vis_statistics_dir, exist_ok=True)

    # Save the figure
    plt.savefig(os.path.join(vis_statistics_dir, "properties_correlation_plot.png"), dpi=300)


@click.command()
@click.option("--coco_dir", type=str, default=COCO_PREP_ALL)
@click.option("--spacy_model", type=str, default="en_core_web_trf")
def check_captions_for_verbs(
    coco_dir: str = COCO_PREP_ALL,
    spacy_model: str = "en_core_web_trf",
):
    """Check how many of the captions actually have a verb in them.

    :param coco_dir: The VG + COCO overlap dataset to check the captions for verbs for
    :type coco_dir: str
    :param spacy_model: The spaCy model to use for POS tagging, defaults to "en_core_web_trf"
    :type spacy_model: str
    """
    # Load the dataset
    coco_ds = datasets.load_from_disk(coco_dir)

    # Get the captions
    captions = coco_ds["sentences_raw"]

    # Check how many of the captions have a verb in them
    nlp = spacy.load(spacy_model)
    n_verbs = 0

    for caption in tqdm(captions, desc="Checking captions for verbs"):
        doc = nlp(caption)
        if "VERB" in set(i.pos_ for i in doc):
            n_verbs += 1

    # Print the results
    logger.info(f"{n_verbs/len(captions) * 100:.2f}% of the captions have a verb in them.")


@click.group()
def cli() -> None:
    """Generate basic statistics on the VG + COCO overlap captions dataset."""


if __name__ == "__main__":
    cli.add_command(sent_len_histogram)
    cli.add_command(dep_parse_tree_depth_histogram)
    cli.add_command(sent_len_dep_depth_correlation)
    cli.add_command(all_properties_corr)
    cli.add_command(check_captions_for_verbs)
    cli()
