"""Visualize the category distribution in the selected stimuli."""
# Imports
import os

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_from_disk
from scipy.stats import chi2_contingency

from compositionality_study.constants import VG_COCO_SELECTED_STIMULI_DIR, VISUALIZATIONS_DIR


@click.command()
@click.option("--selected_stimuli_dir", default=VG_COCO_SELECTED_STIMULI_DIR, type=str)
@click.option("--output_dir", default=VISUALIZATIONS_DIR, type=str)
def visualize_categories_in_images(
    selected_stimuli_dir: str = VG_COCO_SELECTED_STIMULI_DIR,
    output_dir: str = VISUALIZATIONS_DIR,
):
    """Visualize the category distribution in the selected stimuli across conditions (high/low complexity).

    :param selected_stimuli_dir: Directory with the selected stimuli, defaults to VG_COCO_SELECTED_STIMULI_DIR
    :type selected_stimuli_dir: str
    :param output_dir: Output directory for the visualization, defaults to VISUALIZATIONS_DIR
    :type output_dir: str
    """
    # Load the dataset
    ds = load_from_disk(selected_stimuli_dir)
    # Map the complexity column (only use the part after the second "_")
    ds = ds.map(
        lambda ex: {"complexity": ex["complexity"].split("_")[2]},
        remove_columns=["complexity"],
    )
    # Map each list of categories to a set of categories
    ds = ds.map(
        lambda ex: {"coco_categories": set(ex["coco_categories"])},
        remove_columns=["coco_categories"],
    )

    # Explode the "coco_categories" column
    df = ds.to_pandas()
    df = df.explode("coco_categories")

    # Plot the category distribution
    # Create a crosstab to prepare data for chi-squared test
    crosstab = pd.crosstab(df["complexity"], df["coco_categories"])

    # Plot the category distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="coco_categories", hue="complexity", palette=["#87CEEB", "#B19CD9"])
    plt.xticks(rotation=90)
    plt.xlabel("COCO Categories")
    plt.ylabel("Count")

    # Perform a chi-squared test to assess the overall difference
    _, p, _, _ = chi2_contingency(crosstab)

    # Annotate the plot with the overall p-value
    plt.title(
        f"Distribution of COCO Categories by Complexity with a Chi-Squared Test resulting in p={round(p, 5)}"
    )

    # Save the figure
    plt.savefig(
        os.path.join(output_dir, "statistics", "category_distribution_in_selected_stimuli.png"),
        dpi=300,
        bbox_inches="tight",
    )


@click.group()
def cli() -> None:
    """Visualize the category distribution in the selected stimuli."""


if __name__ == "__main__":
    cli.add_command(visualize_categories_in_images)
    cli()
