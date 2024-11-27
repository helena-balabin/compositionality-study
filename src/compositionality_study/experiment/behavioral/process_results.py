"""Process the outcome of the behavioral experiment based on the extracted results from the output CSV files."""

import os
from typing import Tuple

import click
import pandas as pd
from scipy.stats import chi2_contingency

from compositionality_study.constants import BEHAV_OUTPUT_IMAGES_DIR, BEHAV_OUTPUT_TEXT_DIR


def process_local_results_per_modality(output_dir: str) -> Tuple[float, float]:
    """Process the local results per modality.

    :param output_dir: Directory where the batch csv files for the experiment are stored
    :type output_dir: str
    :return: Percentage of total correct answers and p-value for a Chi-squared test
    :rtype: Tuple[float, float]
    """
    # Load all the batch files for the experiment and concatenate them
    batch_files = [os.path.join(output_dir, file) for file in os.listdir(output_dir) if file.endswith(".csv")]
    batch_dfs = [pd.read_csv(file) for file in batch_files]
    batch_df = pd.concat(batch_dfs, ignore_index=True)
    # Calculate the total percentage of correct answers
    batch_df["correct"] = (
        (batch_df["Input.high_complex_on"] == "left") & (batch_df["Answer.more-complex.label"] == "A")
    ) | ((batch_df["Input.high_complex_on"] == "right") & (batch_df["Answer.more-complex.label"] == "B"))
    total_correct = batch_df["correct"].mean()
    # Create a contingency table for the Chi-squared test
    contingency_table = pd.crosstab(batch_df["Input.high_complex_on"], batch_df["Answer.more-complex.label"])
    # Perform the Chi-squared test, get the p-value
    _, p_value, _, _ = chi2_contingency(contingency_table)
    return total_correct, p_value


@click.command()
@click.option(
    "--image_output_dir",
    default=BEHAV_OUTPUT_IMAGES_DIR,
)
@click.option(
    "--text_output_dir",
    default=BEHAV_OUTPUT_TEXT_DIR,
)
def process_local_results(
    image_output_dir: str,
    text_output_dir: str,
):
    """Process the outcome of the local experiment based on the extracted results from the CSV files.

    :param image_output_dir: The directory where the batch csv files for the image experiment are stored.
    :type image_output_dir: str
    :param text_output_dir: The directory where the batch csv files for the text experiment are stored.
    :type text_output_dir: str
    """
    # Get the results
    img_acc, img_p = process_local_results_per_modality(image_output_dir)
    text_acc, text_p = process_local_results_per_modality(text_output_dir)
    # Write the results to a file (in the parent directory of the output directory)
    with open(os.path.join(os.path.dirname(image_output_dir), "local_results.txt"), "w") as f:
        f.write(f"Image accuracy: {img_acc}\n")
        f.write(f"Image p-value: {img_p}\n")
        f.write(f"Text accuracy: {text_acc}\n")
        f.write(f"Text p-value: {text_p}\n")


@click.group()
def cli() -> None:
    """Analyse the results of the behavioral experiment."""


if __name__ == "__main__":
    cli.add_command(process_local_results)
    cli()
