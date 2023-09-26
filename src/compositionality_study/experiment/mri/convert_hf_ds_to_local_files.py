"""Convert the HF dataset with the selected stimuli into locally saved images and text."""
import os
import random
import string
from typing import List

import click
import pandas as pd
import requests
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

from compositionality_study.constants import (  # noqa
    VG_COCO_LOCAL_STIMULI_DIR,
    VG_COCO_SELECTED_STIMULI_DIR,
)
from compositionality_study.experiment.mri.create_fourier_scrambled_images import fft_phase_scrambling  # noqa
# noqa
# Set a random seed for reproducibility  # noqa
random.seed(42)  # noqa


def estimate_letter_frequency(
    text_series: pd.Series,
) -> List[float]:
    """Estimate the frequency of each English letter based on a series of text.

    :param text_series: The series of text to use.
    :type text_series: pd.Series
    :return: The estimated letter frequencies.
    :rtype: List[float]
    """
    # Concatenate all the text
    text = " ".join(text_series.values)

    # Count the number of occurrences of each letter
    letter_counts = [text.count(letter) for letter in string.ascii_lowercase]

    # Return the letter frequencies
    return [count / sum(letter_counts) * 100 for count in letter_counts]


def generate_non_word_sentence(
    input_sentence: str,
    letter_frequencies: List[float],
) -> str:
    """Generate nonword from a sentence by sampling letters according to their frequency.

    :param input_sentence: The input sentence.
    :type input_sentence: str
    :param letter_frequencies: The letter frequencies to use.
    :type letter_frequencies: List[float]
    :return: The pseudoword sentence.
    :rtype: str
    """
    # Split the sentence into words
    words = input_sentence.split(" ")

    # For each word, generate a non word of the same length
    non_words = []
    for word in words:
        # Sample letters according to their frequency
        letters = random.choices(string.ascii_lowercase, weights=letter_frequencies, k=len(word))  # noqa

        # Add the non word to the list
        non_words.append("".join(letters))

    # Return the non word sentence
    return " ".join(non_words)


@click.command()
@click.option("--hf_stimuli_dir", default=VG_COCO_SELECTED_STIMULI_DIR, type=str)
@click.option("--local_stimuli_dir", default=VG_COCO_LOCAL_STIMULI_DIR, type=str)
@click.option("--delete_existing", default=True, type=bool)
@click.option("--random_seed", default=42, type=int)
@click.option("--only_control", default=False, type=bool, is_flag=True)
@click.option("--control_fraction", default=0.2, type=float)
def convert_hf_dataset_to_local_stimuli(
    hf_stimuli_dir: str = VG_COCO_SELECTED_STIMULI_DIR,
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    delete_existing: bool = True,
    random_seed: int = 42,
    only_control: bool = False,
    control_fraction: float = 0.2,
) -> pd.DataFrame:
    """Convert the stimuli from the huggingface dataset to locally stimuli (images/text).

    :param hf_stimuli_dir: The directory containing the stimuli from the huggingface dataset.
    :type hf_stimuli_dir: str
    :param local_stimuli_dir: The directory to save the locally stimuli to.
    :type local_stimuli_dir: str
    :param delete_existing: Whether to delete the existing stimuli in the local stimuli directory.
    :type delete_existing: bool
    :param random_seed: The random seed to use for reproducibility.
    :type random_seed: int
    :param only_control: Whether to only generate the control stimuli (scrambled images and non word sentences), assumes
        that the stimuli have already been downloaded.
    :type only_control: bool
    :param control_fraction: The fraction of stimuli to use for the control condition.
    :type control_fraction: float
    :return: A dataframe containing the text and path to the image and image ID.
    :rtype: pd.DataFrame
    """
    # Load the dataset
    dataset = load_from_disk(hf_stimuli_dir)
    # Initialize a dataframe that contains the text, path to the image, image ID and the condition
    stimuli_df = pd.DataFrame(columns=["text", "img_path", "img_id", "complexity"])

    if not only_control:
        # Create the output directory if it does not exist
        if not os.path.exists(local_stimuli_dir):
            os.makedirs(local_stimuli_dir)
        # Delete the existing stimuli if specified
        if delete_existing:
            for f in os.listdir(local_stimuli_dir):
                os.remove(os.path.join(local_stimuli_dir, f))

        # Iterate through the dataset and download the images
        for ex in tqdm(dataset, desc="Downloading images"):
            # Download and save the image
            img = Image.open(requests.get(ex["vg_url"], stream=True, timeout=10).raw)
            output_path = f"{ex['vg_image_id']}_{ex['sentids']}_{ex['complexity']}.jpg"
            img.save(os.path.join(local_stimuli_dir, output_path))
            # Add the text and image path to the dataframe
            stimuli_df = pd.concat(
                [
                    stimuli_df,
                    pd.DataFrame(
                        {
                            "text": ex["sentences_raw"],
                            "img_path": output_path,
                            "img_id": f"{ex['vg_image_id']}_{ex['sentids']}",
                            "complexity": ex["complexity"],
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
    else:
        # Load the existing stimuli_df
        stimuli_df = pd.read_csv(os.path.join(local_stimuli_dir, "stimuli_text_and_im_paths.csv"))

    # Generate null condition stimuli by scrambling control_fraction of the images for each complexity condition
    # Take a subset of the stimuli, stratified by complexity
    stimuli_df_subset = stimuli_df.groupby("complexity").sample(frac=control_fraction, random_state=random_seed)

    # Estimate the letter frequencies
    letter_frequencies = estimate_letter_frequency(stimuli_df_subset["text"])

    for ex in tqdm(stimuli_df_subset.itertuples(), desc="Generating null condition stimuli"):
        # Create a scrambled version of the image and turn it into a PIL image
        scrambled_img = Image.fromarray(
            fft_phase_scrambling(os.path.join(local_stimuli_dir, ex.img_path)),
        )
        sc_img_output_path = f"scrambled_{ex.img_id}.jpg"
        # Write the scrambled image to disk
        scrambled_img.save(os.path.join(local_stimuli_dir, sc_img_output_path))

        # Generate a scrambled (non word) sentence as well
        non_word_sentence = generate_non_word_sentence(ex.text, letter_frequencies)
        # Add the info to the dataframe
        stimuli_df = pd.concat(
            [
                stimuli_df,
                pd.DataFrame(
                    {
                        "text": non_word_sentence,
                        "img_path": sc_img_output_path,
                        "img_id": ex.img_id,
                        "complexity": f"scrambled_{ex.complexity}",
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    # Save the dataframe and return it
    stimuli_df.to_csv(os.path.join(local_stimuli_dir, "stimuli_text_and_im_paths.csv"), index=False)

    return stimuli_df


@click.group()
def cli() -> None:
    """Convert the HF dataset with the selected stimuli into locally saved images and text."""


if __name__ == "__main__":
    cli.add_command(convert_hf_dataset_to_local_stimuli)
    cli()
