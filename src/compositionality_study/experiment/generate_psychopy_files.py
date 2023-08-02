"""Generate the psychopy experiment files for the fMRI experiment based on the chosen stimuli."""
import os
import random
import string
from typing import List

# Imports
import click
import pandas as pd
from psychopy import core, visual
import requests
from PIL import Image, ImageOps
from datasets import load_from_disk
from tqdm import tqdm

from compositionality_study.constants import (
    VG_COCO_LOCAL_STIMULI_DIR,
    VG_COCO_SELECTED_STIMULI_DIR,
)
from compositionality_study.experiment.create_fourier_scrambled_images import fft_phase_scrambling

# Set a random seed for reproducibility
random.seed(42)


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
        letters = random.choices(string.ascii_lowercase, weights=letter_frequencies, k=len(word))

        # Add the non word to the list
        non_words.append("".join(letters))

    # Return the non word sentence
    return " ".join(non_words)


@click.command()
@click.option("--hf_stimuli_dir", default=VG_COCO_SELECTED_STIMULI_DIR, type=str)
@click.option("--local_stimuli_dir", default=VG_COCO_LOCAL_STIMULI_DIR, type=str)
@click.option("--delete_existing", default=True, type=bool)
@click.option("--random_seed", default=42, type=int)
def convert_hf_dataset_to_local_stimuli(
    hf_stimuli_dir: str = VG_COCO_SELECTED_STIMULI_DIR,
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    delete_existing: bool = True,
    random_seed: int = 42,
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
    :return: A dataframe containing the text and path to the image and image ID.
    :rtype: pd.DataFrame
    """
    # Load the dataset
    dataset = load_from_disk(hf_stimuli_dir)
    # Initialize a dataframe that contains the text, path to the image, image ID and the condition
    stimuli_df = pd.DataFrame(columns=["text", "img_path", "img_id", "complexity"])

    # Create the output directory if it does not exist
    if not os.path.exists(local_stimuli_dir):
        os.makedirs(local_stimuli_dir)
    # Delete the existing stimuli if specified
    if delete_existing:
        for f in os.listdir(local_stimuli_dir):
            os.remove(os.path.join(local_stimuli_dir, f))

    # Iterate through the dataset
    for ex in tqdm(dataset, desc="Downloading images"):
        # Download and save the image
        img = Image.open(requests.get(ex["vg_url"], stream=True).raw)
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

    # Generate null condition stimuli by scrambling half of the images for each complexity condition
    # Take a subset of the stimuli, stratified by complexity
    stimuli_df_subset = stimuli_df.groupby("complexity").sample(frac=0.5, random_state=random_seed)

    # Estimate the letter frequencies
    letter_frequencies = estimate_letter_frequency(stimuli_df_subset["text"])

    for ex in tqdm(stimuli_df_subset.itertuples(), desc="Generating null condition stimuli"):
        # Create a scrambled version of the image and turn it into a PIL image
        scrambled_img = Image.fromarray(fft_phase_scrambling(os.path.join(local_stimuli_dir, ex.img_path)))
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


def run_images(
    win: visual.Window,
    stimuli: List,
    image_duration: float = 1.0,
    inter_trial_interval: float = 0.5,
):
    """Present the images in the stimuli list in a psychopy window.

    :param win: The psychopy window.
    :type win: visual.Window
    :param stimuli: The stimuli to present (either visual or textual).
    :type stimuli: List
    :param image_duration: The duration of the image presentation in seconds.
    :type image_duration: float
    :param inter_trial_interval: The inter-trial interval in seconds.
    :type inter_trial_interval: float
    """
    # Create the visual or textual stimuli
    for ex in stimuli:
        if isinstance(ex, Image.Image):
            img = visual.ImageStim(win, image=ex)
            img.draw()
        else:
            txt = visual.TextStim(win, text=ex, color=(0.0, 0.0, 0.0))
            txt.draw()
        win.flip()
        core.wait(image_duration)
        win.flip()
        core.wait(inter_trial_interval)


@click.command()
@click.option("--local_stimuli_dir", type=str, default=VG_COCO_LOCAL_STIMULI_DIR)
@click.option("--image_duration", type=float, default=1.0)
@click.option("--inter_trial_interval", type=float, default=0.5)
@click.option("--n_repetitions", type=int, default=3)
def generate_psychopy_exp(
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    image_duration: float = 1.0,
    inter_trial_interval: float = 0.5,
    n_repetitions: int = 3,
):
    """Load the selected stimuli and present them in a psychopy experiment.

    :param local_stimuli_dir: The directory containing the previously selected stimuli (converted from the HF dataset).
    :type local_stimuli_dir: str
    :param image_duration: The duration of the image presentation in seconds.
    :type image_duration: float
    :param inter_trial_interval: The inter-trial interval in seconds.
    :type inter_trial_interval: float
    :param n_repetitions: The number of repetitions of each stimuli.
    :type n_repetitions: int
    """
    # Load the csv file for the stimuli
    stimuli_df = pd.read_csv(os.path.join(local_stimuli_dir, "stimuli_text_and_im_paths.csv"))
    # Load the captions and images from the URL in the dataset
    stimuli = []
    for _, row in tqdm(stimuli_df.iterrows(), desc="Loading stimuli"):
        # Load the image from disk and resize it
        img = Image.open(os.path.join(local_stimuli_dir, row["img_path"]))
        # Add the image to the stimuli list (n_repetitions times)
        stimuli = stimuli + [ImageOps.contain(img, (800, 600))] * n_repetitions
        # Add the text to the stimuli list (n_repetitions times)
        stimuli = stimuli + [row["text"]] * n_repetitions

    # Shuffle the stimuli (deterministically, because we set the random seed)
    random.shuffle(stimuli)

    # Create the psychopy window
    win = visual.Window(size=(1600, 1200), fullscr=True, allowGUI=True, color='white')

    # Run the experiment
    run_images(
        win=win,
        stimuli=stimuli,
        image_duration=image_duration,
        inter_trial_interval=inter_trial_interval,
    )


@click.group()
def cli() -> None:
    """Generate the psychopy experiment files for the fMRI experiment based on the chosen stimuli."""


if __name__ == "__main__":
    cli.add_command(generate_psychopy_exp)
    cli.add_command(convert_hf_dataset_to_local_stimuli)
    cli()
