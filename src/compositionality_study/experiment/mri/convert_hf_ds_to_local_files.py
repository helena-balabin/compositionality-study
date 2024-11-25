"""Convert the HF dataset with the selected stimuli into locally saved images and text."""

import os
import random
import string
from typing import List

import click
import numpy as np
import pandas as pd
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

from compositionality_study.constants import (
    THINGS_IMAGE_DIR,
    VG_COCO_LOCAL_STIMULI_DIR,
    VG_COCO_SELECTED_STIMULI_DIR,
)
from compositionality_study.utils import apply_gamma_correction

# Set a random seed for reproducibility
random.seed(42)


def create_balanced_conditions(
    df: pd.DataFrame,
    n_groups: int = 8,
    new_group_key: str = "runs",
    avoid_pairings: bool = False,
) -> pd.DataFrame:
    """Randomize the order of the stimuli, while keeping the number of stimuli per condition balanced.

    :param df: The dataframe to randomize.
    :type df: pd.DataFrame
    :param n_groups: The number of groups to randomize the stimuli into.
    :type n_groups: int
    :param new_group_key: The name of the column to add to the dataframe that contains the group number.
    :type new_group_key: str
    :param avoid_pairings: Whether to avoid having paired images and text in the same run.
    :type avoid_pairings: bool
    :return: The randomized dataframe.
    :rtype: pd.DataFrame
    """
    # Group by complexity and modality
    stimuli_rep_df_grouped = df.groupby(["complexity", "modality"])
    # Create separate dataframes for the groups
    stimuli_rep_dfs = [group for _, group in stimuli_rep_df_grouped]
    # For each group, add a run column that counts from 0, 1 ..., n_runs - 1 and then repeats 0, 1, ..., n_runs - 1
    for i, group in enumerate(stimuli_rep_dfs):
        if avoid_pairings and group["modality"].iloc[0] == "image":
            # If the modality is "image", add the run numbers in a different order to avoid having paired images and
            # text in the same run
            # This is probably overly complicated, but it works
            effective_group_size = min(n_groups, len(group))
            run_order = np.concatenate(
                [
                    np.arange(effective_group_size)[effective_group_size // 2 :],
                    np.arange(effective_group_size)[: effective_group_size // 2],
                ]
            )
            group[new_group_key] = np.tile(run_order, len(group) // effective_group_size)
        else:
            effective_group_size = min(n_groups, len(group))
            group[new_group_key] = np.tile(np.arange(effective_group_size), len(group) // effective_group_size)

        stimuli_rep_dfs[i] = group

    # Stitch the dataframe together
    return pd.concat(stimuli_rep_dfs, ignore_index=True)


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
    consonant_letter_string: bool = False,
) -> str:
    """Generate nonword from a sentence by sampling letters according to their frequency.

    :param input_sentence: The input sentence.
    :type input_sentence: str
    :param letter_frequencies: The letter frequencies to use.
    :type letter_frequencies: List[float]
    :param consonant_letter_string: Whether to generate a nonword with only consonants.
    :type consonant_letter_string: bool
    :return: The pseudoword sentence.
    :rtype: str
    """
    # Split the sentence into words
    words = input_sentence.split(" ")

    # For each word, generate a non word of the same length
    non_words = []
    for word in words:
        # Sample letters according to their frequency
        letters = random.choices(  # noqa
            ["x"] if consonant_letter_string else string.ascii_lowercase,
            weights=[1.0] if consonant_letter_string else letter_frequencies,
            k=len(word),
        )  # noqa

        # Add the non word to the list
        non_words.append("".join(letters))

    # Return the non word sentence
    return " ".join(non_words)


@click.command()
@click.option("--hf_stimuli_dir", default=VG_COCO_SELECTED_STIMULI_DIR, type=str)
@click.option("--local_stimuli_dir", default=VG_COCO_LOCAL_STIMULI_DIR, type=str)
@click.option("--things_stimuli_dir", default=THINGS_IMAGE_DIR, type=str)
@click.option("--text_feature", default="amr_graph_depth", type=str)
@click.option("--delete_existing", default=True, type=bool)
@click.option("--random_state", default=42, type=int)
def convert_hf_dataset_to_local_stimuli(
    hf_stimuli_dir: str = VG_COCO_SELECTED_STIMULI_DIR,
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    things_stimuli_dir: str = THINGS_IMAGE_DIR,
    text_feature: str = "amr_graph_depth",
    delete_existing: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convert the stimuli from the huggingface dataset to locally stimuli (images/text).

    :param hf_stimuli_dir: The directory containing the stimuli from the huggingface dataset.
    :type hf_stimuli_dir: str
    :param local_stimuli_dir: The directory to save the locally stimuli to.
    :type local_stimuli_dir: str
    :param things_stimuli_dir: The directory containing the THINGS dataset images.
    :type things_stimuli_dir: str
    :param text_feature: The text feature to use for the description of the local stimuli.
    :type text_feature: str
    :param delete_existing: Whether to delete the existing stimuli in the local stimuli directory.
    :type delete_existing: bool
    :param random_state: The random state to use for reproducibility.
    :type random_state: int
    :return: A dataframe containing the text and path to the image and image ID.
    :rtype: pd.DataFrame
    """
    # Load the dataset
    dataset = load_from_disk(hf_stimuli_dir)
    # Initialize a dataframe that contains the text, path to the image, image ID and the condition
    stimuli_df = pd.DataFrame(columns=["text", "img_path", "img_id"])

    # Create the output directory if it does not exist
    if not os.path.exists(local_stimuli_dir):
        os.makedirs(local_stimuli_dir)
    # Delete the existing stimuli if specified
    if delete_existing:
        for f in os.listdir(local_stimuli_dir):  # type: ignore
            os.remove(os.path.join(local_stimuli_dir, f))  # type: ignore

    # Iterate through the dataset and load the images
    for ex in tqdm(dataset, desc="Loading images"):
        # Load the image
        img = ex["img"]
        # Get the aspect ratio of the image
        aspect_ratio = img.size[0] / img.size[1]
        output_name = f"{ex['sentids']}"

        # Apply gamma correction to the image
        img = apply_gamma_correction(img)

        # Save the images to disk
        img.save(os.path.join(local_stimuli_dir, output_name + ".png"))

        # Add the text and image path to the dataframe
        stimuli_df = pd.concat(
            [
                stimuli_df,
                pd.DataFrame(
                    {
                        "text": ex["sentences_raw"],
                        "img_path": output_name + ".png",
                        "img_id": f"{ex['sentids']}",
                        text_feature: ex[text_feature],
                        "coco_a_graph_depth": ex["coco_a_graph_depth"],
                        "cocoid": ex["cocoid"],
                        "coco_person": ex["coco_person"],
                        "aspect_ratio": aspect_ratio,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    # Generate null condition stimuli based on the images in the THINGS dataset
    # Estimate the letter frequencies
    letter_frequencies = estimate_letter_frequency(stimuli_df["text"])
    # Load all the paths in the THINGS dataset
    things_img_paths = os.listdir(things_stimuli_dir)
    # Get a random subset of the texts in stimuli_df of the same length as the THINGS dataset
    texts = list(stimuli_df.sample(n=len(things_img_paths), random_state=random_state)["text"].values)

    # Get the "control" stimuli
    for things_img_path, text in tqdm(zip(things_img_paths, texts), desc="Generating null condition stimuli"):
        # Save the images to disk
        img = Image.open(os.path.join(things_stimuli_dir, things_img_path))
        img.save(os.path.join(local_stimuli_dir, things_img_path))

        # Get the thing by stripping the last 8 characters and replacing the underscores with spaces
        thing = things_img_path[:-8].replace("_", " ")

        # Generate a scrambled (non word) sentence as well
        non_word_sentence = generate_non_word_sentence(text, letter_frequencies, consonant_letter_string=True)
        # Replace the middle word with the THING
        non_word_sentence = non_word_sentence.split(" ")
        non_word_sentence[len(non_word_sentence) // 2] = thing
        # Join the words back together
        non_word_sentence = " ".join(non_word_sentence)
        # Add the info to the dataframe
        stimuli_df = pd.concat(
            [
                stimuli_df,
                pd.DataFrame(
                    {
                        "text": non_word_sentence,
                        "img_path": things_img_path,
                        "img_id": things_img_path.strip(".jpg"),
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    # Save the dataframe and return it
    stimuli_df.to_csv(os.path.join(local_stimuli_dir, "stimuli_text_and_im_paths.csv"), index=False)

    return stimuli_df


@click.command()
@click.option("--local_stimuli_dir", default=VG_COCO_LOCAL_STIMULI_DIR, type=str)
@click.option("--n_subjects", default=48, type=int)
@click.option("--n_runs", default=12, type=int)
@click.option("--n_run_blocks", default=6, type=int)
@click.option("--n_repetitions", default=3, type=int)
@click.option("--duration", default=3.0, type=float)
@click.option("--isi", default=1.0, type=float)
@click.option("--dummy_scan_duration", default=8.0, type=float)
def generate_subject_specific_stimulus_files(
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    n_subjects: int = 48,
    n_runs: int = 12,
    n_run_blocks: int = 6,
    n_repetitions: int = 3,
    duration: float = 3.0,
    isi: float = 1.0,
    dummy_scan_duration: float = 8.0,
) -> None:
    """Generate subject specific stimulus files with the correct randomization and repetitions across runs.

    :param local_stimuli_dir: The directory containing the filtered stimuli images and text.
    :type local_stimuli_dir: str
    :param n_subjects: The number of subjects to generate stimulus files for.
    :type n_subjects: int
    :param n_runs: The number of runs to generate stimulus files for.
    :type n_runs: int
    :param n_run_blocks: The number of blocks per run (only relevant for randomization).
    :type n_run_blocks: int
    :param n_repetitions: The number of repetitions of each stimulus across runs.
    :type n_repetitions: int
    :param duration: The duration of each stimulus in seconds.
    :type duration: float
    :param isi: The duration of the inter stimulus interval in seconds.
    :type isi: float
    :param dummy_scan_duration: The duration of the dummy scan in seconds.
    :type dummy_scan_duration: float
    """
    # Load the stimuli dataframe
    stimuli_df = pd.read_csv(os.path.join(local_stimuli_dir, "stimuli_text_and_im_paths.csv"))
    # Drop empty rows
    stimuli_df.dropna(inplace=True, how="all")

    # Create the output directory
    subj_output_dir = os.path.join(local_stimuli_dir, "subject_specific_stimuli")
    if not os.path.exists(subj_output_dir):
        os.makedirs(subj_output_dir)

    # Replace "scrambled_*" with "scrambled" in the complexity column
    stimuli_df["complexity"] = stimuli_df["complexity"].str.replace(
        "scrambled_.*",
        "scrambled",
        regex=True,
    )
    # Repeat each stimulus twice to separate image and text
    stimuli_df = stimuli_df.loc[stimuli_df.index.repeat(2)].reset_index(drop=True)
    # Separate text and images
    # Create a new column for the stimulus, every 2nd row is the image, every 2nd row is the text
    stimuli_df["stimulus"] = ""
    stimuli_df["stimulus"][::2] = stimuli_df["img_path"][::2]
    stimuli_df["stimulus"][1::2] = stimuli_df["text"][1::2]
    stimuli_df["modality"] = ""
    stimuli_df["modality"][::2] = ["image"] * len(stimuli_df[::2])
    stimuli_df["modality"][1::2] = ["text"] * len(stimuli_df[1::2])
    # Drop the text and image columns
    stimuli_df.drop(columns=["text", "img_path"], inplace=True)

    # Repeat each stimulus n_repetitions times right after each other
    stimuli_rep_df = stimuli_df.loc[stimuli_df.index.repeat(n_repetitions)].reset_index(drop=True)

    # Randomize the order of the stimuli, while keeping the number of stimuli per condition balanced
    stimuli_rep_df_rd = create_balanced_conditions(
        stimuli_rep_df,
        n_groups=n_runs,
        new_group_key="run",
        avoid_pairings=True,
    )

    # Iterate through the subjects and generate the stimulus files
    for subject in tqdm(range(1, n_subjects + 1), desc="Generating subject specific stimulus files"):
        # Create a copy of the stimuli dataframe
        subj_stimuli_rep_df = stimuli_rep_df_rd

        # Randomly change the run order by mapping the run numbers to a random permutation of the run numbers, using
        # the subject number as the random seed
        new_run_order = np.random.RandomState(subject).permutation(np.arange(n_runs))
        subj_stimuli_rep_df["run"] = subj_stimuli_rep_df["run"].map(lambda x: new_run_order[x])  # noqa

        # Within each run, create blocks of stimuli
        subj_stimuli_rep_df = (
            subj_stimuli_rep_df.groupby("run")
            .apply(
                lambda x: create_balanced_conditions(
                    x,
                    n_groups=n_run_blocks,
                    new_group_key="block",
                    avoid_pairings=False,
                )
            )
            .reset_index(drop=True)
        )

        # Shuffle within each block within each run
        subj_stimuli_rep_df = (
            subj_stimuli_rep_df.groupby(["run", "block"])
            .apply(lambda x: x.sample(frac=1, random_state=subject))  # noqa
            .reset_index(drop=True)
        )

        # Sort the dataframe by run, then by block
        subj_stimuli_rep_df.sort_values(by=["run", "block"], inplace=True)
        # Reset the index
        subj_stimuli_rep_df.reset_index(drop=True, inplace=True)

        # Add the onset and duration columns
        # The onset is relative to the start of the run and starts with the dummy scan duration
        subj_stimuli_rep_df["onset"] = np.arange(len(subj_stimuli_rep_df)) % (len(subj_stimuli_rep_df) // n_runs)
        subj_stimuli_rep_df["onset"] *= duration + isi
        subj_stimuli_rep_df["onset"] += dummy_scan_duration
        subj_stimuli_rep_df["duration"] = duration

        # Save the dataframe to disk
        subj_stimuli_rep_df.to_csv(
            os.path.join(subj_output_dir, f"sub-{subject}_task-comp_events.tsv"),
            index=False,
            sep="\t",
        )


@click.group()
def cli() -> None:
    """Convert the HF dataset with the selected stimuli into locally saved images and text."""


if __name__ == "__main__":
    cli.add_command(convert_hf_dataset_to_local_stimuli)
    cli.add_command(generate_subject_specific_stimulus_files)
    cli()
