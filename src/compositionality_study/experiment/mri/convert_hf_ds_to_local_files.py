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

from compositionality_study.constants import (  # noqa
    VG_COCO_LOCAL_STIMULI_DIR,
    VG_COCO_SELECTED_STIMULI_DIR,
    VG_OBJ_REL_IDX_FILE,
    VG_OBJECTS_FILE,
    VG_RELATIONSHIPS_FILE,
)
from compositionality_study.experiment.mri.create_fourier_scrambled_images import (
    fft_phase_scrambling,
)
from compositionality_study.utils import apply_gamma_correction, draw_objs_and_rels  # noqa

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
        letters = random.choices(  # noqa
            string.ascii_lowercase, weights=letter_frequencies, k=len(word)
        )  # noqa

        # Add the non word to the list
        non_words.append("".join(letters))

    # Return the non word sentence
    return " ".join(non_words)


@click.command()
@click.option("--hf_stimuli_dir", default=VG_COCO_SELECTED_STIMULI_DIR, type=str)
@click.option("--local_stimuli_dir", default=VG_COCO_LOCAL_STIMULI_DIR, type=str)
@click.option("--text_feature", default="amr_graph_depth", type=str)
@click.option("--delete_existing", default=True, type=bool)
@click.option("--random_seed", default=42, type=int)
@click.option("--only_control", default=False, type=bool, is_flag=True)
@click.option("--control_fraction", default=0.2, type=float)
def convert_hf_dataset_to_local_stimuli(
    hf_stimuli_dir: str = VG_COCO_SELECTED_STIMULI_DIR,
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    text_feature: str = "amr_graph_depth",
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
    :param text_feature: The text feature to use for the description of the local stimuli.
    :type text_feature: str
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
            for f in os.listdir(local_stimuli_dir):  # type: ignore
                os.remove(os.path.join(local_stimuli_dir, f))  # type: ignore

        # Iterate through the dataset and load the images
        for ex in tqdm(dataset, desc="Loading images"):
            # Load the image
            img = ex["img"]
            output_name = f"{ex['sentids']}_{ex['complexity']}"

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
                            "complexity": ex["complexity"],
                            text_feature: ex[text_feature],
                            "coco_a_graph_depth": ex["coco_a_graph_depth"],
                            "cocoid": ex["cocoid"],
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
    stimuli_df_subset = stimuli_df.groupby("complexity").sample(
        frac=control_fraction, random_state=random_seed
    )

    # Estimate the letter frequencies
    letter_frequencies = estimate_letter_frequency(stimuli_df_subset["text"])

    for ex in tqdm(stimuli_df_subset.itertuples(), desc="Generating null condition stimuli"):
        # Create a scrambled version of the image and turn it into a PIL image
        scrambled_img = Image.fromarray(
            fft_phase_scrambling(os.path.join(local_stimuli_dir, ex.img_path)),
        )
        sc_img_output_path = f"scrambled_{ex.img_id}.png"
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


@click.command()
@click.option("--local_stimuli_dir", default=VG_COCO_LOCAL_STIMULI_DIR, type=str)
@click.option("--n_subjects", default=48, type=int)
@click.option("--n_runs", default=8, type=int)
@click.option("--n_repetitions", default=3, type=int)
def generate_subject_specific_stimulus_files(
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    n_subjects: int = 48,
    n_runs: int = 8,
    n_repetitions: int = 3,
) -> None:
    """Generate subject specific stimulus files with the correct randomization and repetitions across runs.

    :param local_stimuli_dir: The directory containing the filtered stimuli images and text.
    :type local_stimuli_dir: str
    :param n_subjects: The number of subjects to generate stimulus files for.
    :type n_subjects: int
    :param n_runs: The number of runs to generate stimulus files for.
    :type n_runs: int
    :param n_repetitions: The number of repetitions of each stimulus across runs.
    :type n_repetitions: int
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

    # Iterate through the subjects and generate the stimulus files
    for subject in tqdm(range(n_subjects), desc="Generating subject specific stimulus files"):
        # Group by complexity and modality
        stimuli_rep_df_grouped = stimuli_rep_df.groupby(["complexity", "modality"])
        # Create separate dataframes for the groups
        stimuli_rep_dfs = [group for _, group in stimuli_rep_df_grouped]
        # For each group, add a run column that counts from 0, 1 ..., n_runs - 1 and then repeats 0, 1, ..., n_runs - 1
        for i, group in enumerate(stimuli_rep_dfs):
            # If the modality is "image", add the run numbers in a different order to avoid having paired images and
            # text in the same run
            if group["modality"].iloc[0] == "image":
                group["run"] = np.tile(np.arange(n_runs), len(group) // n_runs)
            else:
                # This is probably overly complicated, but it works
                run_order = np.concatenate(
                    [np.arange(n_runs)[n_runs // 2 :], np.arange(n_runs)[: n_runs // 2]]
                )
                group["run"] = np.tile(run_order, len(group) // n_runs)

            stimuli_rep_dfs[i] = group

        # Concatenate the dataframes back together
        stimuli_rep_df = pd.concat(stimuli_rep_dfs, ignore_index=True)

        # Randomly change the run order by mapping the run numbers to a random permutation of the run numbers, using
        # the subject number as the random seed
        new_run_order = np.random.RandomState(subject).permutation(np.arange(n_runs))
        stimuli_rep_df["run"] = stimuli_rep_df["run"].map(lambda x: new_run_order[x])  # noqa

        # Within each run, shuffle the stimuli, using the subject number as the random seed
        stimuli_rep_df = (
            stimuli_rep_df.groupby("run")
            .apply(lambda x: x.sample(frac=1, random_state=subject))  # noqa
            .reset_index(drop=True)
        )

        # Save the dataframe to disk
        stimuli_rep_df.to_csv(os.path.join(subj_output_dir, f"subj_{subject}.csv"), index=False)


@click.group()
def cli() -> None:
    """Convert the HF dataset with the selected stimuli into locally saved images and text."""


if __name__ == "__main__":
    cli.add_command(convert_hf_dataset_to_local_stimuli)
    cli.add_command(generate_subject_specific_stimulus_files)
    cli()
