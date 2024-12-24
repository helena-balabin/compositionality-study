"""Convert the HF dataset with the selected stimuli into locally saved images and text."""

import os
import random
import string
from typing import List

import click
import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

from compositionality_study.constants import (
    VG_COCO_LOCAL_STIMULI_DIR,
    VG_COCO_SELECTED_STIMULI_DIR,
)
from compositionality_study.utils import apply_gamma_correction

# Set a random seed for reproducibility
random.seed(42)


def create_balanced_conditions(
    df: pd.DataFrame,
    n_groups: int = 7,
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
    # TODO fix n_groups
    # Group by complexity and modality
    stimuli_rep_df_grouped = df.groupby(["complexity", "modality"])
    # Create separate dataframes for the groups
    stimuli_rep_dfs = [group.reset_index(drop=True) for _, group in stimuli_rep_df_grouped]
    # For each group, add a run column that counts from 0, 1 ..., n_runs - 1 and then repeats 0, 1, ..., n_runs - 1
    for i, group in enumerate(stimuli_rep_dfs):
        effective_group_size = min(n_groups, len(group))
        if avoid_pairings and group["modality"].iloc[0] == "image":
            # If the modality is "image", add the run numbers in a different order to avoid having paired images and
            # text in the same run
            # This is probably overly complicated, but it works
            run_order = np.concatenate(
                [
                    np.arange(effective_group_size)[effective_group_size // 2 :],
                    np.arange(effective_group_size)[: effective_group_size // 2],
                ]
            )
            group[new_group_key] = np.resize(run_order, len(group))
        else:
            group[new_group_key] = np.resize(np.arange(effective_group_size), len(group))

        stimuli_rep_dfs[i] = group

    # Stitch the dataframe together
    return pd.concat(stimuli_rep_dfs, ignore_index=True)


def estimate_letter_frequency(
    text_series: pd.Series,
) -> List[float]:
    """Estimate the frequency of each English letter excluding vowels based on a series of text.

    :param text_series: The series of text to use.
    :type text_series: pd.Series
    :return: The estimated letter frequencies.
    :rtype: List[float]
    """
    # Concatenate all the text
    text = " ".join(text_series.values)

    # Count the number of occurrences of each letter, but filter out vowels
    letter_counts = [text.count(letter) for letter in string.ascii_lowercase if letter not in "aeiou"]

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
            "".join([i for i in string.ascii_lowercase if i not in "aeiou"]),
            weights=letter_frequencies,
            k=len(word),
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
def convert_hf_dataset_to_local_stimuli(
    hf_stimuli_dir: str = VG_COCO_SELECTED_STIMULI_DIR,
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    text_feature: str = "amr_graph_depth",
    delete_existing: bool = True,
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
                        "complexity": "high" if ex["coco_a_graph_depth"] > 1 else "low",
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
@click.option("--n_subjects", default=12, type=int)
@click.option("--n_runs", default=36, type=int)
@click.option("--n_run_blocks", default=6, type=int)
@click.option("--n_blanks_per_run", default=4, type=int)
@click.option("--n_repetitions", default=3, type=int)
@click.option("--duration", default=3.0, type=float)
@click.option("--isi", default=3.0, type=float)
@click.option("--n_sessions", default=3, type=int)
@click.option("--dummy_scan_duration", default=12.0, type=float)
def generate_subject_specific_stimulus_files(
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    n_subjects: int = 12,
    n_runs: int = 36,
    n_run_blocks: int = 6,
    n_blanks_per_run: int = 4,
    n_repetitions: int = 3,
    duration: float = 3.0,
    isi: float = 3.0,
    n_sessions: int = 3,
    dummy_scan_duration: float = 12.0,
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
    :param n_blanks_per_run: The number of blank trials per run.
    :type n_blanks_per_run: int
    :param n_repetitions: The number of repetitions of each stimulus across runs.
    :type n_repetitions: int
    :param duration: The duration of each stimulus in seconds.
    :type duration: float
    :param isi: The duration of the inter stimulus interval in seconds.
    :type isi: float
    :param n_sessions: The number of sessions to split the runs into.
    :type n_sessions: int
    :param dummy_scan_duration: The duration of the dummy scan in seconds (at the start and end).
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
            .apply(lambda x: x.sample(
                frac=1,
                random_state=subject + x["run"].iloc[0] + x["block"].iloc[0] * 1000,
            ))
            .reset_index(drop=True)
        )

        # Add blank trials to each run
        blank_trial = {
            "stimulus": "blank",
            "modality": "blank",
            "run": None,
            "block": None,
            "onset": None,
            "duration": duration,
        }
        for run in range(n_runs):
            run_df = subj_stimuli_rep_df[subj_stimuli_rep_df["run"] == run].copy()
            blank_trials = pd.DataFrame([blank_trial] * n_blanks_per_run)
            blank_trials["run"] = run
            blank_trials["block"] = np.random.RandomState(subject + run).randint(
                n_run_blocks, size=n_blanks_per_run
            )
            insert_positions = np.random.RandomState(subject + run).choice(
                len(run_df) + 1, n_blanks_per_run, replace=False
            )
            # Insert blank trials at pseudorandom positions within the run
            insert_positions.sort()
            for i, insert_position in enumerate(insert_positions):
                run_df = pd.concat(
                    [run_df.iloc[:insert_position + i], blank_trials.iloc[[i]], run_df.iloc[insert_position + i:]]
                ).reset_index(drop=True)

            subj_stimuli_rep_df = pd.concat(
                [subj_stimuli_rep_df[subj_stimuli_rep_df["run"] != run], run_df],
                ignore_index=True,
            )

        # Sort the dataframe by run, then by block, then by onset
        subj_stimuli_rep_df.sort_values(by=["run", "block", "onset"], inplace=True)
        # Reset the index
        subj_stimuli_rep_df.reset_index(drop=True, inplace=True)

        # Add the onset and duration columns
        # The onset is relative to the start of the run and starts with the dummy scan duration
        subj_stimuli_rep_df["onset"] = np.arange(len(subj_stimuli_rep_df)) % (len(subj_stimuli_rep_df) // n_runs)
        subj_stimuli_rep_df["onset"] *= duration + isi
        subj_stimuli_rep_df["onset"] += dummy_scan_duration
        subj_stimuli_rep_df["duration"] = duration

        # Split the runs into multiple sessions and files
        n_runs_per_session = n_runs // n_sessions
        for session in range(n_sessions):
            # Select the runs for the current session
            session_runs = np.arange(session * n_runs_per_session, (session + 1) * n_runs_per_session)
            session_stimuli_rep_df = subj_stimuli_rep_df[subj_stimuli_rep_df["run"].isin(session_runs)].copy()

            # Save each run as a separate file within the session
            for run in session_runs:
                run_stimuli_rep_df = session_stimuli_rep_df[session_stimuli_rep_df["run"] == run].copy()
                # Run % n_runs_per_session
                run_stimuli_rep_df["run"] = run % n_runs_per_session + 1
                run_stimuli_rep_df.to_csv(
                    os.path.join(
                        subj_output_dir,
                        f"sub-{subject}_ses-{session + 1}_task-comp_run-{run % n_runs_per_session + 1}_events.tsv",
                    ),
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
