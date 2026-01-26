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
    COCO_IMAGE_DIR,
    COCO_LOCAL_STIMULI_DIR,
    COCO_SELECTED_STIMULI_DIR,
)
from compositionality_study.utils import apply_gamma_correction

# Set a random seed for reproducibility
random.seed(42)


def create_balanced_conditions(
    df: pd.DataFrame,
    n_groups: int = 7,
    new_group_key: str = "block",
) -> pd.DataFrame:
    """Randomize the order of the stimuli, while keeping the number of stimuli per condition balanced.

    :param df: The dataframe to randomize.
    :type df: pd.DataFrame
    :param n_groups: The number of groups to randomize the stimuli into.
    :type n_groups: int
    :param new_group_key: The name of the column to add to the dataframe that contains the group number.
    :type new_group_key: str
    :return: The randomized dataframe.
    :rtype: pd.DataFrame
    """
    # Group by complexity and modality to ensure balanced distribution
    stimuli_rep_df_grouped = df.groupby(["complexity", "modality"])
    # Create separate dataframes for the groups
    stimuli_rep_dfs = [group.reset_index(drop=True) for _, group in stimuli_rep_df_grouped]

    # For each group, distribute trials evenly across runs
    for i, group in enumerate(stimuli_rep_dfs):
        # Calculate how many trials from this group each run should get
        n_trials_in_group = len(group)
        base_trials_per_run = n_trials_in_group // n_groups
        extra_trials = n_trials_in_group % n_groups

        # Create block assignments ensuring even distribution
        block_assignments = []
        for block_idx in range(n_groups):
            # Some blocks get one extra trial to handle remainder
            trials_in_this_block = base_trials_per_run + (1 if block_idx < extra_trials else 0)
            block_assignments.extend([block_idx] * trials_in_this_block)

        # Shuffle the group to randomize which specific stimuli go to which blocks
        group_shuffled = group.sample(frac=1, random_state=42 + i).reset_index(drop=True)
        group_shuffled[new_group_key] = block_assignments

        stimuli_rep_dfs[i] = group_shuffled

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
    letter_counts = [
        text.count(letter) for letter in string.ascii_lowercase if letter not in "aeiou"
    ]

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
@click.option("--hf_stimuli_dir", default=COCO_SELECTED_STIMULI_DIR, type=str)
@click.option("--local_stimuli_dir", default=COCO_LOCAL_STIMULI_DIR, type=str)
@click.option("--coco_image_dir", default=COCO_IMAGE_DIR, type=str)
@click.option("--text_feature", default="amr_graph_depth", type=str)
@click.option("--delete_existing", default=True, type=bool)
def convert_hf_dataset_to_local_stimuli(
    hf_stimuli_dir: str = COCO_SELECTED_STIMULI_DIR,
    local_stimuli_dir: str = COCO_LOCAL_STIMULI_DIR,
    coco_image_dir: str = COCO_IMAGE_DIR,
    text_feature: str = "amr_n_nodes",
    delete_existing: bool = True,
) -> pd.DataFrame:
    """Convert the stimuli from the huggingface dataset to locally stimuli (images/text).

    :param hf_stimuli_dir: The directory containing the stimuli from the huggingface dataset.
    :type hf_stimuli_dir: str
    :param local_stimuli_dir: The directory to save the locally stimuli to.
    :type local_stimuli_dir: str
    :param coco_image_dir: The directory containing the COCO images.
    :type coco_image_dir: str
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
        # Make sure ex is a dict
        ex = dict(ex)
        # Load the image using filepath
        img = Image.open(os.path.join(coco_image_dir, ex["filepath"]))

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
                        "amr_graph_depth": ex["amr_graph_depth"],
                        "coco_a_graph_depth": ex["coco_a_graph_depth"],
                        "coco_a_edges": ex["coco_a_edges"],
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
@click.option("--local_stimuli_dir", default=COCO_LOCAL_STIMULI_DIR, type=str)
@click.option("--n_subjects", default=12, type=int)
@click.option("--n_runs", default=36, type=int)
@click.option("--n_run_blocks", default=6, type=int)
@click.option("--n_blanks_per_run", default=4, type=int)
@click.option("--n_repetitions", default=6, type=int)
@click.option("--duration", default=3.0, type=float)
@click.option("--isi", default=3.0, type=float)
@click.option("--n_sessions", default=3, type=int)
@click.option("--dummy_scan_duration", default=12.0, type=float)
def generate_subject_specific_stimulus_files(
    local_stimuli_dir: str = COCO_LOCAL_STIMULI_DIR,
    n_subjects: int = 12,
    n_runs: int = 36,
    n_run_blocks: int = 6,
    n_blanks_per_run: int = 4,
    n_repetitions: int = 6,
    duration: float = 3.0,
    isi: float = 3.0,
    n_sessions: int = 3,
    dummy_scan_duration: float = 12.0,
) -> None:
    """Generate subject specific stimulus files with balanced modality assignment across participants.

    Each of the 252 text-image pairs will be split across participants such that each participant
    sees only one modality (either text or image) per pair. Across all participants, each item
    will be shown in both modalities, with half of the participants viewing the text and the
    other half viewing the corresponding image. Each participant will see 126 texts (from one half
    of the stimulus set) and 126 images (from the other half), totaling 252 stimuli per participant.
    Each stimulus will be presented six times, yielding a total of 252 × 6 = 1512 trials per participant.

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

    # Create two complementary splits of the stimuli
    # Each participant will see only one modality per stimulus pair
    # Split 1: all stimuli from first half as text + all stimuli from second half as images
    # Split 2: all stimuli from first half as images + all stimuli from second half as text

    # Sort stimuli by complexity and img_id to ensure balanced splits
    stimuli_df = stimuli_df.sort_values(["complexity", "img_id"]).reset_index(drop=True)

    # Split the data into two halves by complexity to maintain balance
    low_complexity = stimuli_df[stimuli_df["complexity"] == "low"].reset_index(drop=True)
    high_complexity = stimuli_df[stimuli_df["complexity"] == "high"].reset_index(drop=True)

    # Split each complexity group in half
    n_low_per_split = len(low_complexity) // 2
    n_high_per_split = len(high_complexity) // 2

    # First half of stimuli (for modality assignment)
    first_half_low = low_complexity.iloc[:n_low_per_split].copy()
    first_half_high = high_complexity.iloc[:n_high_per_split].copy()

    # Second half of stimuli (for modality assignment)
    second_half_low = low_complexity.iloc[n_low_per_split : n_low_per_split * 2].copy()
    second_half_high = high_complexity.iloc[n_high_per_split : n_high_per_split * 2].copy()

    # Create dataframes for each split
    def create_split_df(text_low, text_high, image_low, image_high):
        """Create a split dataframe with text and image stimuli."""
        split_data = []

        # Add text stimuli
        for df in [text_low, text_high]:
            df_copy = df.copy()
            df_copy["stimulus"] = df_copy["text"]
            df_copy["modality"] = "text"
            split_data.append(df_copy)

        # Add image stimuli
        for df in [image_low, image_high]:
            df_copy = df.copy()
            df_copy["stimulus"] = df_copy["img_path"]
            df_copy["modality"] = "image"
            split_data.append(df_copy)

        result_df = pd.concat(split_data, ignore_index=True)
        result_df.drop(columns=["text", "img_path"], inplace=True)
        return result_df

    # Split 1: first half as text + second half as images
    split1_df = create_split_df(first_half_low, first_half_high, second_half_low, second_half_high)
    # Split 2: first half as images + second half as text
    split2_df = create_split_df(second_half_low, second_half_high, first_half_low, first_half_high)
    # Create list of splits for assignment to subjects
    splits = [split1_df, split2_df]

    # Iterate through the subjects and generate the stimulus files
    for subject in tqdm(
        range(1, n_subjects + 1), desc="Generating subject specific stimulus files"
    ):
        # Assign split based on subject number: first half gets split 1, second half gets split 2
        subject_stimuli_df = splits[subject % 2].copy()

        # First, ensure each unique stimulus is distributed across runs without repetition within runs
        # Create a mapping of each unique stimulus to runs (cycle through runs)
        unique_stimuli = subject_stimuli_df.copy().reset_index(drop=True)

        # Assign each unique stimulus to runs in a round-robin fashion to ensure even distribution
        run_assignments = []
        for i in range(len(unique_stimuli)):
            run_assignments.append(i % n_runs)
        unique_stimuli["base_run"] = run_assignments

        # Now replicate each stimulus n_repetitions times and assign to different runs
        stimuli_with_runs = []
        for _, stimulus in unique_stimuli.iterrows():  # type: ignore
            for rep in range(n_repetitions):
                stimulus_copy = stimulus.copy()
                # Assign this repetition to a run: base_run + rep, cycling through all runs
                stimulus_copy["run"] = (stimulus["base_run"] + rep) % n_runs
                stimuli_with_runs.append(stimulus_copy)

        stimuli_rep_df_rd = pd.DataFrame(stimuli_with_runs).reset_index(drop=True)

        # Shuffle within each run to randomize order while maintaining the distribution
        stimuli_rep_df_rd = (
            stimuli_rep_df_rd.groupby("run")
            .apply(lambda x: x.sample(frac=1, random_state=42).reset_index(drop=True))
            .reset_index(drop=True)
        )

        # Create a copy of the stimuli dataframe for this subject
        subj_stimuli_rep_df = stimuli_rep_df_rd

        # Randomly change the run order by mapping the run numbers to a random permutation of the run numbers, using
        # the subject number as the random seed
        new_run_order = np.random.RandomState(subject).permutation(np.arange(n_runs))
        subj_stimuli_rep_df["run"] = subj_stimuli_rep_df["run"].map(
            lambda x: new_run_order[x]  # type: ignore
        )

        # Within each run, create blocks of stimuli
        subj_stimuli_rep_df = (
            subj_stimuli_rep_df.groupby("run")
            .apply(
                lambda x: create_balanced_conditions(
                    x,
                    n_groups=n_run_blocks,
                    new_group_key="block",
                )
            )
            .reset_index(drop=True)
        )

        # Shuffle within each block within each run
        subj_stimuli_rep_df = (
            subj_stimuli_rep_df.groupby(["run", "block"])
            .apply(
                lambda x: x.sample(
                    frac=1,
                    random_state=subject + x["run"].iloc[0] + x["block"].iloc[0] * 1000,  # noqa
                )
            )
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
                    [
                        run_df.iloc[: insert_position + i],
                        blank_trials.iloc[[i]],
                        run_df.iloc[insert_position + i :],
                    ]
                ).reset_index(drop=True)

            subj_stimuli_rep_df = pd.concat(
                [subj_stimuli_rep_df[subj_stimuli_rep_df["run"] != run], run_df],
                ignore_index=True,
            )

        # Sort the dataframe by run, then by block (don't sort by onset yet since it's not calculated)
        subj_stimuli_rep_df.sort_values(by=["run", "block"], inplace=True)
        # Reset the index
        subj_stimuli_rep_df.reset_index(drop=True, inplace=True)

        # Add the onset and duration columns per run
        # The onset is relative to the start of each run and starts with the dummy scan duration
        for run in range(n_runs):
            run_mask = subj_stimuli_rep_df["run"] == run
            run_indices = subj_stimuli_rep_df[run_mask].index

            # Calculate onsets for this run, starting from dummy scan duration
            run_onsets = np.arange(len(run_indices)) * (duration + isi) + dummy_scan_duration
            subj_stimuli_rep_df.loc[run_indices, "onset"] = run_onsets

        subj_stimuli_rep_df["duration"] = duration

        # Split the runs into multiple sessions and files
        n_runs_per_session = n_runs // n_sessions
        for session in range(n_sessions):
            # Select the runs for the current session
            session_runs = np.arange(
                session * n_runs_per_session, (session + 1) * n_runs_per_session
            )
            session_stimuli_rep_df = subj_stimuli_rep_df[
                subj_stimuli_rep_df["run"].isin(session_runs)
            ].copy()

            # Save each run as a separate file within the session
            for run in session_runs:
                run_stimuli_rep_df = session_stimuli_rep_df[
                    session_stimuli_rep_df["run"] == run
                ].copy()
                # Run % n_runs_per_session
                run_stimuli_rep_df["run"] = run % n_runs_per_session + 1
                run_stimuli_rep_df.to_csv(
                    os.path.join(
                        subj_output_dir,
                        # add plus one here, because subj01 was our pilot
                        f"sub-{subject + 1}_ses-{session + 1}_task-comp_run-{run % n_runs_per_session + 1}_events.tsv",
                    ),
                    index=False,
                    sep="\t",
                )


@click.command()
@click.option("--filtered_stimuli_dir", default=COCO_LOCAL_STIMULI_DIR, type=str)
def map_stimuli_ids_to_design_matrix_idx(
    filtered_stimuli_dir: str = COCO_LOCAL_STIMULI_DIR,
):
    """Map the stimuli IDs to the design matrix indices for later GLMSingle beta estimation.

    :param filtered_stimuli_dir: The directory containing the filtered stimuli images and text,
        defaults to COCO_LOCAL_STIMULI_DIR.
    :type filtered_stimuli_dir: str
    """
    # Load the stimuli dataframe
    stimuli_df = pd.read_csv(os.path.join(filtered_stimuli_dir, "stimuli_text_and_im_paths.csv"))
    # Get the sorted COCO IDs
    sorted_coco_ids = stimuli_df["cocoid"].sort_values().astype(str)
    # Create a df with sorted_coco_ids + "_text" and "_image" below, and index as a new column
    design_matrix_mapping_df = pd.DataFrame(
        {
            "design_matrix_idx": np.arange(2 * len(sorted_coco_ids)),
            "coco_id": pd.concat([sorted_coco_ids + "_text", sorted_coco_ids + "_image"]),
        }
    )
    # Save the mapping
    design_matrix_mapping_df.to_csv(
        os.path.join(filtered_stimuli_dir, "design_matrix_mapping.csv"),
        index=False,
    )


@click.group()
def cli() -> None:
    """Convert the HF dataset with the selected stimuli into locally saved images and text."""


if __name__ == "__main__":
    cli.add_command(convert_hf_dataset_to_local_stimuli)
    cli.add_command(generate_subject_specific_stimulus_files)
    cli.add_command(map_stimuli_ids_to_design_matrix_idx)
    cli()
