"""Memory tests to test subjects for paying attention in the scanner."""

import glob
import os
import shutil

import click
import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

from compositionality_study.constants import (
    COCO_IMAGE_DIR,
    COCO_LOCAL_STIMULI_DIR,
    COCO_PREP_ALL,
    MEMORY_TEST_DIR,
)


@click.command()
@click.option("--n_subjects", type=int, default=12, help="Number of subjects.")
@click.option("--n_days", type=int, default=3, help="Number of days.")
@click.option(
    "--input_dir", default=COCO_LOCAL_STIMULI_DIR, type=str, help="Directory with the true stimuli."
)
@click.option(
    "--preprocessed_coco_dir",
    default=COCO_PREP_ALL,
    type=str,
    help="Directory with the lure stimuli.",
)
@click.option(
    "--output_dir", default=MEMORY_TEST_DIR, type=str, help="Directory to save the memory test."
)
@click.option("--n_test", default=50, type=int, help="Number of stimuli in the memory test.")
def create_all_memory_tests(
    n_subjects: int,
    n_days: int,
    input_dir: str,
    preprocessed_coco_dir: str,
    output_dir: str,
    n_test: int = 50,
) -> None:
    """Create memory tests for all subjects and days.

    :param n_subjects: Number of subjects.
    :type n_subjects: int
    :param n_days: Number of days.
    :type n_days: int
    :param input_dir: Directory with the stimuli.
    :type input_dir: str
    :param preprocessed_coco_dir: Directory with the lure stimuli.
    :type preprocessed_coco_dir: str
    :param output_dir: Directory to save the memory test.
    :type output_dir: str
    :param n_test: Number of stimuli in the memory test, defaults to 50.
    :type n_test: int
    """
    for subject_id in range(1, n_subjects + 1):
        for day in range(1, n_days + 1):
            create_memory_test_helper(
                day=str(day),
                subject_id=subject_id + 1,
                input_dir=input_dir,
                preprocessed_coco_dir=preprocessed_coco_dir,
                output_dir=output_dir,
                n_test=n_test,
            )


def copy_memory_test_images(
    memory_test: pd.DataFrame,
    subject_id: int,
    day: str,
    input_dir: str,
    output_dir: str,
) -> None:
    """Copy images for memory test to subject/day specific folder.

    :param memory_test: DataFrame containing memory test information
    :type memory_test: pd.DataFrame
    :param subject_id: Subject ID
    :type subject_id: int
    :param day: Day of experiment
    :type day: str
    :param input_dir: Directory containing real stimuli
    :type input_dir: str
    :param output_dir: Base output directory
    :type output_dir: str
    """
    # Create subject/day specific image directory
    img_dir = os.path.join(output_dir, f"sub-{subject_id}", f"ses-{day}", "images")
    os.makedirs(img_dir, exist_ok=True)

    for _, row in tqdm(
        memory_test.iterrows(),
        desc=f"Copying images for subject {subject_id} day {day}",
        total=len(memory_test),
    ):
        img_id = row["stimulus"]
        # Skip if neither ".jpg" nor ".png" is in the stimulus
        if not any(ext in img_id for ext in [".jpg", ".png"]):
            continue
        is_real = row["label"] == "real"

        if is_real:
            # Real stimuli are in the input directory structure
            src_path = os.path.join(
                input_dir,
                img_id,
            )
        else:
            # Lure stimuli are in the COCO_IMAGE_DIR
            src_path = os.path.join(COCO_IMAGE_DIR, img_id)

        # Copy image to the memory test folder
        if "COCO" in img_id:
            img_id = img_id.replace("COCO_train2014_000000", "")
        dst_path = os.path.join(img_dir, img_id)
        shutil.copy2(src_path, dst_path)


def create_memory_test_helper(
    day: str,
    subject_id: int,
    input_dir: str,
    preprocessed_coco_dir: str,
    output_dir: str,
    n_test: int = 50,
) -> None:
    """Create a memory test for a single subject and day.

    :param day: Day of the experiment.
    :type day: str
    :param subject_id: Subject ID.
    :type subject_id: int
    :param input_dir: Directory with the stimuli.
    :type input_dir: str
    :param preprocessed_coco_dir: Directory with the lure stimuli.
    :type preprocessed_coco_dir: str
    :param output_dir: Directory to save the memory test.
    :type output_dir: str
    :param n_test: Number of stimuli in the memory test, defaults to 50.
    :type n_test: int
    """
    shown_stimuli = load_stimuli(day=day, subject_id=subject_id, input_dir=input_dir)
    # Remove any blank stimuli from the shown stimuli
    shown_stimuli = shown_stimuli[shown_stimuli["stimulus"] != "blank"]
    all_stimuli = load_from_disk(preprocessed_coco_dir).to_pandas()  # type: ignore
    # Rename imgid to img_id in all_stimuli
    all_stimuli.rename(columns={"imgid": "img_id"}, inplace=True)  # type: ignore

    # Get the stimuli that were not shown to the subject
    lure_stimuli = all_stimuli[~all_stimuli["img_id"].isin(shown_stimuli["img_id"])]  # type: ignore
    real_test = shown_stimuli.sample(n=n_test, random_state=42)
    lure_test = lure_stimuli.sample(n=len(real_test), random_state=subject_id * int(day) + int(day))

    # Randomly choose between sentences_raw and filepath for each lure stimulus
    lure_test["stimulus"] = lure_test.apply(
        lambda row: row[np.random.choice(["sentences_raw", "filepath"])], axis=1
    )

    # Combine the real and lure test stimuli
    memory_test = pd.concat([real_test, lure_test])
    memory_test = memory_test.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save one version with labels (create a label 'real' for real stimuli and 'lure' for lure stimuli)
    memory_test["label"] = "real"
    memory_test.loc[memory_test["img_id"].isin(lure_test["img_id"]), "label"] = "lure"

    # Copy images for the memory test
    copy_memory_test_images(
        memory_test=memory_test,
        subject_id=subject_id,
        day=day,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    # Replace "COCO_train2014_000000" with "" in the stimulus
    memory_test["stimulus"] = memory_test["stimulus"].str.replace("COCO_train2014_000000", "")
    memory_test.to_csv(
        os.path.join(
            output_dir,
            f"sub-{subject_id}",
            f"ses-{day}",
            f"memory_test_day_{day}_subject_{subject_id}_with_labels.csv",
        ),
        index=False,
    )

    # Remove all unnecessary columns from the memory test, keep only the "stimulus" column
    memory_test = memory_test["stimulus"]
    # Add another empty column for the response named "write_X_if_seen"
    memory_test = pd.DataFrame(
        {
            "stimulus": memory_test,
            "write_X_if_seen": "" * len(memory_test),
        }
    )

    memory_test.to_csv(
        os.path.join(
            output_dir,
            f"sub-{subject_id}",
            f"ses-{day}",
            f"memory_test_day_{day}_subject_{subject_id}.csv",
        ),
        index=False,
    )


def load_stimuli(
    day: str,
    subject_id: int,
    input_dir: str,
) -> pd.DataFrame:
    """Load stimuli for a given day and subject from all runs.

    :param day: Day of the experiment.
    :type day: str
    :param subject_id: Subject ID.
    :type subject_id: int
    :param input_dir: Directory with the stimuli.
    :type input_dir: str
    :return: Stimuli for the given day and subject.
    :rtype: pd.DataFrame
    """
    stimuli = []
    pattern = os.path.join(
        input_dir,
        "subject_specific_stimuli",
        f"sub-{subject_id}_ses-{day}_task-comp_run-*_events.tsv",
    )
    for file_path in glob.glob(pattern):
        df = pd.read_csv(file_path, sep="\t")
        stimuli.append(df)
    return pd.concat(stimuli, ignore_index=True)


@click.group()
def cli() -> None:
    """Memory test."""


if __name__ == "__main__":
    cli.add_command(create_all_memory_tests)
    cli()
