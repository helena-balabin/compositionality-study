"""Memory tests to test subjects for paying attention in the scanner."""

import os
import glob

import click
import pandas as pd
from datasets import load_from_disk

from compositionality_study.constants import (
    MEMORY_TEST_DIR,
    VG_COCO_LOCAL_STIMULI_DIR,
    VG_COCO_PREP_ALL,
)


@click.command()
@click.option('--n_subjects', type=int, default=8, help='Number of subjects.')
@click.option('--n_days', type=int, default=3, help='Number of days.')
@click.option('--input_dir', default=VG_COCO_LOCAL_STIMULI_DIR, type=str, help='Directory with the true stimuli.')
@click.option('--preprocessed_vg_coco_dir', default=VG_COCO_PREP_ALL, type=str, help='Directory with the lure stimuli.')
@click.option('--output_dir', default=MEMORY_TEST_DIR, type=str, help='Directory to save the memory test.')
@click.option('--n_test', default=50, type=int, help='Number of stimuli in the memory test.')
def create_all_memory_tests(
    n_subjects: int,
    n_days: int,
    input_dir: str,
    preprocessed_vg_coco_dir: str,
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
    :param preprocessed_vg_coco_dir: Directory with the lure stimuli.
    :type preprocessed_vg_coco_dir: str
    :param output_dir: Directory to save the memory test.
    :type output_dir: str
    :param n_test: Number of stimuli in the memory test, defaults to 50.
    :type n_test: int, optional
    """
    for subject_id in range(1, n_subjects + 1):
        for day in range(1, n_days + 1):
            create_memory_test_helper(
                day=str(day),
                subject_id=subject_id,
                input_dir=input_dir,
                preprocessed_vg_coco_dir=preprocessed_vg_coco_dir,
                output_dir=output_dir,
                n_test=n_test
            )


def create_memory_test_helper(
    day: str,
    subject_id: int,
    input_dir: str,
    preprocessed_vg_coco_dir: str,
    output_dir: str,
    n_test: int = 50,
) -> None:
    """Helper function to create a memory test for a single subject and day.
    
    :param day: Day of the experiment.
    :type day: str
    :param subject_id: Subject ID.
    :type subject_id: int
    :param input_dir: Directory with the stimuli.
    :type input_dir: str
    :param preprocessed_vg_coco_dir: Directory with the lure stimuli.
    :type preprocessed_vg_coco_dir: str
    :param output_dir: Directory to save the memory test.
    :type output_dir: str
    :param n_test: Number of stimuli in the memory test, defaults to 50.
    :type n_test: int, optional
    """
    shown_stimuli = load_stimuli(day=day, subject_id=subject_id, input_dir=input_dir)
    # Remove any blank stimuli from the shown stimuli
    shown_stimuli = shown_stimuli[shown_stimuli['stimulus'] != 'blank']
    all_stimuli = load_from_disk(preprocessed_vg_coco_dir).to_pandas()
    # Rename imgid to img_id in all_stimuli
    all_stimuli.rename(columns={'imgid': 'img_id'}, inplace=True)

    lure_stimuli = all_stimuli[~all_stimuli['img_id'].isin(shown_stimuli['img_id'])]

    real_test = shown_stimuli.sample(n=n_test, random_state=42)
    lure_test = lure_stimuli.sample(n=len(real_test), random_state=42)

    memory_test = pd.concat([real_test, lure_test])
    # Save one version with labels (create a label 'real' for real stimuli and 'lure' for lure stimuli)
    memory_test['label'] = 'real'
    memory_test.loc[memory_test['img_id'].isin(lure_test['img_id']), 'label'] = 'lure'
    memory_test.to_csv(
        os.path.join(output_dir, f'memory_test_day_{day}_subject_{subject_id}_with_labels.csv'),
        index=False,
    )

    # Save another version without labels and shuffle the rows
    memory_test = memory_test.drop(columns='label')
    memory_test.sample(frac=1, random_state=42).reset_index(drop=True)
    memory_test.to_csv(
        os.path.join(output_dir, f'memory_test_day_{day}_subject_{subject_id}.csv'),
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
        f'sub-{subject_id}_ses-{day}_task-comp_run-*_events.tsv'
    )
    for file_path in glob.glob(pattern):
        df = pd.read_csv(file_path, sep='\t')
        stimuli.append(df)
    return pd.concat(stimuli, ignore_index=True)


@click.group()
def cli() -> None:
    """Memory test."""


if __name__ == '__main__':
    cli.add_command(create_all_memory_tests)
    cli()
