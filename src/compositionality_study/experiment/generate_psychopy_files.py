"""Generate the psychopy experiment files for the fMRI experiment based on the chosen stimuli."""
import os
import random
from typing import List

# Imports
import click
import pandas as pd
from psychopy import core, visual
import requests
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

from compositionality_study.constants import (
    VG_COCO_LOCAL_STIMULI_DIR,
    VG_COCO_SELECTED_STIMULI_DIR,
)
from compositionality_study.experiment.create_fourier_scrambled_images import fft_phase_scrambling


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

    for ex in tqdm(stimuli_df_subset.itertuples(), desc="Generating null condition stimuli"):
        # Create a scrambled version of the image and turn it into a PIL image
        scrambled_img = Image.fromarray(fft_phase_scrambling(os.path.join(local_stimuli_dir, ex.img_path)))
        sc_img_output_path = f"scrambled_{ex['vg_image_id']}_{ex['sentids']}.jpg"
        # Write the scrambled image to disk
        scrambled_img.save(os.path.join(local_stimuli_dir, sc_img_output_path))

        # TODO generate scrambled text as well
        # TODO add the info to the dataframe

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
    :param stimuli: The stimuli to present.
    :type stimuli: List
    :param image_duration: The duration of the image presentation in seconds.
    :type image_duration: float
    :param inter_trial_interval: The inter-trial interval in seconds.
    :type inter_trial_interval: float
    """
    # Create the visual stimuli
    for ex in stimuli:
        img = visual.ImageStim(win, image=ex[0])
        img.draw()
        win.flip()
        core.wait(image_duration)
        win.flip()
        core.wait(inter_trial_interval)


@click.command()
@click.option("--local_stimuli_dir", type=str, default=VG_COCO_SELECTED_STIMULI_DIR)
@click.option("--image_duration", type=float, default=1.0)
@click.option("--inter_trial_interval", type=float, default=0.5)
@click.option("--n_repetitions", type=int, default=3)
def generate_psychopy_exp(
    local_stimuli_dir: str = VG_COCO_SELECTED_STIMULI_DIR,
    image_duration: float = 1.0,
    inter_trial_interval: float = 0.5,
    n_repetitions: int = 3,
):
    """Load the selected stimuli and present them in a psychopy experiment.

    :param local_stimuli_dir: The directory containing the previously selected stimuli.
    :type local_stimuli_dir: str
    :param image_duration: The duration of the image presentation in seconds.
    :type image_duration: float
    :param inter_trial_interval: The inter-trial interval in seconds.
    :type inter_trial_interval: float
    :param n_repetitions: The number of repetitions of each stimuli.
    :type n_repetitions: int
    """
    # TODO load the stimuli locally
    stimuli_ds = load_from_disk(local_stimuli_dir)
    # Load the captions and images from the URL in the dataset
    stimuli = []
    for example in tqdm(stimuli_ds, desc="Loading stimuli"):
        stimuli = stimuli + [(
            Image.open(requests.get(example["vg_url"], stream=True).raw), example["sentences_raw"]
        )] * n_repetitions
    # Shuffle the stimuli (deterministically)
    random.seed(42)
    random.shuffle(stimuli)

    # Create the psychopy window
    win = visual.Window(size=(800, 600), fullscr=False, allowGUI=True, color='white')

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
