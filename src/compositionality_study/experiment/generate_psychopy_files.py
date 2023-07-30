"""Generate the psychopy experiment files for the fMRI experiment based on the chosen stimuli."""
import random
from typing import List

# Imports
import click
from psychopy import core, visual
import requests
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

from compositionality_study.constants import (
    VG_COCO_SELECTED_STIMULI_DIR,
)


def run_images(
    win: visual.Window,
    stimuli: List[Image],
    image_duration: float = 1.0,
    inter_trial_interval: float = 0.5,
):
    """Present the images in the stimuli list in a psychopy window.

    :param win: The psychopy window.
    :type win: visual.Window
    :param stimuli: The stimuli to present.
    :type stimuli: List[Image]
    :param image_duration: The duration of the image presentation in seconds.
    :type image_duration: float
    :param inter_trial_interval: The inter-trial interval in seconds.
    :type inter_trial_interval: float
    """
    # Create the visual stimuli
    for ex in stimuli:
        img = visual.ImageStim(win, image=ex)
        img.draw()
        win.flip()
        core.wait(image_duration)
        win.flip()
        core.wait(inter_trial_interval)


@click.command()
@click.option("--stimuli_dir", type=str, default=VG_COCO_SELECTED_STIMULI_DIR)
def generate_psychopy_exp(
    stimuli_dir=VG_COCO_SELECTED_STIMULI_DIR,
    image_duration=1.0,
    inter_trial_interval=0.5,
    n_repetitions=3,
):
    """Load the selected stimuli and present them in a psychopy experiment.

    :param stimuli_dir: The directory containing the previously selected stimuli.
    :type stimuli_dir: str
    :param image_duration: The duration of the image presentation in seconds.
    :type image_duration: float
    :param inter_trial_interval: The inter-trial interval in seconds.
    :type inter_trial_interval: float
    :param n_repetitions: The number of repetitions of each stimuli.
    :type n_repetitions: int
    """
    # Load the stimuli using huggingface datasets
    stimuli_ds = load_from_disk(stimuli_dir)
    # Load the images from the URL in the dataset
    stimuli = []
    for example in tqdm(stimuli_ds, desc="Loading stimuli"):
        stimuli = stimuli + [Image.open(requests.get(example["vg_url"], stream=True).raw)] * n_repetitions
    # Shuffle the stimuli (deterministically)
    random.seed(42)
    random.shuffle(stimuli)

    # Create the psychopy window
    win = visual.Window(size=(800, 600), fullscr=False, allowGUI=True, color='white')


@click.group()
def cli() -> None:
    """Generate the psychopy experiment files for the fMRI experiment based on the chosen stimuli."""


if __name__ == "__main__":
    cli.add_command(generate_psychopy_exp)
    cli()

