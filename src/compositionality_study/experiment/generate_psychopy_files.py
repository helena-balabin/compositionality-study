"""Generate the psychopy experiment files for the fMRI experiment based on the chosen stimuli."""
import os
import random
from typing import List

import click
import pandas as pd
from PIL import Image, ImageOps
from psychopy import core, visual
from tqdm import tqdm

from compositionality_study.constants import VG_COCO_LOCAL_STIMULI_DIR

# Set a random seed for reproducibility
random.seed(42)


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
    stimuli: List = []
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
    win = visual.Window(
        size=(1600, 1200),
        fullscr=True,
        allowGUI=True,
        color="white",
    )

    # Run the experiment
    run_images(
        win=win,
        stimuli=stimuli,
        image_duration=image_duration,
        inter_trial_interval=inter_trial_interval,
    )


@click.group()
def cli() -> None:
    """Generate the psychopy experiment files for the fMRI experiment based on the local chosen stimuli."""


if __name__ == "__main__":
    cli.add_command(generate_psychopy_exp)
    cli()
