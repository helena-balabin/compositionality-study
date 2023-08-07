"""Generate the psychopy experiment files for the fMRI experiment based on the chosen stimuli."""
import os
import random
from typing import List

import click
import pandas as pd
from PIL import Image, ImageOps
from psychopy import core, event, visual
from tqdm import tqdm

from compositionality_study.constants import VG_COCO_LOCAL_STIMULI_DIR

# Set a random seed for reproducibility
random.seed(42)


def check_quit_skip_exp() -> bool:
    """Check if the experiment should be exited or whether the stimulus should be skipped (press space/right arrow).

    :return: Return True if the stimulus should be skipped.
    :rtype: bool
    """
    keys = event.getKeys()
    if "escape" in keys:
        core.quit()
        return False
    elif "space" in keys or "right" in keys:
        return True
    else:
        return False


def run_single_run(
    win: visual.Window,
    stimuli: List,
    duration: float = 6.0,
    inter_trial_interval: float = 2.0,
    frame_rate: int = 60,
    trigger_button: str = "t",
):
    """Present a given stimuli list in a psychopy window.

    :param win: The psychopy window.
    :type win: visual.Window
    :param stimuli: The stimuli to present (either visual or textual).
    :type stimuli: List
    :param duration: The duration of the text/image presentation in seconds.
    :type duration: float
    :param inter_trial_interval: The inter-trial interval in seconds.
    :type inter_trial_interval: float
    :param frame_rate: The frame rate of the psychopy window in Hz.
    :type frame_rate: int
    :param trigger_button: The trigger button to press to start the experiment.
    :type trigger_button: str
    """
    # Get the number of frames to present the stimuli and the inter-trial interval
    n_frames_stimuli = int(duration * frame_rate)
    n_frames_inter_trial_interval = int(inter_trial_interval * frame_rate)

    # Have a welcome screen
    welcome_txt = visual.TextStim(
        win,
        text=f"Press {trigger_button} to start",
        color=(0.0, 0.0, 0.0),
    )

    # Wait for the experiment to start
    while True:
        welcome_txt.draw()
        win.flip()
        keys = event.getKeys()
        if trigger_button in keys:
            break
        elif "escape" in keys:
            core.quit()

    # Create the visual or textual stimuli
    for ex in stimuli:
        # Show the stimuli for the duration (number of frames)
        for _ in range(n_frames_stimuli):
            if isinstance(ex, Image.Image):
                img = visual.ImageStim(win, image=ex)
                img.draw()
            else:
                txt = visual.TextStim(win, text=ex, color=(0.0, 0.0, 0.0))
                txt.draw()
            win.flip()
            # Check if the experiment should be exited or the stimuli skipped
            if check_quit_skip_exp():
                break

        # Wait for the inter-trial interval (number of frames)
        for _ in range(n_frames_inter_trial_interval):
            win.flip()
            # Check if the experiment should be exited or the stimuli skipped
            if check_quit_skip_exp():
                break


@click.command()
@click.option("--local_stimuli_dir", type=str, default=VG_COCO_LOCAL_STIMULI_DIR)
@click.option("--duration", type=float, default=6.0)
@click.option("--inter_trial_interval", type=float, default=2.0)
@click.option("--n_repetitions", type=int, default=3)
@click.option("--n_runs", type=int, default=8)
@click.option("--frame_rate", type=int, default=60)
@click.option("--fullscreen", type=bool, default=False)
@click.option("--trigger_button", type=str, default="t")
def run_psychopy_exp(
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    duration: float = 6.0,
    inter_trial_interval: float = 2.0,
    n_repetitions: int = 3,
    n_runs: int = 8,
    frame_rate: int = 60,
    fullscreen: bool = False,
    trigger_button: str = "t",
):
    """Load the selected stimuli and present them in a psychopy experiment.

    :param local_stimuli_dir: The directory containing the previously selected stimuli (converted from the HF dataset).
    :type local_stimuli_dir: str
    :param duration: The duration of the image/text presentation in seconds.
    :type duration: float
    :param inter_trial_interval: The inter-trial interval in seconds.
    :type inter_trial_interval: float
    :param n_repetitions: The number of repetitions of each stimuli.
    :type n_repetitions: int
    :param n_runs: The number of runs to present the stimuli.
    :type n_runs: int
    :param frame_rate: The frame rate of the psychopy window in Hz.
    :type frame_rate: int
    :param fullscreen: Whether to run the experiment in fullscreen mode.
    :type fullscreen: bool
    :param trigger_button: The trigger button to press to start the experiment.
    :type trigger_button: str
    :raises ValueError: If the number of stimuli is not divisible by the number of runs.
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

    # Make sure the number of stimuli is divisible by the number of runs, throw an error otherwise
    if len(stimuli) % n_runs != 0:
        raise ValueError(
            f"{len(stimuli)} stimuli (including {n_repetitions} repetitions) are not divisible into {n_runs} runs."
        )
    # Split the stimuli into runs
    stimuli_runs = [stimuli[i::n_runs] for i in range(n_runs)]

    # Create the psychopy window
    win = visual.Window(
        size=(1200, 800),
        fullscr=fullscreen,
        allowGUI=True,
        color="white",
    )

    # Show the stimuli for each run
    for stimuli_run in stimuli_runs:
        # Run a single run
        run_single_run(
            win=win,
            stimuli=stimuli_run,
            duration=duration,
            inter_trial_interval=inter_trial_interval,
            frame_rate=frame_rate,
            trigger_button=trigger_button,
        )


@click.group()
def cli() -> None:
    """Run the psychopy experiment files for the fMRI experiment based on the local chosen stimuli."""


if __name__ == "__main__":
    cli.add_command(run_psychopy_exp)
    cli()
