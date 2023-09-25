"""Generate the psychopy experiment files for the fMRI experiment based on the chosen stimuli."""
import os
import random
from typing import List

import click
import pandas as pd
from PIL import Image, ImageOps
from psychopy import core, event, logging, visual
from tqdm import tqdm

from compositionality_study.constants import VG_COCO_LOCAL_STIMULI_DIR, VG_COCO_PRACTICE_STIMULI_DIR

# Set a random seed for reproducibility
random.seed(42)
# Set the logging level
logging.console.setLevel(logging.INFO)


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
    inter_stimulus_interval: float = 2.0,
    dummy_scan_duration: float = 6.0,
    frame_rate: int = 60,
    mri_trigger_button: str = "t",
    manual_trigger_button: str = "space",
):
    """Present a given stimuli list in a psychopy window.

    :param win: The psychopy window.
    :type win: visual.Window
    :param stimuli: The stimuli to present (either visual or textual).
    :type stimuli: List
    :param duration: The duration of the text/image presentation in seconds.
    :type duration: float
    :param inter_stimulus_interval: The inter-trial interval in seconds.
    :type inter_stimulus_interval: float
    :param dummy_scan_duration: The duration of the dummy scans in seconds.
    :type dummy_scan_duration: float
    :param frame_rate: The frame rate of the psychopy window in Hz.
    :type frame_rate: int
    :param mri_trigger_button: The MRI trigger button that starts each run.
    :type mri_trigger_button: str
    :param manual_trigger_button: The manual trigger button that starts each run.
    :type manual_trigger_button: str
    """
    # Create either TextStim or ImageStim objects
    stim_objects = []
    for stim in stimuli:
        if isinstance(stim, Image.Image):
            stim_objects.append(visual.ImageStim(win, image=stim))
        else:
            stim_objects.append(visual.TextStim(win, text=stim, color=(0.0, 0.0, 0.0)))

    # Get the number of frames for time durations in seconds
    n_frames_stimuli = int(duration * frame_rate)
    n_frames_inter_stimulus_interval = int(inter_stimulus_interval * frame_rate)
    n_frames_dummy_scan_duration = int(dummy_scan_duration * frame_rate)

    # Start screen
    start_text = visual.TextStim(
        win,
        text=f"Press {manual_trigger_button} to start \n (before starting the MRI sequence)",
        color=(0.0, 0.0, 0.0),
    )
    # Wait for the input
    while True:
        start_text.draw()
        win.flip()
        keys = event.getKeys()
        if manual_trigger_button in keys:
            break
        elif "escape" in keys:
            core.quit()

    # Wait for the MRI trigger
    mri_trigger_text = visual.TextStim(
        win,
        text="Waiting for the trigger to start",
        color=(0.0, 0.0, 0.0),
    )
    # Wait for the experiment to start
    while True:
        mri_trigger_text.draw()
        win.flip()
        keys = event.getKeys()
        if mri_trigger_button in keys:
            break
        elif "escape" in keys:
            core.quit()

    # Have a blank window for the duration of the dummy scans
    for _ in range(n_frames_dummy_scan_duration):
        # Check if the experiment should be exited or the stimuli skipped
        if check_quit_skip_exp():
            break

    # Have a global timer to log the time
    timer = core.Clock()
    start_time = timer.getTime()

    # Create the visual or textual stimuli
    for ex in stim_objects:
        # Show the stimuli for the duration
        for _ in range(n_frames_stimuli):
            ex.draw()
            win.flip()
            # Check if the experiment should be exited or the stimuli skipped
            if check_quit_skip_exp():
                break

        # Wait for the inter stimulus interval
        for _ in range(n_frames_inter_stimulus_interval):
            win.flip()
            # Check if the experiment should be exited or the stimuli skipped
            if check_quit_skip_exp():
                break

    end_time = timer.getTime()
    logging.log(f"Total time: {end_time - start_time}", level=logging.INFO)


@click.command()
@click.option("--local_stimuli_dir", type=str, default=VG_COCO_LOCAL_STIMULI_DIR)
@click.option("--practice_stimuli_dir", type=str, default=VG_COCO_PRACTICE_STIMULI_DIR)
@click.option("--duration", type=float, default=6.0)
@click.option("--inter_stimulus_interval", type=float, default=2.0)
@click.option("--dummy_scan_duration", type=float, default=6.0)
@click.option("--n_repetitions", type=int, default=3)
@click.option("--n_runs", type=int, default=16)
@click.option("--frame_rate", type=int, default=60)
@click.option("--fullscreen", type=bool, default=False)
@click.option("--manual_trigger_button", type=str, default="space")
@click.option("--mri_trigger_button", type=str, default="t")
@click.option("--break_duration", type=float, default=30.0)
def run_psychopy_exp(
    local_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    practice_stimuli_dir: str = VG_COCO_PRACTICE_STIMULI_DIR,
    duration: float = 6.0,
    inter_stimulus_interval: float = 2.0,
    dummy_scan_duration: float = 6.0,
    n_repetitions: int = 3,
    n_runs: int = 16,
    frame_rate: int = 60,
    fullscreen: bool = False,
    manual_trigger_button: str = "space",
    mri_trigger_button: str = "t",
    break_duration: float = 30.0,
):
    """Load the selected stimuli and present them in a psychopy experiment.

    :param local_stimuli_dir: The directory containing the previously selected stimuli (converted from the HF dataset).
    :type local_stimuli_dir: str
    :param practice_stimuli_dir: The directory containing the practice stimuli.
    :type practice_stimuli_dir: str
    :param duration: The duration of the image/text presentation in seconds.
    :type duration: float
    :param inter_stimulus_interval: The interval between stimuli in seconds.
    :type inter_stimulus_interval: float
    :param dummy_scan_duration: The duration of the dummy scans in seconds.
    :type dummy_scan_duration: float
    :param n_repetitions: The number of repetitions of each stimuli.
    :type n_repetitions: int
    :param n_runs: The number of runs to present the stimuli.
    :type n_runs: int
    :param frame_rate: The frame rate of the psychopy window in Hz.
    :type frame_rate: int
    :param fullscreen: Whether to run the experiment in fullscreen mode.
    :type fullscreen: bool
    :param manual_trigger_button: The manual trigger button that starts the experiment.
    :type manual_trigger_button: str
    :param mri_trigger_button: The MRI trigger button that starts each run.
    :type mri_trigger_button: str
    :param break_duration: The duration of the break after every two runs in seconds.
    :type break_duration: float
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

    # Load the practice stimuli
    practice_stimuli: List = []
    practice_stimuli_df = pd.read_csv(
        os.path.join(practice_stimuli_dir, "stimuli_text_and_im_paths.csv"),
    )
    for _, row in tqdm(practice_stimuli_df.iterrows(), desc="Loading practice stimuli"):
        # Load the image from disk and resize it
        img = Image.open(os.path.join(practice_stimuli_dir, row["img_path"]))
        # Add the image to the stimuli list
        practice_stimuli = practice_stimuli + [ImageOps.contain(img, (800, 600))]
        # Add the text to the stimuli list
        practice_stimuli = practice_stimuli + [row["text"]]

    # Shuffle the practice stimuli
    random.shuffle(practice_stimuli)

    # Create the psychopy window
    win = visual.Window(
        size=(1200, 800),
        fullscr=fullscreen,
        allowGUI=True,
        color="white",
    )

    # First do a practice run
    run_single_run(
        win=win,
        stimuli=practice_stimuli,
        duration=duration,
        inter_stimulus_interval=inter_stimulus_interval,
        dummy_scan_duration=1.0,
        frame_rate=frame_rate,
        mri_trigger_button=mri_trigger_button,
        manual_trigger_button=manual_trigger_button,
    )

    # Then do the actual runs: Show the stimuli for each run
    for idx, stimuli_run in enumerate(stimuli_runs):
        # Run a single run
        run_single_run(
            win=win,
            stimuli=stimuli_run,
            duration=duration,
            inter_stimulus_interval=inter_stimulus_interval,
            dummy_scan_duration=dummy_scan_duration,
            frame_rate=frame_rate,
            mri_trigger_button=mri_trigger_button,
            manual_trigger_button=manual_trigger_button,
        )
        # Have a break every two runs
        if idx % 2 == 1:
            break_txt = visual.TextStim(
                win,
                text=f"Break for {int(break_duration)}s",
                color=(0.0, 0.0, 0.0),
            )
            break_frames = int(break_duration * frame_rate)
            for _ in range(break_frames):
                break_txt.draw()
                win.flip()
                # Check if the experiment should be exited or the stimuli skipped
                if check_quit_skip_exp():
                    break


@click.group()
def cli() -> None:
    """Run the psychopy experiment files for the fMRI experiment based on the local chosen stimuli."""


if __name__ == "__main__":
    cli.add_command(run_psychopy_exp)
    cli()
