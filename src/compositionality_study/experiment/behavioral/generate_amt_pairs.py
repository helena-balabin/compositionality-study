"""Generate pairings of text or image stimuli for the AMT experiment."""
import os
from itertools import product

import click
import pandas as pd

from compositionality_study.constants import AMT_DIR, VG_COCO_LOCAL_STIMULI_DIR


def create_all_stim_pairings(
    high_comp: pd.Series,
    low_comp: pd.Series,
) -> pd.DataFrame:
    """Create all possible pairings of high and low complexity stimuli.

    :param high_comp: Series with the high complexity stimuli
    :type high_comp: pd.Series
    :param low_comp: Series with the low complexity stimuli
    :type low_comp: pd.Series
    :return: Dataframe with all possible pairings
    :rtype: pd.DataFrame
    """
    pairings = pd.DataFrame(list(product(high_comp, low_comp)), columns=["A", "B"])
    pairings["high_complex_on"] = "left"
    # Swap the A,B columns for 50% of the cases
    swap_idx = pairings.sample(frac=0.5, random_state=42).index
    pairings.loc[swap_idx, ["A", "B"]] = pairings.loc[swap_idx, ["B", "A"]].values
    # Indicate that the high complexity stimulus is on the right for these cases
    pairings.loc[swap_idx, "high_complex_on"] = "right"

    # Shuffle the order of the pairings
    pairings = pairings.sample(frac=1, random_state=24).reset_index(drop=True)

    return pairings


@click.command()
@click.option("--filtered_stimuli_dir", type=str, default=VG_COCO_LOCAL_STIMULI_DIR)
@click.option("--output_dir", type=str, default=AMT_DIR)
def create_amt_files(
    filtered_stimuli_dir: str = VG_COCO_LOCAL_STIMULI_DIR,
    output_dir: str = AMT_DIR,
) -> pd.DataFrame:
    """Create the AMT files.

    :param filtered_stimuli_dir: Path to the directory with the filtered stimuli
    :type filtered_stimuli_dir: str
    :param output_dir: Path to the directory to save the AMT files to
    :type output_dir: str
    :return: Dataframe with the pairings for the AMT experiment
    :rtype: pd.DataFrame
    """
    # Load the stimuli overview file from the directory
    stimuli_df = pd.read_csv(os.path.join(filtered_stimuli_dir, "stimuli_text_and_im_paths.csv"))
    # Filter out any control stimuli, so stimuli that have "scrambled" in the complexity column
    stimuli_df = stimuli_df[~stimuli_df["complexity"].str.contains("scrambled")]
    text_stim_high_comp = stimuli_df[stimuli_df["complexity"].str.contains("high")]["text"]
    text_stim_low_comp = stimuli_df[stimuli_df["complexity"].str.contains("low")]["text"]
    img_stim_high_comp = stimuli_df[stimuli_df["complexity"].str.contains("high")]["img_path"]
    img_stim_low_comp = stimuli_df[stimuli_df["complexity"].str.contains("low")]["img_path"]

    # Create all text pairings between high and low complexity
    text_pairings = create_all_stim_pairings(text_stim_high_comp, text_stim_low_comp)

    # Create all image pairings between high and low complexity
    img_pairings = create_all_stim_pairings(img_stim_high_comp, img_stim_low_comp)

    # Save the pairings to a csv file
    text_pairings.to_csv(os.path.join(output_dir, "amt_text_pairings.csv"), index=False)
    img_pairings.to_csv(os.path.join(output_dir, "amt_img_pairings.csv"), index=False)
    # Split each pairing file into 8 files of 1/8th of the pairings
    for pairing, mod in zip([text_pairings, img_pairings], ["text", "img"]):
        pairing_split = [
            pairing.iloc[i : i + int(len(pairing) / 8)]
            for i in range(0, len(pairing), int(len(pairing) / 8))
        ]
        for i, split in enumerate(pairing_split):
            split.to_csv(os.path.join(output_dir, f"amt_{mod}_pairings_{i}.csv"), index=False)

    # Return the combined pairings
    return pd.concat([text_pairings, img_pairings])


@click.group()
def cli() -> None:
    """Create pairings of text or image stimuli for the AMT experiment."""


if __name__ == "__main__":
    cli.add_command(create_amt_files)
    cli()
