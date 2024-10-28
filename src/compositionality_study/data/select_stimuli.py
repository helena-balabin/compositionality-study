"""Select stimuli from the VG + COCO overlap dataset."""

import copy
import os
from typing import Any, Dict, List

import click
import datasets
import numpy as np
import pandas as pd
from datasets import load_from_disk
from loguru import logger
from PIL import Image
from pytesseract import image_to_string

from compositionality_study.constants import VG_COCO_PREP_ALL, VG_IMAGE_DIR
from compositionality_study.utils import get_image_aspect_ratio_from_local_path


def map_conditions(
    example: Dict[str, Any],
    text_feature: str = "parse_tree_depth",
    graph_feature: str = "coco_a_graph_depth",
    min_text_feature_depth: int = 5,
    max_text_feature_depth: int = 10,
    min_n_coco_a_actions: int = 3,
    max_n_coco_a_actions: int = 18,
) -> Dict:
    """Map the conditions for parametric text and image complexity.

    :param example: The hf dataset example to map the conditions to
    :type example: Dict[str, Any]
    :param text_feature: The text feature to parameterize with, defaults to "parse_tree_depth"
    :type text_feature: str
    :param graph_feature: The graph feature to parameterize with, defaults to "coco_a_graph_depth"
    :type graph_feature: str
    :param min_text_feature_depth: The minimum text feature depth to select stimuli for, defaults to 5
    :type min_text_feature_depth: int
    :param max_text_feature_depth: The maximum text feature depth to select stimuli for, defaults to 10
    :type max_text_feature_depth: int
    :param min_n_coco_a_actions: The minimum number of COCO actions to select stimuli for, defaults to 3
    :type min_n_coco_a_actions: int
    :param max_n_coco_a_actions: The maximum number of COCO actions to select stimuli for, defaults to 18
    :type max_n_coco_a_actions: int
    """
    # Instead of binary high/low, use the actual values normalized to [0,1]
    text_norm = (example[text_feature] - min_text_feature_depth) / (max_text_feature_depth - min_text_feature_depth)
    example["textual_complexity_param"] = text_norm

    img_norm = (example[graph_feature] - min_n_coco_a_actions) / (max_n_coco_a_actions - min_n_coco_a_actions)
    example["img_act_complexity_param"] = img_norm

    return example


@click.command()
@click.option("--vg_coco_preprocessed_dir", type=str, default=VG_COCO_PREP_ALL)
@click.option("--sent_len", type=int, default=9)
@click.option("--sent_len_tol", type=int, default=2)
@click.option("--verbs", type=bool, default=True)
@click.option("--img_comp", type=float, default=0.5)
@click.option("--img_comp_tol", type=float, default=0.15)
@click.option("--asp_min", type=float, default=1.0)
@click.option("--asp_max", type=float, default=2.0)
@click.option("--text_feature", type=str, default="parse_tree_depth")
@click.option("--graph_feature", type=str, default="coco_a_graph_depth")
@click.option("--image_quality_threshold", type=int, default=400)
@click.option("--filter_text_on_images", type=bool, default=False)
@click.option("--filter_by_person", type=bool, default=True)
@click.option("--n_stimuli", type=int, default=378)
def select_stimuli(
    vg_coco_preprocessed_dir: str = VG_COCO_PREP_ALL,
    sent_len: int = 9,
    sent_len_tol: int = 2,
    verbs: bool = True,
    img_comp: float = 0.5,
    img_comp_tol: float = 0.15,
    asp_min: float = 1.2,
    asp_max: float = 1.8,
    text_feature: str = "parse_tree_depth",
    graph_feature: str = "coco_a_graph_depth",
    image_quality_threshold: int = 400,
    filter_text_on_images: bool = False,
    filter_by_person: bool = True,
    n_stimuli: int = 378,
):
    """Select stimuli from the VG + COCO overlap dataset.

    :param vg_coco_preprocessed_dir: The preprocessed VG + COCO overlap dataset to select stimuli from, defaults to
        VG_COCO_PREP_ALL
    :type vg_coco_preprocessed_dir: str
    :param sent_len: The sentence length to control for (controlled variable
    :type sent_len: int
    :param sent_len_tol: The tolerance for the sentence length (+- sent_len_tol within sent_len)
    :type sent_len_tol: int
    :param verbs: Whether to control for verbs (controlled variable), i.e., only select captions with verbs,
        defaults to True
    :type verbs: bool
    :param img_comp: The image complexity to control for (controlled variable), defaults to 0.5
    :type img_comp: float
    :param img_comp_tol: The tolerance for the image complexity (+- img_comp_tol within img_comp), defaults to 0.15
    :type img_comp_tol: float
    :param asp_min: The min aspect ratio of the images to select stimuli for, defaults to 1.0
    :type asp_min: float
    :param asp_max: The max aspect ratio of the images to select stimuli for, defaults to 2.0
    :type asp_max: float
    :param text_feature: The text feature to parameterize with, defaults to "parse_tree_depth", alternatively one can
        use "amr_tree_depth"
    :type text_feature: str
    :param graph_feature: The graph feature to parameterize with, defaults to "coco_a_graph_depth", alternatively one
        can use "sg_depth"
    :type graph_feature: str
    :param image_quality_threshold: Minimum number of pixels (height) of the image, defaults to 400
    :type image_quality_threshold: int
    :param filter_text_on_images: Whether to filter out images with text on them, defaults to False
    :type filter_text_on_images: bool
    :param filter_by_person: Whether to filter out images that do not contain any people based on the COCO annotation
        data, defaults to True
    :type filter_by_person: bool
    :param n_stimuli: The number of stimuli to select
    :type n_stimuli: int
    """
    # Load the dataset
    vg_ds = load_from_disk(vg_coco_preprocessed_dir)
    file_prefix = "vg_" if "vg_" in vg_coco_preprocessed_dir else ""
    logger.info(f"Loaded the preprocessed VG + COCO overlap dataset with {len(vg_ds)} entries.")
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Calculate min/max depth values for the dependency parse tree depth and the COCO action graph depth
    dep_min = int(pd.Series(vg_ds[text_feature]).min())
    dep_max = int(pd.Series(vg_ds[text_feature]).max())
    ac_min = int(pd.Series(vg_ds[graph_feature]).min())
    ac_max = int(pd.Series(vg_ds[graph_feature]).max())

    # Filter by sentence length (within a tolerance)
    vg_ds = vg_ds.filter(
        lambda x: abs(x["sentence_length"] - sent_len) <= sent_len_tol,
        num_proc=24,
    )
    logger.info(
        f"Controlled the dataset for a sentence length of {sent_len} within a tolerance of {sent_len_tol}, "
        f"{len(vg_ds)} entries remain."
    )
    # Also filter by verbs if specified
    if verbs:
        vg_ds = vg_ds.filter(
            lambda x: x["n_verbs"] > 0,
            num_proc=24,
        )
        logger.info(f"Controlled the dataset for captions with verbs, {len(vg_ds)} entries remain.")

    if filter_by_person:
        vg_ds = vg_ds.filter(
            lambda x: x["coco_person"] >= 1,
            num_proc=24,
        )
        logger.info(f"Controlled the dataset for the number of people (at least 1), " f"{len(vg_ds)} entries remain.")

    # Filter by image complexity (within a tolerance)
    vg_ds = vg_ds.filter(
        lambda x: abs(x["ic_score"] - img_comp) <= img_comp_tol,
        num_proc=24,
    )
    logger.info(
        f"Controlled the dataset for an image complexity of {img_comp} within a tolerance of {img_comp_tol}, "
        f"{len(vg_ds)} entries remain."
    )

    # Also filter by aspect ratio of the images (e.g., only horizontal images)
    if "aspect_ratio" in vg_ds.features:
        vg_ds = vg_ds.filter(
            lambda x: asp_min <= x["aspect_ratio"] <= asp_max,
            num_proc=24,
        )
    else:
        vg_ds = vg_ds.filter(
            lambda x: asp_min <= get_image_aspect_ratio_from_local_path(x["filepath"]) <= asp_max,
            num_proc=24,
        )
    logger.info(
        f"Controlled the dataset for the aspect ratio of the images (between {asp_min} and {asp_max}), "
        f"{len(vg_ds)} entries remain."
    )

    # Filter out image duplicates
    vg_df = pd.DataFrame(vg_ds)
    vg_df = vg_df.drop_duplicates(subset=["filepath"])
    vg_ds = datasets.Dataset.from_pandas(vg_df)
    logger.info(f"Filtered out duplicate images, {len(vg_ds)} entries remain.")

    # Filter out black and white images and images with text on them and images with low quality
    # Add the images to the dataset, load based on the filenames
    # Avoid PIL-related bugs by copying the images
    vg_ds = vg_ds.map(
        lambda x: {
            **x,
            "img": copy.deepcopy(Image.open(os.path.join(VG_IMAGE_DIR, x["filepath"]))),
        },
        num_proc=24,
    )
    # Filter out images with low quality
    vg_ds = vg_ds.filter(
        lambda x: x["img"].size[0] > 1.5 * image_quality_threshold and x["img"].size[1] > image_quality_threshold,
        num_proc=24,
    )
    # Take a 20 x 20 patch and if it's equal across two RGB channels, it's a black and white image
    vg_ds = vg_ds.filter(
        # Take possibly faulty images into account
        lambda x: np.array(x["img"]).ndim > 2
        and not np.all(np.array(x["img"])[:20, :20, 0] == np.array(x["img"])[:20, :20, 1]),
        num_proc=24,
    )
    if filter_text_on_images:
        # Filter out images with text on them using pytesseract (make sure to install it first, see
        # https://tesseract-ocr.github.io/tessdoc/Installation.html)
        vg_ds = vg_ds.filter(
            lambda x: len(image_to_string(x["img"]).strip(" \n\x0c")) == 0,
            num_proc=8,  # Lower parallelization because of pytesseract
        )
    logger.info(
        f"Filtered out black and white images, images with text and low quality images, {len(vg_ds)} "
        f"entries remain."
    )

    # Map the conditions to the examples
    vg_ds = vg_ds.map(
        lambda x: map_conditions(
            x,
            text_feature=text_feature,
            min_text_feature_depth=dep_min,
            max_text_feature_depth=dep_max,
            min_n_coco_a_actions=ac_min,
            max_n_coco_a_actions=ac_max,
        ),
        num_proc=24,
    )

    # Instead of covering the full 2D grid, select points along the diagonal
    vg_n_stim: List = []

    # Create evenly spaced points along the diagonal from (0,0) to (1,1)
    complexity_points = np.linspace(0, 1, n_stimuli)
    for target_complexity in complexity_points:
        # Calculate distance to target point for all examples
        distances = np.sqrt(
            (np.array(vg_ds["textual_complexity_param"]) - target_complexity) ** 2
            + (np.array(vg_ds["img_act_complexity_param"]) - target_complexity) ** 2
        )

        # Find available indices that haven't been used yet
        available_indices = [
            i for i in range(len(vg_ds)) if vg_ds[i]["vg_image_id"] not in [x["vg_image_id"] for x in vg_n_stim]
        ]

        if not available_indices:
            logger.warning(f"Not enough stimuli for complexity level {target_complexity}")
            continue

        # Normalize distances to [0,1] for available indices to make them comparable with correlation impacts
        # Without normalization, the raw distances could dominate the combined score since they're on a different scale
        # TODO double check if this makes sense
        distances_norm = distances[available_indices]
        distances_norm = (distances_norm - np.min(distances_norm)) / (np.max(distances_norm) - np.min(distances_norm))

        # Calculate correlation impact for each candidate
        correlation_impacts = []
        for idx in available_indices:
            temp_stim = vg_n_stim + [vg_ds[idx]]
            if len(temp_stim) < 2:
                correlation_impacts.append(0)
                continue

            people = [x["coco_person"] for x in temp_stim]  # type: ignore
            text_comp = [x["textual_complexity_param"] for x in temp_stim]  # type: ignore
            img_comp = [x["img_act_complexity_param"] for x in temp_stim]  # type: ignore

            # TODO double check if this makes sense
            corr1 = abs(np.corrcoef(people, text_comp)[0, 1])
            corr2 = abs(np.corrcoef(people, img_comp)[0, 1])
            correlation_impacts.append((corr1 + corr2) / 2)

        correlation_impacts = np.array(correlation_impacts)

        # TODO double check if this makes sense
        # Combine both metrics with equal weight
        combined_score = distances_norm + correlation_impacts

        # Select the example with the lowest combined score
        best_match_idx = available_indices[np.argmin(combined_score)]
        vg_n_stim.append(vg_ds[best_match_idx])

    vg_ds_n_stimuli = datasets.Dataset.from_dict(
        {k: [example[k] for example in vg_n_stim] for k in vg_ds.features.keys()}
    )

    # If the "__index_level_0__" column exists, drop it
    if "__index_level_0__" in vg_ds_n_stimuli.features:
        vg_ds_n_stimuli = vg_ds_n_stimuli.remove_columns(["__index_level_0__"])

    # As a sanity check: Check that the number of people is not correlated with the complexities
    # Get the correlation between the number of people and the complexities
    corr_text: float = float(
        np.corrcoef(np.array(vg_ds_n_stimuli["coco_person"]), np.array(vg_ds_n_stimuli["textual_complexity_param"]))[
            0, 1
        ]
    )
    corr_img: float = float(
        np.corrcoef(np.array(vg_ds_n_stimuli["coco_person"]), np.array(vg_ds_n_stimuli["img_act_complexity_param"]))[
            0, 1
        ]
    )
    logger.info(f"Correlation between the number of people and the textual complexity: {corr_text}")
    logger.info(f"Correlation between the number of people and the image complexity: {corr_img}")

    # Save the dataset
    output_dir = os.path.split(vg_coco_preprocessed_dir)[0]
    vg_ds_n_stimuli.save_to_disk(
        os.path.join(
            output_dir,
            f"{file_prefix}coco_{n_stimuli}_stimuli_parametric",
        )
    )


@click.group()
def cli() -> None:
    """Select stimuli from a COCO-based dataset."""


if __name__ == "__main__":
    cli.add_command(select_stimuli)
    cli()
