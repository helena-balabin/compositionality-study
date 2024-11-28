"""Select stimuli from the VG + COCO overlap dataset."""

import copy
import os
from collections import Counter
from typing import Any, Dict

import click
import datasets
import numpy as np
import pandas as pd
from datasets import load_from_disk
from loguru import logger
from PIL import Image
from pytesseract import image_to_string
from scipy.stats import chi2_contingency, spearmanr

from compositionality_study.constants import VG_COCO_PREP_ALL, VG_IMAGE_DIR
from compositionality_study.utils import get_image_aspect_ratio_from_local_path


def map_conditions(
    example: Dict[str, Any],
    text_feature: str = "amr_graph_depth",
    graph_feature: str = "coco_a_graph_depth",
    min_text_feature_depth: int = 5,
    max_text_feature_depth: int = 10,
    min_n_coco_a_actions: int = 3,
    max_n_coco_a_actions: int = 18,
) -> Dict:
    """Map the conditions for parametric text and image complexity.

    :param example: The hf dataset example to map the conditions to
    :type example: Dict[str, Any]
    :param text_feature: The text feature to parameterize with, defaults to "amr_graph_depth"
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
@click.option("--sent_len", type=int, default=10)
@click.option("--sent_len_tol", type=int, default=2)
@click.option("--verbs", type=bool, default=False)
@click.option("--img_comp", type=float, default=0.5)
@click.option("--img_comp_tol", type=float, default=0.2)
@click.option("--asp_min", type=float, default=1.0)
@click.option("--asp_max", type=float, default=2.0)
@click.option("--text_feature", type=str, default="amr_graph_depth")
@click.option("--graph_feature", type=str, default="coco_a_graph_depth")
@click.option("--image_quality_threshold", type=int, default=400)
@click.option("--filter_text_on_images", type=bool, default=False)
@click.option("--filter_by_person", type=bool, default=True)
@click.option("--person_count", type=int, default=5)
@click.option("--person_tol", type=int, default=2)
@click.option("--n_stimuli", type=int, default=252)
@click.option("--buffer_stimuli_fraction", type=float, default=0.2)
def select_stimuli(
    vg_coco_preprocessed_dir: str = VG_COCO_PREP_ALL,
    sent_len: int = 10,
    sent_len_tol: int = 2,
    verbs: bool = False,
    img_comp: float = 0.5,
    img_comp_tol: float = 0.2,
    asp_min: float = 1.0,
    asp_max: float = 2.0,
    text_feature: str = "amr_graph_depth",
    graph_feature: str = "coco_a_graph_depth",
    image_quality_threshold: int = 400,
    filter_text_on_images: bool = False,
    filter_by_person: bool = True,
    person_count: int = 5,
    person_tol: int = 2,
    n_stimuli: int = 252,
    buffer_stimuli_fraction: float = 0.2,
):
    """Select stimuli from the VG + COCO overlap dataset.

    :param vg_coco_preprocessed_dir: The preprocessed VG + COCO overlap dataset to select stimuli from, defaults to
        VG_COCO_PREP_ALL
    :type vg_coco_preprocessed_dir: str
    :param sent_len: The sentence length to control for (controlled variable), defaults to 10
    :type sent_len: int
    :param sent_len_tol: The tolerance for the sentence length (+- sent_len_tol within sent_len), defaults to 2
    :type sent_len_tol: int
    :param verbs: Whether to control for verbs (controlled variable), i.e., only select captions with verbs,
        defaults to False
    :type verbs: bool
    :param img_comp: The image complexity to control for (controlled variable), defaults to 0.5
    :type img_comp: float
    :param img_comp_tol: The tolerance for the image complexity (+- img_comp_tol within img_comp), defaults to 0.2
    :type img_comp_tol: float
    :param asp_min: The min aspect ratio of the images to select stimuli for, defaults to 1.0
    :type asp_min: float
    :param asp_max: The max aspect ratio of the images to select stimuli for, defaults to 2.0
    :type asp_max: float
    :param text_feature: The text feature to parameterize with, defaults to "amr_graph_depth", alternatively one can
        use "parse_tree_depth"
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
    :param person_count: The number of people to filter for, defaults to 3
    :type person_count: int
    :param person_tol: The tolerance for the number of people (+- person_tol within person_count), defaults to 2
    :type person_tol: int
    :param n_stimuli: The number of stimuli to select
    :type n_stimuli: int
    :param buffer_stimuli_fraction: The fraction of stimuli to buffer for the parametric stimuli selection, defaults to
        0.2, meaning that in total n_stimuli * (1 + buffer_stimuli_fraction) stimuli will be selected
    """
    # Apply a buffer to the number of stimuli to select
    n_stimuli = int(n_stimuli * (1 + buffer_stimuli_fraction))
    # Make sure it's an even number
    n_stimuli = n_stimuli if n_stimuli % 2 == 0 else n_stimuli + 1
    logger.info(f"Selecting {n_stimuli} stimuli (including buffer).")

    # Load the dataset
    vg_ds = load_from_disk(vg_coco_preprocessed_dir)
    file_prefix = "vg_" if "vg_" in vg_coco_preprocessed_dir else ""
    logger.info(f"Loaded the preprocessed VG + COCO overlap dataset with {len(vg_ds)} entries.")
    # Set a random seed for reproducibility
    np.random.seed(42)

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

    # Filter by person based on the desired number of people in the dataset
    if filter_by_person:
        vg_ds = vg_ds.filter(
            lambda x: np.abs(person_count - x["coco_person"]) <= person_tol,
            num_proc=24,
        )
        logger.info(
            f"Controlled the dataset for the number of people {person_count} +- {person_tol},"
            f"{len(vg_ds)} entries remain."
        )

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

    # Pre-compute arrays for faster access
    text_complexities = np.array(vg_ds[text_feature])
    img_complexities = np.array(vg_ds[graph_feature])
    image_ids = np.array(vg_ds["imgid"])

    # Create target complexity points (1,1) and (2,2)
    target_complexities = [(1, 1), (2, 2)]
    n_each = n_stimuli // len(target_complexities)

    vg_n_stim = []
    for target_text, target_img in target_complexities:
        # Filter for the target complexities
        filtered_indices = np.where((text_complexities == target_text) & (img_complexities == target_img))[0]

        if len(filtered_indices) < n_each:
            raise ValueError(f"Not enough stimuli with complexities ({target_text}, {target_img}).")

        # Exclude duplicates based on imgid
        unique_imgids = set()
        unique_filtered_indices = []
        for idx in filtered_indices:
            imgid = image_ids[idx]
            if imgid not in unique_imgids:
                unique_imgids.add(imgid)
                unique_filtered_indices.append(idx)

        if len(unique_filtered_indices) < n_each:
            raise ValueError(f"Not enough unique stimuli with complexities ({target_text}, {target_img}).")

        selected_indices = np.random.choice(unique_filtered_indices, n_each, replace=False)

        for idx in selected_indices:
            vg_n_stim.append(vg_ds[int(idx)])

    vg_ds_n_stimuli = datasets.Dataset.from_dict(
        {k: [example[k] for example in vg_n_stim] for k in vg_ds.features.keys()}
    )
    # Assert that the number of unique image ids is equal to the number of stimuli
    assert len(set(vg_ds_n_stimuli["imgid"])) == n_stimuli

    # Check that the number of people is not correlated with the complexities using a chi-square test
    # Create contingency tables for the chi-square test
    text_contingency_table = pd.crosstab(vg_ds_n_stimuli["coco_person"], vg_ds_n_stimuli[text_feature])
    img_contingency_table = pd.crosstab(vg_ds_n_stimuli["coco_person"], vg_ds_n_stimuli[graph_feature])

    # Perform chi-square test
    chi2_text, p_text, _, _ = chi2_contingency(text_contingency_table)
    chi2_img, p_img, _, _ = chi2_contingency(img_contingency_table)

    logger.info(
        f"Chi-square test between the number of people and the textual complexity: chi2={chi2_text}, p-value={p_text}"
    )
    logger.info(
        f"Chi-square test between the number of people and the image complexity: chi2={chi2_img}, p-value={p_img}"
    )

    # Another check: Look at the possible (unique) categories in coco_categories
    # and see that they are not correlated with the complexities
    # 1. Get all the unique categories
    coco_categories = vg_ds_n_stimuli["coco_categories"]
    coco_categories = [cat for sublist in coco_categories for cat in sublist if cat != "person"]
    coco_categories = np.array(coco_categories)
    unique_coco_categories = np.unique(coco_categories)
    category_counter = {cat: 0 for cat in unique_coco_categories}

    # 2. For each category, get a binary absence/presence indicator
    # and correlate it with the complexities
    for cat in unique_coco_categories:
        cat_indicator = [1 if cat in x["coco_categories"] else 0 for x in vg_ds_n_stimuli]
        category_counter[cat] = np.sum(cat_indicator)
        corr_text, p_text = spearmanr(cat_indicator, vg_ds_n_stimuli[text_feature])
        corr_img, p_img = spearmanr(cat_indicator, vg_ds_n_stimuli[graph_feature])
        logger.info(f"Correlation between the presence of category {cat} and the textual complexity: {corr_text}")
        logger.info(f"Text correlation p-value: {p_text}")
        logger.info(f"Correlation between the presence of category {cat} and the image complexity: {corr_img}")
        logger.info(f"Image correlation p-value: {p_img}")

    # Also get a counter on the image complexity and the textual complexity values
    logger.info(f"Textual complexity values: {Counter(vg_ds_n_stimuli[text_feature])}")
    logger.info(f"Image complexity values: {Counter(vg_ds_n_stimuli[graph_feature])}")

    # Save the dataset
    output_dir = os.path.split(vg_coco_preprocessed_dir)[0]
    vg_ds_n_stimuli.save_to_disk(
        os.path.join(
            output_dir,
            f"{file_prefix}coco_{n_stimuli}_stimuli",
        )
    )


@click.group()
def cli() -> None:
    """Select stimuli from a COCO-based dataset."""


if __name__ == "__main__":
    cli.add_command(select_stimuli)
    cli()
