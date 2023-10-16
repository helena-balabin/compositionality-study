"""Select stimuli from the VG + COCO overlap dataset."""
import copy
# Imports
import os
from typing import Any, Dict

import click
import numpy as np
import pandas as pd
from PIL import Image
from datasets import concatenate_datasets, load_from_disk
from loguru import logger
from pytesseract import image_to_string

from compositionality_study.constants import VG_COCO_PREP_ALL, VG_IMAGE_DIR
from compositionality_study.utils import get_image_aspect_ratio_from_local_path


def map_conditions(
    example: Dict[str, Any],
    min_dep_parse_tree_depth: int = 5,
    max_dep_parse_tree_depth: int = 10,
    min_sg_filtered_depth: int = 3,
    max_sg_filtered_depth: int = 18,
) -> Dict:
    """Map the conditions (high/low textual complexity, high/low visual complexity) to the example.

    :param example: The hf dataset example to map the conditions to (high/low textual complexity,
        high/low visual complexity)
    :type example: Dict[str, Any]
    :param min_dep_parse_tree_depth: The minimum dependency parse tree depth
    :type min_dep_parse_tree_depth: int
    :param max_dep_parse_tree_depth: The maximum dependency parse tree depth
    :type max_dep_parse_tree_depth: int
    :param min_sg_filtered_depth: The minimum scene graph depth
    :type min_sg_filtered_depth: int
    :param max_sg_filtered_depth: The maximum scene graph depth
    :type max_sg_filtered_depth: int
    :return: The example with the mapped conditions, i.e., extra features for the conditions
    :rtype: Dict
    """
    # Map the textual complexity condition
    middle_dep_parse_tree_depth = (min_dep_parse_tree_depth + max_dep_parse_tree_depth) / 2
    textual_complexity = "high" if example["parse_tree_depth"] >= middle_dep_parse_tree_depth else "low"
    example["textual_complexity"] = textual_complexity

    # Map the visual complexity condition
    middle_sg_filtered_depth = (min_sg_filtered_depth + max_sg_filtered_depth) / 2
    img_seg_complexity = "high" if example["sg_filtered_depth"] >= middle_sg_filtered_depth else "low"
    example["img_act_complexity"] = img_seg_complexity

    # Combine the two
    example["complexity"] = f"{textual_complexity}_text_{img_seg_complexity}_image"

    return example


@click.command()
@click.option("--vg_coco_preprocessed_dir", type=str, default=VG_COCO_PREP_ALL)
@click.option("--sent_len", type=int, default=15)
@click.option("--sent_len_tol", type=int, default=2)
@click.option("--verbs", type=bool, default=True)
@click.option("--img_comp", type=float, default=0.5)
@click.option("--img_comp_tol", type=float, default=0.15)
@click.option("--asp_min", type=float, default=1.2)
@click.option("--asp_max", type=float, default=1.8)
@click.option("--dep_quantile", type=float, default=0.01)
@click.option("--sg_filtered_depth_quantile", type=float, default=0.05)
@click.option("--filter_outliers", type=bool, default=True)
@click.option("--image_quality_threshold", type=int, default=400)
@click.option("--filter_text_on_images", type=bool, default=True)
@click.option("--filter_by_animal_person", type=bool, default=True)
@click.option("--binarized_conditions", type=bool, default=True)
@click.option("--n_stimuli", type=int, default=80)
def select_stimuli(
    vg_coco_preprocessed_dir: str = VG_COCO_PREP_ALL,
    sent_len: int = 15,
    sent_len_tol: int = 2,
    verbs: bool = True,
    img_comp: float = 0.5,
    img_comp_tol: float = 0.15,
    asp_min: float = 1.2,
    asp_max: float = 1.8,
    dep_quantile: float = 0.01,
    sg_filtered_depth_quantile: float = 0.05,
    filter_outliers: bool = True,
    image_quality_threshold: int = 400,
    filter_text_on_images: bool = True,
    filter_by_animal_person: bool = True,
    binarized_conditions: bool = True,
    n_stimuli: int = 80,
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
    :param asp_min: The min aspect ratio of the images to select stimuli for, defaults to 1.2
    :type asp_min: float
    :param asp_max: The max aspect ratio of the images to select stimuli for, defaults to 1.8
    :type asp_max: float
    :param dep_quantile: The quantile of the dependency parse tree depth to select stimuli for, e.g., 0.05 means
        that the stimuli with the lowest 5% and highest 5% dependency parse tree depth are selected, defaults to 0.05
    :type dep_quantile: float
    :param sg_filtered_depth_quantile: The quantile of the number of filtered verbs to select stimuli for, e.g., 0.05
        means that the stimuli with the lowest 5% and highest 5% number of sg depths are selected, defaults to 0.05
    :type sg_filtered_depth_quantile: float
    :param filter_outliers: Whether to filter out outliers (more than 3x of the standard deviation) for the dependency
        parse tree depth and number of filtered verbs, defaults to True
    :type filter_outliers: bool
    :param image_quality_threshold: Minimum number of pixels (height) of the image, defaults to 400
    :type image_quality_threshold: int
    :param filter_text_on_images: Whether to filter out images with text on them, defaults to True
    :type filter_text_on_images: bool
    :param filter_by_animal_person: Whether to filter out images that do not contain any animals or people based on the
        COCO annotation data, defaults to True
    :type filter_by_animal_person: bool
    :param binarized_conditions: Whether to use binarized conditions (only high img + high text complexity, low img +
        low text complexity, rather than all 4 high/low combinations), defaults to True
    :param n_stimuli: The number of stimuli to select, must be divisible by 4
    :type n_stimuli: int
    """
    # Load the dataset
    vg_ds = load_from_disk(vg_coco_preprocessed_dir)
    logger.info(f"Loaded the preprocessed VG + COCO overlap dataset with {len(vg_ds)} entries.")
    # Set a random seed for reproducibility
    np.random.seed(42)

    # 1. Filter by sentence length (within a tolerance)
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

    # 2. Filter by image complexity (within a tolerance)
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
    logger.info(f"Controlled the dataset for the aspect ratio of the images (between {asp_min} and {asp_max}), "
                f"{len(vg_ds)} entries remain.")

    # 3. Filter out any outliers for dep parse tree depth and number of filtered verbs
    if filter_outliers:
        dep_m = pd.Series(vg_ds["parse_tree_depth"]).mean()
        dep_std = pd.Series(vg_ds["parse_tree_depth"]).std()
        vg_ds = vg_ds.filter(
            lambda x: abs(x["parse_tree_depth"] - dep_m) <= 3 * dep_std,
            num_proc=24,
        )
        logger.info(f"Filtered out outlier dependency parse tree depth values, {len(vg_ds)} entries remain.")

        ac_m = pd.Series(vg_ds["sg_filtered_depth"]).mean()
        ac_std = pd.Series(vg_ds["sg_filtered_depth"]).std()
        vg_ds = vg_ds.filter(
            lambda x: abs(x["sg_filtered_depth"] - ac_m) <= 3 * ac_std,
            num_proc=24,
        )
        logger.info(f"Filtered out outliers for the number of filtered verbs values, {len(vg_ds)} entries remain.")

    # 4. Select by dependency parse tree depth that match max and min quantile
    dep_min = int(pd.Series(vg_ds["parse_tree_depth"]).quantile(dep_quantile))
    dep_max = int(pd.Series(vg_ds["parse_tree_depth"]).quantile(1 - dep_quantile))
    vg_ds = vg_ds.filter(
        lambda x: x["parse_tree_depth"] <= dep_min or x["parse_tree_depth"] >= dep_max,
        num_proc=24,
    )
    logger.info(
        f"Filtered the dataset for dependency parse tree depths of either <= {dep_min} or "
        f">={dep_max}, {len(vg_ds)} entries remain."
    )

    # 5. Select by number scene graph depths that match max and min quantiles (min 1 filtered verb)
    ac_min = max(1, int(pd.Series(vg_ds["sg_filtered_depth"]).quantile(sg_filtered_depth_quantile)))
    ac_max = int(pd.Series(vg_ds["sg_filtered_depth"]).quantile(1 - sg_filtered_depth_quantile))
    vg_ds = vg_ds.filter(
        lambda x: x["sg_filtered_depth"] > 0 and (
            x["sg_filtered_depth"] <= ac_min or x["sg_filtered_depth"] >= ac_max
        ),
        num_proc=24,
    )
    logger.info(
        f"Filtered the dataset for a number of filtered verbs of either <= {ac_min} or "
        f">={ac_max}, {len(vg_ds)} entries remain."
    )

    # 6. Filter out black and white images and images with text on them and images with low quality
    # Add the images to the dataset, load based on the filenames
    # Avoid PIL-related bugs by copying the images
    vg_ds = vg_ds.map(
        lambda x: {**x, "img": copy.deepcopy(Image.open(os.path.join(VG_IMAGE_DIR, x["filename"])))},
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
        lambda x: np.array(x["img"]).ndim > 2 and not np.all(
            np.array(x["img"])[:20, :20, 0] == np.array(x["img"])[:20, :20, 1]
        ),
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
    if filter_by_animal_person:
        # Filter out images that do not contain any animals or people based on the coco_animal_person column
        vg_ds = vg_ds.filter(
            lambda x: x["coco_animal_person"],
            num_proc=24,
        )

    # 7. Map the conditions to the examples
    vg_ds = vg_ds.map(
        lambda x: map_conditions(
            x,
            min_dep_parse_tree_depth=dep_min,
            max_dep_parse_tree_depth=dep_max,
            min_sg_filtered_depth=ac_min,
            max_sg_filtered_depth=ac_max,
        ),
        num_proc=24,
    )

    # Select n_stimuli many stimuli, evenly distributed over the conditions
    vg_n_stim = []
    conditions = ["low_text_low_image", "high_text_high_image"] if binarized_conditions else [
        "low_text_low_image", "low_text_high_image", "high_text_low_image", "high_text_high_image"
    ]
    for comp in conditions:
        try:
            filtered_ds = vg_ds.filter(
                lambda x: x["complexity"] == comp,
                num_proc=24,
            )
            if len(filtered_ds) < n_stimuli // len(conditions):
                print(f"Not enough stimuli for the {comp} condition")
                return
            # Select the remaining stimuli randomly
            random_indices = np.random.choice(len(filtered_ds), size=n_stimuli // len(conditions), replace=False)
            filtered_ds = filtered_ds.select(random_indices)
            vg_n_stim.append(filtered_ds)
        except:  # noqa
            print(f"Not enough stimuli for the {comp} condition")
            return

    vg_ds_n_stimuli = concatenate_datasets(vg_n_stim)

    # 8. Save the dataset
    output_dir = os.path.split(vg_coco_preprocessed_dir)[0]
    vg_ds_n_stimuli.save_to_disk(
        os.path.join(
            output_dir,
            f"vg_coco_{n_stimuli}_stimuli",
        )
    )


@click.group()
def cli() -> None:
    """Select stimuli from the VG + COCO overlap dataset."""


if __name__ == "__main__":
    cli.add_command(select_stimuli)
    cli()
