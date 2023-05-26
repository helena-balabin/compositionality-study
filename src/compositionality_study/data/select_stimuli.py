"""Select stimuli from the VG + COCO overlap dataset."""
# Imports
import os
from typing import Any, Dict

import click
import numpy as np
from datasets import concatenate_datasets, load_from_disk
from loguru import logger

from compositionality_study.constants import VG_COCO_PREP_TEXT_IMG_SEG_DIR


def map_conditions(
    example: Dict[str, Any],
    min_dep_parse_tree_depth: int = 5,
    max_dep_parse_tree_depth: int = 10,
    min_n_obj_img_seg: int = 3,
    max_n_obj_img_seg: int = 18,
) -> Dict:
    """Map the conditions (high/low textual complexity, high/low visual complexity) to the example.

    :param example: The hf dataset example to map the conditions to (high/low textual complexity,
        high/low visual complexity)
    :type example: Dict[str, Any]
    :param min_dep_parse_tree_depth: The minimum dependency parse tree depth
    :type min_dep_parse_tree_depth: int
    :param max_dep_parse_tree_depth: The maximum dependency parse tree depth
    :type max_dep_parse_tree_depth: int
    :param min_n_obj_img_seg: The minimum number of objects (image segmentation based)
    :type min_n_obj_img_seg: int
    :param max_n_obj_img_seg: The maximum number of objects (image segmentation based)
    :type max_n_obj_img_seg: int
    :return: The example with the mapped conditions, i.e., extra features for the conditions
    :rtype: Dict
    """
    # Map the textual complexity condition
    middle_dep_parse_tree_depth = (min_dep_parse_tree_depth + max_dep_parse_tree_depth) / 2
    textual_complexity = "high" if example["parse_tree_depth"] >= middle_dep_parse_tree_depth else "low"
    example["textual_complexity"] = textual_complexity

    # Map the visual complexity condition
    middle_n_obj_seg = (min_n_obj_img_seg + max_n_obj_img_seg) / 2
    img_seg_complexity = "high" if example["n_img_seg_obj"] >= middle_n_obj_seg else "low"
    example["img_seg_complexity"] = img_seg_complexity

    # Combine the two
    example["complexity"] = f"{textual_complexity}_text_{img_seg_complexity}_objseg"

    return example


@click.command()
@click.option("--vg_coco_text_img_seg_dir", type=str, default=VG_COCO_PREP_TEXT_IMG_SEG_DIR)
@click.option("--sent_len", type=int, default=15)
@click.option("--sent_len_tol", type=int, default=2)
@click.option("--min_dep_parse_tree_depth", type=int, default=5)
@click.option("--max_dep_parse_tree_depth", type=int, default=10)
@click.option("--dep_tol", type=int, default=1)
@click.option("--min_n_obj_img_seg", type=int, default=3)
@click.option("--max_n_obj_img_seg", type=int, default=18)
@click.option("--obj_img_seg_tol", type=int, default=2)
@click.option("--n_stimuli", type=int, default=80)
def select_stimuli(
    vg_coco_text_img_seg_dir: str = VG_COCO_PREP_TEXT_IMG_SEG_DIR,
    sent_len: int = 15,
    sent_len_tol: int = 2,
    min_dep_parse_tree_depth: int = 5,
    max_dep_parse_tree_depth: int = 10,
    dep_tol: int = 1,
    min_n_obj_img_seg: int = 3,
    max_n_obj_img_seg: int = 18,
    obj_img_seg_tol: int = 2,
    n_stimuli: int = 80,
):
    """Select stimuli from the VG + COCO overlap dataset.

    :param vg_coco_text_img_seg_dir: The preprocessed VG + COCO overlap dataset to select stimuli from
    :type vg_coco_text_img_seg_dir: str
    :param sent_len: The sentence length to control for (controlled variable
    :type sent_len: int
    :param sent_len_tol: The tolerance for the sentence length (+- sent_len_tol within sent_len)
    :type sent_len_tol: int
    :param min_dep_parse_tree_depth: The min dependency parse tree depth to select stimuli for
    :type min_dep_parse_tree_depth: int
    :param max_dep_parse_tree_depth: The max dependency parse tree depth to select stimuli for
    :type max_dep_parse_tree_depth: int
    :param dep_tol: The tolerance for the dependency parse tree depth
    :type dep_tol: int
    :param min_n_obj_img_seg: The min number of objects (image segmentation based) to select stimuli for
    :type min_n_obj_img_seg: int
    :param max_n_obj_img_seg: The max number of objects (image segmentation based) to select stimuli for
    :type max_n_obj_img_seg: int
    :param obj_img_seg_tol: The tolerance for the number of objects (image segmentation based)
    :type obj_img_seg_tol: int
    :param n_stimuli: The number of stimuli to select, must be divisible by 4
    :type n_stimuli: int
    """
    # Load the dataset
    vg_ds = load_from_disk(vg_coco_text_img_seg_dir)
    logger.info(f"Loaded the preprocessed VG + COCO overlap dataset with {len(vg_ds)} entries.")
    # Set a random seed for reproducibility
    np.random.seed(42)

    # 1. Filter by sentence length (within a tolerance)
    vg_ds_s_len = vg_ds.filter(
        lambda x: abs(x["sentence_length"] - sent_len) <= sent_len_tol,
        num_proc=4,
    )
    logger.info(f"Controlled the dataset for a sentence length of {sent_len} within a tolerance of {sent_len_tol}, "
                f"{len(vg_ds_s_len)} entries remain.")

    # 2. Select by dependency parse tree depth that match max and min
    vg_ds_dep_p = vg_ds_s_len.filter(
        lambda x: abs(
            x["parse_tree_depth"] - max_dep_parse_tree_depth
        ) <= dep_tol or abs(
            x["parse_tree_depth"] - min_dep_parse_tree_depth
        ) <= dep_tol,
        num_proc=4,
    )
    logger.info(f"Filtered the dataset for dependency parse tree depths of either {min_dep_parse_tree_depth} or "
                f"{max_dep_parse_tree_depth} within a tolerance of {dep_tol}, {len(vg_ds_dep_p)} entries remain.")

    # 3. TODO filter by visual properties

    # 4. Select by number of objects (image segmentation rather than scene graph) that match max and min
    vg_ds_n_obj_img_seg = vg_ds_dep_p.filter(
        lambda x: abs(x["n_img_seg_obj"] - max_n_obj_img_seg) <= obj_img_seg_tol or abs(
            x["n_img_seg_obj"] - min_n_obj_img_seg
        ) <= obj_img_seg_tol,
        num_proc=4,
    )
    logger.info(f"Filtered the dataset for a number of objects (img seg) of either {min_n_obj_img_seg} or "
                f"{max_n_obj_img_seg} within a tolerance of {obj_img_seg_tol}, {len(vg_ds_n_obj_img_seg)} "
                f"many entries remain.")

    # 5. Map the conditions to the examples
    vg_ds_n_obj_img_seg = vg_ds_n_obj_img_seg.map(
        lambda x: map_conditions(
            x,
            min_dep_parse_tree_depth=min_dep_parse_tree_depth,
            max_dep_parse_tree_depth=max_dep_parse_tree_depth,
            min_n_obj_img_seg=min_n_obj_img_seg,
            max_n_obj_img_seg=max_n_obj_img_seg,
        ),
        num_proc=4,
    )

    # 6. Select n_stimuli many stimuli, evenly distributed over the conditions
    vg_n_stim = []
    for comp in ["low_text_low_objseg", "low_text_high_objseg", "high_text_low_objseg", "high_text_high_objseg"]:
        try:
            filtered_ds = vg_ds_n_obj_img_seg.filter(
                lambda x: x["complexity"] == comp,
                num_proc=4,
            )
            if len(filtered_ds) < n_stimuli // 4:
                print(f"Not enough stimuli for the {comp} condition")
                return
            # Select the remaining stimuli randomly
            random_indices = np.random.choice(len(filtered_ds), size=n_stimuli//4, replace=False)
            filtered_ds = filtered_ds.select(random_indices)
            vg_n_stim.append(filtered_ds)
        except:  # noqa
            print(f"Not enough stimuli for the {comp} condition")
            return
    vg_ds_n_stimuli = concatenate_datasets(vg_n_stim)

    # Save the dataset
    output_dir = os.path.split(vg_coco_text_img_seg_dir)[0]
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
