"""Select stimuli from the COCO overlap dataset."""

import copy
import io
import os

import click
import networkx as nx
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from loguru import logger
from PIL import Image
from psmpy import PsmPy
from scipy.stats import ttest_ind

from compositionality_study.constants import COCO_IMAGE_DIR, COCO_PREP_ALL


@click.command()
@click.option("--coco_preprocessed_dir", type=str, default=COCO_PREP_ALL)
@click.option("--sent_len", type=int, default=10)
@click.option("--sent_len_tol", type=int, default=2)
@click.option("--img_comp", type=float, default=0.5)
@click.option("--img_comp_tol", type=float, default=0.3)
@click.option("--asp_min", type=float, default=1.0)
@click.option("--asp_max", type=float, default=2.0)
@click.option("--text_feature", type=str, default="amr_n_edges")
@click.option("--graph_feature", type=str, default="coco_a_edges")
@click.option("--quantile", type=float, default=0.3)
@click.option("--image_quality_threshold", type=int, default=300)
@click.option("--filter_by_person", type=bool, default=True)
@click.option("--person_count", type=int, default=5)
@click.option("--person_tol", type=int, default=2)
@click.option("--n_stimuli", type=int, default=252)
@click.option("--buffer_stimuli_fraction", type=float, default=0.25)
def select_stimuli(
    coco_preprocessed_dir: str = COCO_PREP_ALL,
    sent_len: int = 10,
    sent_len_tol: int = 2,
    img_comp: float = 0.5,
    img_comp_tol: float = 0.3,
    asp_min: float = 1.0,
    asp_max: float = 2.0,
    text_feature: str = "amr_n_edges",
    graph_feature: str = "coco_a_edges",
    quantile: float = 0.3,
    image_quality_threshold: int = 300,
    filter_by_person: bool = True,
    person_count: int = 5,
    person_tol: int = 2,
    n_stimuli: int = 252,
    buffer_stimuli_fraction: float = 0.25,
):
    """Select stimuli from the COCO overlap dataset.

    :param coco_preprocessed_dir: The preprocessed COCO overlap dataset to select stimuli from, defaults to
        COCO_PREP_ALL
    :type coco_preprocessed_dir: str
    :param sent_len: The sentence length to control for (controlled variable), defaults to 10
    :type sent_len: int
    :param sent_len_tol: The tolerance for the sentence length (+- sent_len_tol within sent_len), defaults to 2
    :type sent_len_tol: int
    :param img_comp: The image complexity to control for (controlled variable), defaults to 0.5
    :type img_comp: float
    :param img_comp_tol: The tolerance for the image complexity (+- img_comp_tol within img_comp), defaults to 0.3
    :type img_comp_tol: float
    :param asp_min: The min aspect ratio of the images to select stimuli for, defaults to 1.0
    :type asp_min: float
    :param asp_max: The max aspect ratio of the images to select stimuli for, defaults to 2.0
    :type asp_max: float
    :param text_feature: The text feature to parameterize with, defaults to "amr_n_edges"
    :type text_feature: str
    :param graph_feature: The graph feature to parameterize with, defaults to "coco_a_edges"
    :type graph_feature: str
    :param quantile: The quantile to use for the selection of the stimuli, defaults to 0.3
    :type quantile: float
    :param image_quality_threshold: Minimum number of pixels (height) of the image, defaults to 300
    :type image_quality_threshold: int
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
        0.25, meaning that in total n_stimuli * (1 + buffer_stimuli_fraction) stimuli will be selected
    """
    # Apply a buffer to the number of stimuli to select
    n_stimuli = int(n_stimuli * (1 + buffer_stimuli_fraction))
    # Make sure it"s an even number
    n_stimuli = n_stimuli if n_stimuli % 2 == 0 else n_stimuli + 1
    logger.info(f"Selecting {n_stimuli} stimuli (including buffer).")

    # Load the dataset
    ds = load_from_disk(coco_preprocessed_dir)
    logger.info(f"Loaded the preprocessed COCO overlap dataset with {len(ds)} entries.")
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Filter by sentence length (within a tolerance)
    ds = ds.filter(
        lambda x: abs(x["sentence_length"] - sent_len) <= sent_len_tol,
        num_proc=24,
    )
    logger.info(
        f"Controlled the dataset for a sentence length of {sent_len} within a tolerance of {sent_len_tol}, "
        f"{len(ds)} entries remain."
    )

    # Filter by person based on the desired number of people in the dataset
    if filter_by_person:
        ds = ds.filter(
            lambda x: np.abs(person_count - x["coco_person"]) <= person_tol,
            num_proc=24,
        )
        logger.info(
            f"Controlled the dataset for the number of people {person_count} +- {person_tol},"
            f"{len(ds)} entries remain."
        )

    # Filter by image complexity (within a tolerance)
    ds = ds.filter(
        lambda x: abs(x["ic_score"] - img_comp) <= img_comp_tol,
        num_proc=24,
    )
    logger.info(
        f"Controlled the dataset for an image complexity of {img_comp} within a tolerance of {img_comp_tol}, "
        f"{len(ds)} entries remain."
    )

    # Also filter by aspect ratio of the images (e.g., only horizontal images)
    ds = ds.filter(
        lambda x: asp_min <= x["aspect_ratio"] <= asp_max,
        num_proc=24,
    )
    logger.info(
        f"Controlled the dataset for the aspect ratio of the images (between {asp_min} and {asp_max}), "
        f"{len(ds)} entries remain."
    )

    # Filter out black and white images and images with text on them and images with low quality
    # Add the images to the dataset, load based on the filenames
    # Avoid PIL-related bugs by copying the images
    ds = ds.map(
        lambda x: {
            **x,
            "img": copy.deepcopy(Image.open(os.path.join(COCO_IMAGE_DIR, x["filepath"]))),
        },
        num_proc=24,
    )
    # Filter out images with low quality
    ds = ds.filter(
        lambda x: x["img"].size[0] > 1.5 * image_quality_threshold and x["img"].size[1] > image_quality_threshold,
        num_proc=24,
    )
    logger.info(f"Filtered out low quality images, {len(ds)} " f"entries remain.")

    # Pre-compute arrays for faster access
    text_complexities = np.array(ds[text_feature])
    img_complexities = np.array(ds[graph_feature])

    # Compute 25th & 75th percentiles
    q_text_low, q_text_high = np.quantile(text_complexities, [quantile, 1 - quantile])
    q_img_low, q_img_high = np.quantile(img_complexities, [quantile, 1 - quantile])

    # Filter the dataset based on the quantiles, only keep entries that are outside the quantiles
    ds_n_stimuli = ds.filter(
        lambda x: (x[text_feature] <= q_text_low and x[graph_feature] <= q_img_low)
        or (x[text_feature] >= q_text_high and x[graph_feature] >= q_img_high),
        num_proc=24,
    )
    df_n_stimuli = ds_n_stimuli.to_pandas()
    # Remove duplicates based on the sentids
    df_n_stimuli = df_n_stimuli.drop_duplicates(subset=["sentids"])
    # Use propensity score matching to select maxmimally close pairs of high/low, unique by sentids
    # TODO parameterize
    control_vars = [
        "sentence_length",
        "coco_person",
    ]
    logger.info(f"Text complexity quantiles ({quantile*100}%, {(1-quantile)*100}%): {q_text_low}, {q_text_high}")
    logger.info(f"Img complexity quantiles ({quantile*100}%, {(1-quantile)*100}%): {q_img_low}, {q_img_high}")

    # Build a DataFrame
    df = pd.DataFrame(
        {
            "sentids": df_n_stimuli["sentids"],
            "group": (
                (np.array(df_n_stimuli[text_feature]) >= q_text_high)
                & (np.array(df_n_stimuli[graph_feature]) >= q_img_high)
            ).astype(int),
            **{v: df_n_stimuli[v] for v in control_vars},
        }
    )
    # Init and fit propensity model
    psm = PsmPy(
        data=df,
        treatment="group",
        indx="sentids",
    )
    psm.logistic_ps(balance=False)
    psm.knn_matched(caliper=0.2, replacement=False)
    # TODO how to return all/more maybe
    matched = psm.df_matched
    # Take the group of the first element
    gr = matched["group"].iloc[0]
    matched = matched[matched["group"] == gr]
    # Append the imgids for each matched_ID in a column called "matched_imgid"
    matched["matched_imgid"] = matched["matched_ID"].apply(
        lambda x: df_n_stimuli[df_n_stimuli["sentids"] == x]["imgid"].values[0]
    )
    matched["imgid"] = matched["sentids"].apply(lambda x: df_n_stimuli[df_n_stimuli["sentids"] == x]["imgid"].values[0])
    # Remove any duplicates based on "imgid"
    matched = matched.drop_duplicates(subset=["imgid"])
    # And also based on "matched_imgid"
    matched = matched.drop_duplicates(subset=["matched_imgid"])

    # Getting the best unique pairs of sentidss is a maximum weight matching problem in a bipartite graph
    G = nx.Graph()
    # Add edges with weights as propensity scores
    for _, row in matched.iterrows():
        a = row["sentids"]
        b = row["matched_ID"]
        score = row["propensity_score"]
        G.add_edge(a, b, weight=score)

    # Solve the maximum weight matching
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    # Take n_stimuli // 2 pairs (since we want pairs of high/low) with highest propensity scores
    matching = sorted(
        matching,
        key=lambda x: G[x[0]][x[1]]["weight"],
        reverse=True,
    )[: n_stimuli // 2]
    # Filter ds_n_stimuli down to only the matched sentidss
    selected_ids = set()
    for a, b in matching:
        selected_ids.add(a)
        selected_ids.add(b)
    df_n_stimuli = df_n_stimuli[df_n_stimuli["sentids"].isin(selected_ids)]
    # Convert back to a Dataset
    ds_n_stimuli = Dataset.from_pandas(df_n_stimuli.reset_index(drop=True))
    # Map the condition "group" to 0/1 based on the text and image complexities
    ds_n_stimuli = ds_n_stimuli.map(
        lambda x: {"group": ((x[text_feature] >= q_text_high) and (x[graph_feature] >= q_img_high)).astype(int)},
        num_proc=24,
    )
    logger.info(f"After PSM, selected {len(ds_n_stimuli)} stimuli.")

    # For the control variables, do ttest_ind to check if the means are significantly different
    # TODO parameterize
    # Map image quality
    ds_n_stimuli = ds_n_stimuli.map(
        lambda x: {
            "image_quality": Image.open(io.BytesIO(x["img"]["bytes"])).size[
                1
            ],  # Use height as a proxy for image quality
        },
        num_proc=24,
    )
    check_vars = control_vars + ["ic_score", "aspect_ratio", "image_quality"]
    for var in check_vars:
        high = ds_n_stimuli.filter(lambda x: x["group"] == 1)[var]
        low = ds_n_stimuli.filter(lambda x: x["group"] == 0)[var]
        t_stat, p_val = ttest_ind(high, low)
        logger.info(
            f"Variable {var}: t-statistic={t_stat:.3f}, p-value={p_val:.3e}"
            f"mean_high={np.mean(high):.3f} +- ({np.std(high):.3f}),"
            f"mean_low={np.mean(low):.3f} +- ({np.std(low):.3f})",
        )

    # Save the dataset
    output_dir = os.path.split(coco_preprocessed_dir)[0]
    ds_n_stimuli.save_to_disk(
        os.path.join(
            output_dir,
            f"coco_{n_stimuli}_stimuli",
        )
    )


@click.group()
def cli() -> None:
    """Select stimuli from a COCO-based dataset."""


if __name__ == "__main__":
    cli.add_command(select_stimuli)
    cli()
