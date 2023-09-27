"""Utils for the compositionality project."""
import os
from io import BytesIO
from typing import Any, Dict, List

import PIL
import cv2
import numpy as np
import requests
import spacy
from PIL import Image, ImageOps

from compositionality_study.constants import VG_IMAGE_DIR


def walk_tree(
    node: spacy.tokens.token.Token,  # noqa
    depth: int,
) -> int:
    """Walk the dependency parse tree and return the maximum depth.

    :param node: The current node in the tree
    :type node: spacy.tokens.token.Token
    :param depth: The current depth in the tree
    :type depth: int
    :return: The maximum depth in the tree
    :rtype: int
    """
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


def walk_tree_hf_ds(
    example: Dict[str, Any],
    nlp: spacy.lang,  # noqa
) -> Dict[str, Any]:
    """Walk the dependency parse tree and return the maximum depth as well as the number of verbs.

    :param example: A hf dataset example
    :type example: Dict[str, Any]
    :param nlp: Spacy pipeline to use, initialized using nlp = spacy.load("en_core_web_trf")
    :type nlp: spacy.lang
    :return: The maximum depth in the tree and the number of verbs
    :rtype: Dict
    """
    doc = nlp(example["sentences_raw"])
    new_features = {
        "parse_tree_depth": walk_tree(next(doc.sents).root, 0),
        "n_verbs": len([token for token in doc if token.pos_ == "VERB"]),
    }
    return example | new_features


def flatten_examples(
    examples: Dict[str, List],
    flatten_col_names: List[str] = ["sentences_raw", "sentids"],  # noqa
) -> Dict[str, List]:
    """Flattens the examples in the dataset.

    :param examples: The examples to flatten
    :type examples: Dict[str, List]
    :param flatten_col_names: The column names to flatten, defaults to ["sentences_raw", "sentids"]
    :type flatten_col_names: List[str]
    :return: The flattened examples
    :rtype: Dict[str, List]
    """
    flattened_data = {}
    number_of_sentences = [len(sents) for sents in examples[flatten_col_names[0]]]
    for key, value in examples.items():
        if key in flatten_col_names:
            flattened_data[key] = [sent for sent_list in value for sent in sent_list]
        else:
            flattened_data[key] = np.repeat(value, number_of_sentences).tolist()
    return flattened_data


def get_image_aspect_ratio_from_url(
    url: str,
) -> float:
    """Get the aspect ratio of an image from a URL.

    :param url: The URL of the image
    :type url: str
    :return: The aspect ratio of the image, greater than 1 means the image is horizontal (i.e., wider than it is tall)
    :rtype: float
    """
    response = requests.get(url, timeout=10)
    image_bytes = BytesIO(response.content)

    with Image.open(image_bytes) as img:
        width, height = img.size
        aspect_ratio = width / height
        return aspect_ratio


def get_image_aspect_ratio_from_local_path(
    file_path: str,
    base_path: str = VG_IMAGE_DIR,
) -> float:
    """Get the aspect ratio of an image from a local path.

    :param file_path: The path to the image
    :type file_path: str
    :param base_path: The base path to the directory containing all the images
    :type base_path: str
    :return: The aspect ratio of the image, greater than 1 means the image is horizontal (i.e., wider than it is tall)
    :rtype: float
    """
    with Image.open(os.path.join(base_path, file_path)) as img:
        width, height = img.size
        aspect_ratio = width / height
        return aspect_ratio


def apply_gamma_correction(
    image: PIL.Image,
    target_mean=128.0,
) -> PIL.Image:
    """Apply gamma correction to an image.

    :param image: The image to apply gamma correction to
    :type image: PIL.Image
    :param target_mean: The target mean brightness of the image, defaults to 128.0
    :type target_mean: float, optional
    :return: The image with gamma correction applied
    :rtype: PIL.Image
    """
    # Convert the PIL image to a numpy array
    img_array = np.array(image)

    # Calculate the current mean brightness of the image
    current_mean = np.mean(img_array)

    # Calculate the gamma value to adjust the mean to the target mean
    gamma = np.log(target_mean) / np.log(current_mean)

    # Apply gamma correction to the image
    corrected_image = ImageOps.autocontrast(image, cutoff=gamma)

    return corrected_image
