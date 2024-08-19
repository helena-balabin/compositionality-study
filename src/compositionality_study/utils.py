"""Utils for the compositionality project."""

import os
from io import BytesIO
from typing import Dict, List, Tuple

import amrlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import penman
import PIL
import requests
import spacy
from labellines import labelLines
from nltk.corpus import wordnet as wn
from PIL import Image, ImageOps

from compositionality_study.constants import VG_IMAGE_DIR

# Prefer GPU if available
spacy.prefer_gpu()
# Set up the spacy amrlib extension
amrlib.setup_spacy_extension()


def get_amr_graph_depth(
    amr_graph: str,
) -> int:
    """Get the depth of the AMR graph for a given example.

    :param amr_graph: The AMR graph to get the depth for (output of a spacy doc._.to_amr()[0] call)
    :type amr_graph: str
    :return: The maximum "depth" of the AMR graph (longest shortest path)
    :rtype: int
    """
    # Convert to a Penman graph (with de-inverted edges)
    penman_graph = penman.decode(amr_graph)
    # Convert to a nx graph, first initialize the nx graph
    nx_graph = nx.DiGraph()
    # Add edges
    for e in penman_graph.edges():
        nx_graph.add_edge(e.source, e.target)
    # Get the characteristic path length of the graph
    amr_graph_depth = (
        max([max(nx.shortest_path_length(nx_graph, source=n).values()) for n in nx_graph.nodes()])
        if nx.number_of_nodes(nx_graph) > 0
        else 0
    )

    return amr_graph_depth


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


def derive_text_depth_features(
    examples: Dict[str, List],
    nlp: spacy.lang,  # noqa
) -> Dict[str, List]:
    """Get the depth of the dep parse tree, number of verbs and "depth" of the AMR graph of an example caption.

    The AMR model needs to be downloaded separately, see https://github.com/bjascob/amrlib-models.

    :param examples: A batch of hf dataset examples
    :type examples: Dict[str, List]
    :param nlp: Spacy pipeline to use, initialized using nlp = spacy.load("en_core_web_trf")
    :type nlp: spacy.lang
    :return: The batch with the added features
    :rtype: Dict[str, List]
    """
    result: Dict = {"parse_tree_depth": [], "n_verbs": [], "amr_graph_depth": []}
    doc_batched = nlp.pipe(examples["sentences_raw"])

    for doc in doc_batched:
        # Also derive the AMR graph for the caption and derive its depth
        amr_graph = doc._.to_amr()[0]  # noqa
        try:
            amr_depth = get_amr_graph_depth(amr_graph)
        except penman.exceptions.DecodeError:
            amr_depth = 0
        # Determine the depth of the dependency parse tree
        result["parse_tree_depth"].append(walk_tree(next(doc.sents).root, 0))
        result["n_verbs"].append(len([token for token in doc if token.pos_ == "VERB"]))
        result["amr_graph_depth"].append(amr_depth)

    return examples | result


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


def draw_objs_and_rels(
    input_img: PIL.Image,
    image_objects: List,
    image_relationships: List,
) -> PIL.Image:
    """Draw Visual Genome objects and relationships onto the respective image.

    :param input_img: Input image from Visual Genome
    :type input_img: PIL.Image
    :param image_objects: Objects for that image
    :type image_objects: List
    :param image_relationships: Relationships for that image
    :type image_relationships: List
    :return: The input image with boxes and relationships, in which the action-based relationships are marked in a
        different color
    :rtype: PIL.Image
    """
    # Create a Matplotlib figure
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(input_img)

    # Draw bounding boxes for objects
    for obj in image_objects:
        rect = patches.Rectangle(
            (obj["x"], obj["y"]),
            obj["w"],
            obj["h"],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    # Draw lines for relationships
    for rel in image_relationships:
        action_verb = check_if_filtered_rel(rel)
        rel_color = "g-" if action_verb[0] else "r-"
        subject = rel["subject"]
        obj = rel["object"]
        if action_verb[0]:
            # Plot the boxes in green again
            for box in [subject, obj]:
                rect = patches.Rectangle(
                    (box["x"], box["y"]),
                    box["w"],
                    box["h"],
                    linewidth=2,
                    edgecolor="g",
                    facecolor="none",
                )
                ax.add_patch(rect)
        # Plot the relationships
        x1 = subject["x"] + subject["w"] / 2
        y1 = subject["y"] + subject["h"] / 2
        x2 = obj["x"] + obj["w"] / 2
        y2 = obj["y"] + obj["h"] / 2
        plt.plot(
            [x1, x2],
            [y1, y2],
            rel_color,
            linewidth=2 if action_verb[0] else 1,
            label=action_verb[1],
        )

    labelLines(ax.get_lines(), zorder=2.5)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    return PIL.Image.open(buffer)


def check_if_filtered_rel(
    rel: Dict,
) -> Tuple[bool, str]:
    """Check if the name of a given relationship is a filtered relationship.

    :param rel: Relationship dictionary
    :type rel: Dict
    :return: True and name if it is a filtered relationship, False and empty string otherwise
    :rtype: Tuple(bool, str)
    """
    name = ""
    filtered_rel = (
        len(rel["object"]["synsets"]) > 0
        and len(rel["subject"]["synsets"]) > 0
        and (
            check_if_living_being(rel["object"]["synsets"][0])
            or check_if_living_being(rel["subject"]["synsets"][0])
        )
    )
    if filtered_rel:
        name = rel["synsets"][0].split(".")[0] if "." in rel else rel["predicate"]
    return filtered_rel, name


def check_if_living_being(
    synset: str,
) -> bool:
    """Check if a given synset is a living being by recursively checking its hypernyms.

    :param synset: The synset to check, e.g., "dog.n.01"
    :type synset: str
    :return: True if the synset describes a living being
    :rtype: bool
    """
    if len(synset) == 0:
        return False
    synset = wn.synset(synset)
    hypernyms = set()

    def recursive_hypernyms(
        syn: nltk.corpus.reader.wordnet.Synset,
    ):
        """Recursively check the hypernyms of a given synset.

        :param syn: The synset to check
        :type syn: wn.Synset
        """
        for hypernym in syn.hypernyms():
            hypernyms.add(hypernym)
            recursive_hypernyms(hypernym)

    recursive_hypernyms(synset)
    return wn.synset("animal.n.01") in hypernyms or wn.synset("person.n.01") in hypernyms
