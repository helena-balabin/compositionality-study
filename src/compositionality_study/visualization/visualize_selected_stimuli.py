"""Create image files for the selected stimuli with their action annotations."""

import json
import os
from pathlib import Path
from typing import Dict, List

import amrlib
import click
import matplotlib.pyplot as plt
import spacy
from amrlib.graph_processing.amr_plot import AMRPlot
from datasets import load_from_disk
from PIL import Image
from spacy import Language
from tqdm import tqdm

from compositionality_study.constants import (
    COCO_A_ANNOT_FILE,
    IMAGES_VG_COCO_SELECTED_STIMULI_DIR,
    VG_COCO_OBJ_SEG_DIR,
    VG_COCO_SELECTED_STIMULI_DIR,
)


def load_coco_actions(
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
) -> Dict:
    """Load COCO-A annotations.

    :param coco_a_annot_file: Path to the COCO-A annotations file
    :type coco_a_annot_file: str
    :return: COCO-A annotations
    :rtype: Dict
    """
    with open(coco_a_annot_file, "r") as f:
        return json.load(f)["annotations"]["3"]


def load_coco_annotations(
    input_dir: str = VG_COCO_OBJ_SEG_DIR,
) -> List[Dict]:
    """Load COCO image annotations.

    :param input_path: Path to the COCO image annotations dir
    :type input_path: str
    :return: COCO image annotations
    :rtype: List[Dict]
    """
    # Load and combine the jsons for instances_train2017 and instances_val2017
    with open(Path(input_dir) / "instances_train2017.json", "r") as f:
        instances_train = json.load(f)["annotations"]
    with open(Path(input_dir) / "instances_val2017.json", "r") as f:
        instances_val = json.load(f)["annotations"]

    # Combine the two lists
    return instances_train + instances_val


def visualize_image_with_actions(
    img: Image, depth: int, actions: List[Dict], coco_annots: List[Dict], output_path: str
) -> None:
    """
    Visualize an image with its action annotations.

    :param img: PIL image
    :type img: Image
    :param depth: Depth of the image
    :type depth: int
    :param actions: List of COCO-A action annotations
    :type actions: List[Dict]
    :param coco_annots: List of COCO image annotations (bounding boxes)
    :type coco_annots: List[Dict]
    :param output_path: Path to save the visualization
    :type output_path: str
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Filter coco_annots: Only include entries where i["id"] is either in
    # "subject_id" or "object_id" of the actions
    all_subjects_objects = [i["subject_id"] for i in actions] + [i["object_id"] for i in actions]
    coco_annots_filtered = [i for i in coco_annots if i["id"] in all_subjects_objects]

    # Draw bounding boxes and actions
    for action in actions:
        # Make sure that neither subject nor object id are -1
        if action["subject_id"] != -1 and action["object_id"] != -1:
            # Get the bounding boxes for the subject and object
            subject_bbox = [i for i in coco_annots_filtered if i["id"] == action["subject_id"]][0]
            object_bbox = [i for i in coco_annots_filtered if i["id"] == action["object_id"]][0]

            # Draw the bounding boxes
            plt.gca().add_patch(
                plt.Rectangle(
                    (subject_bbox["bbox"][0], subject_bbox["bbox"][1]),
                    subject_bbox["bbox"][2],
                    subject_bbox["bbox"][3],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            plt.gca().add_patch(
                plt.Rectangle(
                    (object_bbox["bbox"][0], object_bbox["bbox"][1]),
                    object_bbox["bbox"][2],
                    object_bbox["bbox"][3],
                    linewidth=2,
                    edgecolor="b",
                    facecolor="none",
                )
            )

            # Draw the action, as a line between the centers of the bounding boxes
            plt.plot(
                [
                    subject_bbox["bbox"][0] + subject_bbox["bbox"][2] / 2,
                    object_bbox["bbox"][0] + object_bbox["bbox"][2] / 2,
                ],
                [
                    subject_bbox["bbox"][1] + subject_bbox["bbox"][3] / 2,
                    object_bbox["bbox"][1] + object_bbox["bbox"][3] / 2,
                ],
                color="g",
                linewidth=2,
            )
            # Use the depth as a title
            plt.title(f"Depth: {depth}")

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


@click.command()
@click.option(
    "--dataset_path",
    type=str,
    default=VG_COCO_SELECTED_STIMULI_DIR,
    help="Path to the dataset containing selected stimuli",
)
@click.option(
    "--coco_annotations_dir", type=str, default=VG_COCO_OBJ_SEG_DIR, help="Directory containing COCO image annotations"
)
@click.option(
    "--output_dir",
    type=str,
    default=IMAGES_VG_COCO_SELECTED_STIMULI_DIR,
    help="Directory where visualizations will be saved",
)
def visualize_actions(
    dataset_path: str = VG_COCO_SELECTED_STIMULI_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    coco_annotations_dir: str = VG_COCO_OBJ_SEG_DIR,
    output_dir: str = IMAGES_VG_COCO_SELECTED_STIMULI_DIR,
) -> None:
    """Visualize actions for selected stimuli from the dataset.

    :param dataset_path: Path to the dataset containing selected stimuli
    :type dataset_path: str
    :param coco_a_annot_file: Path to the COCO-A annotations file
    :type coco_a_annot_file: str
    :param coco_annotations_dir: Directory containing COCO image annotations (instances_train/val)
    :type coco_annotations_dir: str
    :param output_dir: Directory where visualizations will be saved
    :type output_dir: str
    """
    # Load COCO-A annotations
    coco_actions = load_coco_actions(coco_a_annot_file)

    # Load the instances_train/val2017 file
    coco_train_val = load_coco_annotations(coco_annotations_dir)

    # Load our selected stimuli dataset
    dataset = load_from_disk(dataset_path)
    dataset_ids = [item["cocoid"] for item in dataset]

    # Filter in coco_actions only the actions that are in our dataset
    coco_actions_filtered = [i for i in coco_actions if i["image_id"] in dataset_ids]
    # Same for coco_train_val
    coco_train_val_filtered = [i for i in coco_train_val if i["image_id"] in dataset_ids]

    # Create output directory with if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in our dataset
    for item in tqdm(dataset, desc="Processing images"):
        image_id = item["cocoid"]
        image_actions = [i for i in coco_actions_filtered if i["image_id"] == image_id]
        image_coco_annots = [i for i in coco_train_val_filtered if i["image_id"] == image_id]

        # Get the PIL image from the dataset
        img = item["img"]
        output_path = os.path.join(output_dir, f"{image_id}_image.png")

        try:
            visualize_image_with_actions(
                img, item["coco_a_graph_depth"], image_actions, image_coco_annots, str(output_path)
            )
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")


@click.command()
@click.option(
    "--dataset_path",
    type=str,
    default=VG_COCO_SELECTED_STIMULI_DIR,
    help="Path to the dataset containing selected stimuli",
)
@click.option(
    "--spacy_model",
    type=str,
    default="en_core_web_trf",
    help="spaCy model to use for text processing",
)
@click.option(
    "--output_dir",
    type=str,
    default=IMAGES_VG_COCO_SELECTED_STIMULI_DIR,
    help="Directory where visualizations will be saved",
)
def visualize_amr_text(
    dataset_path: str = VG_COCO_SELECTED_STIMULI_DIR,
    spacy_model: str = "en_core_web_trf",
    output_dir: str = IMAGES_VG_COCO_SELECTED_STIMULI_DIR,
):
    """Visualize AMR graphs for the text stimuli.

    :param dataset_path: Path to the dataset containing selected stimuli
    :type dataset_path: str
    :spacy_model: spaCy model to use for text processing
    :type spacy_model: str
    :param output_dir: Directory where visualizations will be saved
    :type output_dir: str
    """

    # Initialize the spaCy pipeline
    @Language.component("force_single_sentence")
    def one_sentence_per_doc(
        doc: spacy.tokens.Doc,  # noqa
    ) -> spacy.tokens.Doc:  # noqa
        """Force the document to be one sentence.

        :param doc: The document to force to be one sentence
        :type doc: spacy.tokens.Doc
        :return: The document with one sentence
        :rtype: spacy.tokens.Doc
        """
        doc[0].sent_start = True
        for i in range(1, len(doc)):
            doc[i].sent_start = False
        return doc

    # Add dependency parse tree depth and AMR depth
    # Prefer GPU if available
    spacy.prefer_gpu()
    amrlib.setup_spacy_extension()
    # Disable unnecessary components
    nlp = spacy.load(spacy_model, disable=["tok2vec", "attribute_ruler", "lemmatizer"])
    nlp.add_pipe("force_single_sentence", before="parser")

    # Load the dataset
    dataset = load_from_disk(dataset_path)
    doc_batched = nlp.pipe(dataset["sentences_raw"])

    for doc, ex in tqdm(
        zip(doc_batched, dataset),
        desc="Processing text stimuli",
        total=len(dataset),
    ):
        amr_graph = doc._.to_amr()[0]  # noqa
        # Make a figure and save it
        plot = AMRPlot()
        plot.build_from_graph(amr_graph, allow_deinvert=True)
        # Save the plot
        output_path = os.path.join(output_dir, f"{ex['cocoid']}_text")
        graph = plot.graph
        graph.graph_attr["label"] = f"Depth for '{ex['sentences_raw']}': {ex['amr_graph_depth']}"
        graph.render(output_path, format="png", cleanup=True)

    # Remove all .pdf files in the output directory
    for file in os.listdir(output_dir):
        if file.endswith(".pdf"):
            os.remove(os.path.join(output_dir, file))


@click.group()
def cli() -> None:
    """Visualize actions and text AMR graphs for selected stimuli."""


if __name__ == "__main__":
    cli.add_command(visualize_actions)
    cli.add_command(visualize_amr_text)
    cli()
