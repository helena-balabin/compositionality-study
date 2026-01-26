"""Create image files for the selected stimuli with their action annotations."""

import json
import os
from pathlib import Path
from typing import Dict, List

import amrlib
import click
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import spacy
from amrlib.graph_processing.amr_plot import AMRPlot
from datasets import load_from_disk
from loguru import logger
from PIL import Image  # type: ignore
from scipy.stats import chi2_contingency, ttest_ind
from spacy import Language  # type: ignore
from tqdm import tqdm

from compositionality_study.constants import (
    COCO_A_ANNOT_FILE,
    COCO_DIR,
    COCO_IMAGE_DIR,
    COCO_LOCAL_STIMULI_DIR,
    COCO_OBJ_SEG_DIR,
    COCO_SELECTED_STIMULI_DIR,
    IMAGES_COCO_SELECTED_STIMULI_DIR,
)
from compositionality_study.data.preprocess_coco import get_coco_a_graphs
from compositionality_study.utils import dependency_parse_to_nx, get_amr_graph_depth


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
    input_dir: str = COCO_OBJ_SEG_DIR,
) -> List[Dict]:
    """Load COCO image annotations.

    :param input_dir: Path to the COCO image annotations dir
    :type input_dir: str
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
    img: Image.Image,
    depth: int,
    actions: List[Dict],
    coco_annots: List[Dict],
    output_path: str,
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
                plt.Rectangle(  # type: ignore
                    (subject_bbox["bbox"][0], subject_bbox["bbox"][1]),
                    subject_bbox["bbox"][2],
                    subject_bbox["bbox"][3],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
            )
            plt.gca().add_patch(
                plt.Rectangle(  # type: ignore
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
    default=COCO_SELECTED_STIMULI_DIR,
    help="Path to the dataset containing selected stimuli",
)
@click.option(
    "--coco_annotations_dir",
    type=str,
    default=COCO_OBJ_SEG_DIR,
    help="Directory containing COCO image annotations",
)
@click.option(
    "--output_dir",
    type=str,
    default=IMAGES_COCO_SELECTED_STIMULI_DIR,
    help="Directory where visualizations will be saved",
)
def visualize_actions(
    dataset_path: str = COCO_SELECTED_STIMULI_DIR,
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
    coco_annotations_dir: str = COCO_OBJ_SEG_DIR,
    coco_image_dir: str = COCO_IMAGE_DIR,
    output_dir: str = IMAGES_COCO_SELECTED_STIMULI_DIR,
) -> None:
    """Visualize actions for selected stimuli from the dataset.

    :param dataset_path: Path to the dataset containing selected stimuli
    :type dataset_path: str
    :param coco_a_annot_file: Path to the COCO-A annotations file
    :type coco_a_annot_file: str
    :param coco_annotations_dir: Directory containing COCO image annotations (instances_train/val)
    :type coco_annotations_dir: str
    :param coco_image_dir: Directory containing COCO images
    :type coco_image_dir: str
    :param output_dir: Directory where visualizations will be saved
    :type output_dir: str
    """
    # Load COCO-A annotations
    coco_actions = load_coco_actions(coco_a_annot_file)

    # Load the instances_train/val2017 file
    coco_train_val = load_coco_annotations(coco_annotations_dir)

    # Load our selected stimuli dataset
    dataset = load_from_disk(dataset_path)
    dataset_ids = [item["cocoid"] for item in dataset]  # type: ignore

    # Filter in coco_actions only the actions that are in our dataset
    coco_actions_filtered = [i for i in coco_actions if i["image_id"] in dataset_ids]
    # Same for coco_train_val
    coco_train_val_filtered = [i for i in coco_train_val if i["image_id"] in dataset_ids]

    # Create output directory with if needed
    output_dir = Path(output_dir)  # type: ignore
    output_dir.mkdir(parents=True, exist_ok=True)  # type: ignore

    # Process each image in our dataset
    for item in tqdm(dataset, desc="Processing images"):
        image_id = item["cocoid"]  # type: ignore
        image_actions = [i for i in coco_actions_filtered if i["image_id"] == image_id]
        image_coco_annots = [i for i in coco_train_val_filtered if i["image_id"] == image_id]

        # Get the PIL image from the dataset
        img = Image.open(os.path.join(coco_image_dir, item["filepath"]))  # type: ignore
        image_output_id = item["sentids"]  # type: ignore
        output_path = os.path.join(output_dir, f"{image_output_id}_image.png")

        try:
            visualize_image_with_actions(
                img, item["coco_a_graph_depth"], image_actions, image_coco_annots, str(output_path)  # type: ignore
            )
        except Exception as e:
            logger.info(f"Error processing image {image_id}: {e}")


@click.command()
@click.option(
    "--dataset_path",
    type=str,
    default=COCO_SELECTED_STIMULI_DIR,
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
    default=IMAGES_COCO_SELECTED_STIMULI_DIR,
    help="Directory where visualizations will be saved",
)
def visualize_amr_text(
    dataset_path: str = COCO_SELECTED_STIMULI_DIR,  # noqa
    spacy_model: str = "en_core_web_trf",
    output_dir: str = IMAGES_COCO_SELECTED_STIMULI_DIR,
):
    """Visualize AMR graphs for the text stimuli.

    :param dataset_path: Path to the dataset containing selected stimuli
    :type dataset_path: str
    :param spacy_model: spaCy model to use for text processing
    :type spacy_model: str
    :param output_dir: Directory where visualizations will be saved
    :type output_dir: str
    """

    @Language.component("force_single_sentence")  # noqa
    def one_sentence_per_doc(
        doc: spacy.tokens.Doc,  # type: ignore
    ) -> spacy.tokens.Doc:  # type: ignore
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
    spacy.prefer_gpu()  # type: ignore
    amrlib.setup_spacy_extension()
    # Disable unnecessary components
    nlp = spacy.load(spacy_model, disable=["tok2vec", "attribute_ruler", "lemmatizer"])
    nlp.add_pipe("force_single_sentence", before="parser")

    # Load the dataset
    dataset = load_from_disk(dataset_path)
    doc_batched = nlp.pipe(dataset["sentences_raw"])  # type: ignore

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
        output_path = os.path.join(output_dir, f"{ex['sentids']}_text")  # type: ignore
        graph = plot.graph
        graph.graph_attr["label"] = f"Depth for '{ex['sentences_raw']}': {ex['amr_graph_depth']}"  # type: ignore
        graph.render(output_path, format="png", cleanup=True)

    # Remove all .pdf files in the output directory
    for file in os.listdir(output_dir):
        if file.endswith(".pdf"):
            os.remove(os.path.join(output_dir, file))


def plot_graph_statistics(
    graphs: Dict[str, nx.DiGraph],
    title: str,
    output_path: str,
):
    """Plot statistics for the given graphs.

    :param graphs: Dictionary of graphs where keys are identifiers and values are NetworkX directed graphs
    :type graphs: Dict[str, nx.DiGraph]
    :param title: Title of the plot
    :type title: str
    :param output_path: Path to save the plot
    :type output_path: str
    """
    num_nodes = [len(graph.nodes) for graph in graphs.values()]
    num_edges = [len(graph.edges) for graph in graphs.values()]
    longest_shortest_paths = [
        (
            max([max(nx.shortest_path_length(graph, source=n).values()) for n in graph.nodes()])
            if nx.number_of_nodes(graph) > 0
            else 0
        )
        for graph in graphs.values()
    ]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title)

    sns.histplot(num_nodes, kde=True, ax=axs[0], discrete=True)
    axs[0].set_title("# Nodes")

    sns.histplot(num_edges, kde=True, ax=axs[1], discrete=True)
    axs[1].set_title("# Edges")

    sns.histplot(
        longest_shortest_paths,
        kde=True,
        ax=axs[2],
        discrete=True,
    )
    axs[2].set_title("Depth")

    plt.savefig(output_path)
    plt.close()


@click.command()
@click.option(
    "--stimuli_dir",
    type=str,
    default=COCO_SELECTED_STIMULI_DIR,
    help="Path to the dataset containing selected stimuli",
)
@click.option(
    "--visualizations_dir",
    type=str,
    default=os.path.join(COCO_DIR, "visualizations"),
    help="Directory where visualizations will be saved",
)
@click.option(
    "--coco_a_annot_file",
    type=str,
    default=COCO_A_ANNOT_FILE,
    help="Path to the COCO-A annotations file",
)
def get_summary_statistics(
    stimuli_dir: str = COCO_SELECTED_STIMULI_DIR,
    visualizations_dir: str = os.path.join(COCO_DIR, "visualizations"),  # noqa
    coco_a_annot_file: str = COCO_A_ANNOT_FILE,
):
    """Generate summary statistics plots for the selected stimuli.

    :param stimuli_dir: Path to the dataset containing selected stimuli
    :type stimuli_dir: str
    :param visualizations_dir: Directory where visualizations will be saved
    :type visualizations_dir: str
    :param coco_a_annot_file: Path to the COCO-A annotations file
    :type coco_a_annot_file: str
    """
    # Load the stimuli and convert them to a pandas DataFrame
    stimuli = load_from_disk(stimuli_dir)
    n_stimuli = len(stimuli)
    stimuli_df = pd.DataFrame(stimuli)  # type: ignore
    # Initialize the visualization directory
    os.makedirs(visualizations_dir, exist_ok=True)

    # Perform a Chi-square test between coco_person and amr_graph_depth
    contingency_table_amr = pd.crosstab(stimuli_df["coco_person"], stimuli_df["amr_graph_depth"])
    chi2_amr, p_value_amr, _, _ = chi2_contingency(contingency_table_amr)

    # Perform a Chi-square test between coco_person and coco_a_graph_depth
    contingency_table_coco_a = pd.crosstab(
        stimuli_df["coco_person"], stimuli_df["coco_a_graph_depth"]
    )
    chi2_coco_a, p_value_coco_a, _, _ = chi2_contingency(contingency_table_coco_a)

    # Save the Chi-square test results to a file in the visualization directory
    with open(os.path.join(visualizations_dir, "chi_square_test_results.txt"), "w") as f:
        f.write(
            f"Chi-square test between COCO person and AMR depth: Chi2 = {chi2_amr}, p-value = {p_value_amr}\n"
        )
        f.write(
            f"Chi-square test between COCO person and COCO-A depth: Chi2 = {chi2_coco_a}, p-value = {p_value_coco_a}\n"
        )

    # Generate a seaborn KDE plot of the textual and image complexity values
    sns.kdeplot(stimuli_df, x="amr_graph_depth", y="coco_a_graph_depth", cmap="Blues", fill=True)
    # Draw a diagonal line from (0,0) to (max_depth, max_depth)
    plt.plot(
        [0, max(stimuli_df["amr_graph_depth"])],
        [0, max(stimuli_df["coco_a_graph_depth"])],
        color="blue",
    )
    # And save it to a visualization directory
    plt.savefig(os.path.join(visualizations_dir, f"textual_vs_image_complexity_{n_stimuli}.png"))

    # Re-generate the image-based graphs for the stimuli
    # Re-generate COCO-A sub-graph
    # Get all the COCO-A features
    with open(coco_a_annot_file, "r") as f:
        coco_a_data = json.load(f)["annotations"]["3"]
        coco_a_ids = [ex["image_id"] for ex in coco_a_data]

    coco_a_graphs = get_coco_a_graphs(
        coco_a_data=coco_a_data,
        coco_a_ids=coco_a_ids,
    )
    # Filter by those present in the stimuli
    coco_a_graphs = {k: v for k, v in coco_a_graphs.items() if k in stimuli_df["cocoid"].values}

    # Initialize spaCy model
    nlp = spacy.load("en_core_web_trf")
    # Prefer GPU if available
    spacy.prefer_gpu()  # type: ignore
    # Set up the spacy amrlib extension
    amrlib.setup_spacy_extension()

    amr_graphs = {}
    parse_trees = {}

    # Re-generate the following text-based graphs for the stimuli
    for _, example in tqdm(stimuli_df.iterrows(), total=n_stimuli, desc="Processing stimuli"):
        # Re-generate AMR graph
        doc = nlp(example["sentences_raw"])
        amr_graph = doc._.to_amr()[0]
        _, amr_graph = get_amr_graph_depth(amr_graph, return_graph=True)  # type: ignore
        amr_graphs[example["cocoid"]] = amr_graph

        # Re-generate dependency parse tree
        parse_tree = dependency_parse_to_nx(doc.sents)  # type: ignore
        parse_trees[example["cocoid"]] = parse_tree

    # Plot statistics for AMR graphs
    plot_graph_statistics(
        amr_graphs,
        "AMR Graph Statistics",
        os.path.join(visualizations_dir, "amr_graph_statistics.png"),
    )

    # Plot statistics for dependency parse trees
    plot_graph_statistics(
        parse_trees,
        "Dependency Parse Tree Statistics",
        os.path.join(visualizations_dir, "parse_tree_statistics.png"),
    )

    # Plot statistics for COCO-A graphs
    coco_a_graphs_flat = {k: v["coco_a_graph"] for k, v in coco_a_graphs.items()}
    plot_graph_statistics(
        coco_a_graphs_flat,  # type: ignore
        "COCO-A Graph Statistics",
        os.path.join(visualizations_dir, "coco_a_graph_statistics.png"),
    )

    # Plot histogram for number of COCO actions
    n_coco_actions = [v["n_coco_a_actions"] for v in coco_a_graphs.values()]
    plt.figure(figsize=(6, 6))
    sns.histplot(n_coco_actions, kde=True, discrete=True)
    plt.title("Number of COCO Actions")
    plt.savefig(os.path.join(visualizations_dir, "n_coco_actions_histogram.png"))
    plt.close()


@click.command()
@click.option(
    "--selected_stimuli_dir",
    type=str,
    default=COCO_SELECTED_STIMULI_DIR,
    help="Path to the dataset containing selected stimuli",
)
@click.option(
    "--local_stimuli_dir",
    type=str,
    default=COCO_LOCAL_STIMULI_DIR,
    help="Directory containing local stimuli",
)
@click.option(
    "--output_dir",
    type=str,
    default=os.path.join(COCO_DIR, "visualizations"),
    help="Directory where visualizations will be saved",
)
def check_control_variables_balance(
    selected_stimuli_dir: str = COCO_SELECTED_STIMULI_DIR,
    local_stimuli_dir: str = COCO_LOCAL_STIMULI_DIR,
    output_dir: str = os.path.join(COCO_DIR, "visualizations"),  # noqa
) -> pd.DataFrame:
    """Check if control variables are balanced between high and low complexity groups.

    :param selected_stimuli_dir: Path to the dataset containing selected stimuli
    :type selected_stimuli_dir: str
    :param local_stimuli_dir: Directory containing local stimuli
    :type local_stimuli_dir: str
    :param output_dir: Directory where visualizations will be saved
    :type output_dir: str
    :return: DataFrame with statistics
    :rtype: pd.DataFrame
    """
    # Control variables to check
    control_vars = ["sentence_length", "coco_person", "ic_score"]
    # Load selected stimuli using load_from_disk
    selected_df: pd.DataFrame = load_from_disk(selected_stimuli_dir).to_pandas()  # type: ignore
    # Filter by available local stimuli
    local_stimuli = pd.read_csv(os.path.join(local_stimuli_dir, "stimuli_text_and_im_paths.csv"))
    selected_df = selected_df[selected_df["cocoid"].isin(local_stimuli["cocoid"])]

    # Add image metrics
    def get_image_metrics(row):
        """Get image metrics for a given row.

        :param row: Row of the DataFrame
        :type row: pd.Series
        :return: Series with aspect ratio and image height
        :rtype: pd.Series
        """
        img = Image.open(os.path.join(local_stimuli_dir, row["img_path"]))
        width, height = img.size
        return pd.Series({"aspect_ratio": width / height, "image_height": height})

    image_metrics = local_stimuli.apply(get_image_metrics, axis=1)
    image_metrics = pd.concat([local_stimuli["cocoid"], image_metrics], axis=1)
    selected_df = selected_df.merge(image_metrics, on="cocoid")

    # Split into high/low groups
    high_group = selected_df[selected_df["amr_graph_depth"] == 2]
    low_group = selected_df[selected_df["amr_graph_depth"] == 1]

    # Check all variables
    all_vars = control_vars + ["aspect_ratio", "image_height"]
    results = {}

    for var in all_vars:
        stat, pval = ttest_ind(high_group[var], low_group[var], equal_var=False)
        results[var] = {
            "high_mean": high_group[var].mean(),
            "high_std": high_group[var].std(),
            "low_mean": low_group[var].mean(),
            "low_std": low_group[var].std(),
            "t_stat": stat,
            "p_value": pval,
        }

    # Create a visualization directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save results to a DataFrame and write to a CSV file
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.to_csv(os.path.join(output_dir, "control_variable_balance_check.csv"))

    # Save means and p-values to a DataFrame and write to a CSV file
    summary_df = pd.DataFrame(
        {
            "Variable": results.keys(),
            "High Mean": [stats["high_mean"] for stats in results.values()],
            "High Std": [stats["high_std"] for stats in results.values()],
            "Low Mean": [stats["low_mean"] for stats in results.values()],
            "Low Std": [stats["low_std"] for stats in results.values()],
            "p-value": [stats["p_value"] for stats in results.values()],
            "Significant Difference": [
                "Yes" if stats["p_value"] < 0.05 else "No" for stats in results.values()
            ],
        }
    )
    summary_df.to_csv(os.path.join(output_dir, "control_variable_summary.csv"), index=False)

    # Log the results
    for var, stats_dict in results.items():
        logger.info(f"\n{var}:")
        logger.info(f"High group mean: {stats_dict['high_mean']:.3f}")
        logger.info(f"High group std: {stats_dict['high_std']:.3f}")
        logger.info(f"Low group mean: {stats_dict['low_mean']:.3f}")
        logger.info(f"Low group std: {stats_dict['low_std']:.3f}")
        logger.info(f"p-value: {stats_dict['p_value']:.3f}")
        logger.info(f"Significant difference: {'Yes' if stats_dict['p_value'] < 0.05 else 'No'}")

    return results_df


@click.group()
def cli() -> None:
    """Visualize actions and text AMR graphs and summary statistics for selected stimuli."""


if __name__ == "__main__":
    cli.add_command(visualize_actions)
    cli.add_command(visualize_amr_text)
    cli.add_command(get_summary_statistics)
    cli.add_command(check_control_variables_balance)
    cli()
