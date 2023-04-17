"""Select stimuli from the VG + COCO overlap dataset."""
from typing import Any, Dict

# Imports
import click
# import pydevd_pycharm
from datasets import concatenate_datasets, load_from_disk

from compositionality_study.constants import VG_COCO_PREP_TEXT_GRAPH_DIR


def map_conditions(
    example: Dict[str, Any],
    min_dep_parse_tree_depth: int = 5,
    max_dep_parse_tree_depth: int = 10,
    min_n_rel: int = 10,
    max_n_rel: int = 50,
) -> Dict:
    """Map the conditions (high/low textual complexity, high/low visual complexity) to the example.

    :param example: The hf dataset example to map the conditions to (high/low textual complexity,
        high/low visual complexity)
    :type example: Dict[str, Any]
    :param min_dep_parse_tree_depth: The minimum dependency parse tree depth
    :type min_dep_parse_tree_depth: int
    :param max_dep_parse_tree_depth: The maximum dependency parse tree depth
    :type max_dep_parse_tree_depth: int
    :param min_n_rel: The minimum number of relations
    :type min_n_rel: int
    :param max_n_rel: The maximum number of relations
    :type max_n_rel: int
    :return: The example with the mapped conditions, i.e., extra features for the conditions
    :rtype: Dict
    """
    # Map the textual complexity condition
    middle_dep_parse_tree_depth = (min_dep_parse_tree_depth + max_dep_parse_tree_depth) / 2
    textual_complexity = "high" if example["parse_tree_depth"] >= middle_dep_parse_tree_depth else "low"
    example["textual_complexity"] = textual_complexity

    # Map the graph complexity condition
    middle_n_rel = (min_n_rel + max_n_rel) / 2
    graph_complexity = "high" if example["n_rel"] >= middle_n_rel else "low"
    example["graph_complexity"] = graph_complexity

    # Combine the two
    example["complexity"] = f"{textual_complexity}_text_{graph_complexity}_graph"

    return example


@click.command()
@click.option("--vg_coco_text_graph_dir", type=str, default=VG_COCO_PREP_TEXT_GRAPH_DIR)
@click.option("--sent_len", type=int, default=20)
@click.option("--sent_len_tol", type=int, default=2)
@click.option("--min_dep_parse_tree_depth", type=int, default=5)
@click.option("--max_dep_parse_tree_depth", type=int, default=10)
@click.option("--dep_tol", type=int, default=1)
@click.option("--min_n_rel", type=int, default=10)
@click.option("--max_n_rel", type=int, default=20)
@click.option("--rel_tol", type=int, default=5)
@click.option("--n_stimuli", type=int, default=80)
def select_stimuli(
    vg_coco_text_graph_dir: str = VG_COCO_PREP_TEXT_GRAPH_DIR,
    sent_len: int = 20,
    sent_len_tol: int = 2,
    min_dep_parse_tree_depth: int = 5,
    max_dep_parse_tree_depth: int = 10,
    dep_tol: int = 1,
    min_n_rel: int = 10,
    max_n_rel: int = 20,
    rel_tol: int = 5,
    n_stimuli: int = 80,
):
    """Select stimuli from the VG + COCO overlap dataset.

    :param vg_coco_text_graph_dir: The preprocessed VG + COCO overlap dataset to select stimuli from
    :type vg_coco_text_graph_dir: str
    :param sent_len: The sentence length to control for
    :type sent_len: int
    :param sent_len_tol: The tolerance for the sentence length (+- sent_len_tol within sent_len)
    :type sent_len_tol: int
    :param min_dep_parse_tree_depth: The min dependency parse tree depth to select stimuli for
    :type min_dep_parse_tree_depth: int
    :param max_dep_parse_tree_depth: The max dependency parse tree depth to select stimuli for
    :type max_dep_parse_tree_depth: int
    :param dep_tol: The tolerance for the dependency parse tree depth
    :type dep_tol: int
    :param min_n_rel: The min number of relationships to select stimuli for
    :type min_n_rel: int
    :param max_n_rel: The max number of relationships to select stimuli for
    :type max_n_rel: int
    :param rel_tol: The tolerance for the number of relationships
    :type rel_tol: int
    :param n_stimuli: The number of stimuli to select, must be divisible by 4
    :type n_stimuli: int
    """
    # Load the dataset
    vg_ds = load_from_disk(vg_coco_text_graph_dir)

    # 1. Filter by sentence length (within a tolerance)
    vg_ds_s_len = vg_ds.filter(
        lambda x: abs(x["sentence_length"] - sent_len) <= sent_len_tol,
        num_proc=4,
    )
    # 2. Select by dependency parse tree depth that match max and min
    vg_ds_dep_p = vg_ds_s_len.filter(
        lambda x: abs(
            x["parse_tree_depth"] - max_dep_parse_tree_depth
        ) <= dep_tol or abs(
            x["parse_tree_depth"] - min_dep_parse_tree_depth
        ) <= dep_tol,
        num_proc=4,
    )
    # 3. Select by number of relationships
    vg_ds_n_rel = vg_ds_dep_p.filter(
        lambda x: abs(x["n_rel"] - max_n_rel) <= rel_tol or abs(
            x["n_rel"] - min_n_rel
        ) <= rel_tol,
        num_proc=4,
    )
    # 4. Map the conditions to the examples
    vg_ds_n_rel = vg_ds_n_rel.map(
        lambda x: map_conditions(
            x,
            min_dep_parse_tree_depth=min_dep_parse_tree_depth,
            max_dep_parse_tree_depth=max_dep_parse_tree_depth,
            min_n_rel=min_n_rel,
            max_n_rel=max_n_rel,
        ),
        num_proc=4,
    )
    # 5. Select n_stimuli many stimuli, evenly distributed over the conditions
    vg_n_stim = []
    for comp in ["low_text_low_graph", "low_text_high_graph", "high_text_low_graph", "high_text_high_graph"]:
        try:
            vg_n_stim.append(
                vg_ds_n_rel.filter(
                    lambda x: x["complexity"] == comp,
                    num_proc=4,
                ).select(range(n_stimuli // 4))
            )
        except:  # noqa
            print(f"Not enough stimuli for the {comp} condition")
            return
    vg_ds_n_stimuli = concatenate_datasets(vg_n_stim)

    # pydevd_pycharm.settrace('localhost', port=8223, stdoutToServer=True, stderrToServer=True)
    # TODO save the dataset
    # TODO maybe plot some images + dep trees? -> Get some examples

    return NotImplementedError()


@click.group()
def cli() -> None:
    """Select stimuli from the VG + COCO overlap dataset."""


if __name__ == "__main__":
    cli.add_command(select_stimuli)
    cli()
