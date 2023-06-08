"""Utils for the compositionality project."""
# Imports
from typing import Any, Dict, List

import numpy as np
import spacy


def walk_tree(
    node: spacy.tokens.token.Token,
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
    nlp: spacy.lang,
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
