"""Visualize the stimuli chosen with the data/select_stimuli.py script based on streamlit.

To see the stimuli, run the following command: streamlit run visualize_selected_stimuli.py
"""
import json
import os

import requests
import spacy
import streamlit as st
from datasets import load_dataset, load_from_disk
from PIL import Image
from spacy_streamlit import visualize_parser

from compositionality_study.constants import (
    VG_COCO_SELECTED_STIMULI_DIR,
    VG_OBJECTS_FILE,
    VG_OBJ_REL_IDX_FILE,
    VG_RELATIONSHIPS_FILE,
)
from compositionality_study.utils import draw_objs_and_rels

# Set up the streamlit page
st.set_page_config(layout="wide")

# Load a spacy model to display the dependency trees
nlp = spacy.load("en_core_web_lg")

# Load the dataset
dataset = load_from_disk(VG_COCO_SELECTED_STIMULI_DIR)
# Get the complexities
complexities = list(set([example["complexity"] for example in dataset]))

# Create table-like layout
col1, col2, col3, col4 = st.columns(4, gap="medium")

# Retrieve the id column
id_col = "cocoid" if "cocoid" in dataset.features else "id"

objs, rels = None, None
vg_objs_idx, vg_rels_idx, obj_rel_idx_file = None, None, None

if os.path.exists(VG_OBJECTS_FILE) and os.path.exists(VG_RELATIONSHIPS_FILE) and os.path.exists(
    VG_OBJ_REL_IDX_FILE
):
    objs = load_dataset("json", data_files=VG_OBJECTS_FILE, split="train")
    rels = load_dataset("json", data_files=VG_RELATIONSHIPS_FILE, split="train")
    with open(VG_OBJ_REL_IDX_FILE, "r") as f:
        obj_rel_idx_file = json.load(f)

for cell, comp in zip([col1, col2, col3, col4], complexities):
    with cell:
        header = comp.replace("_", " ")  # noqa
        st.header(f"{header}")
        examples = [example for example in dataset if example["complexity"] == comp]
        for example in examples:
            image = Image.open(requests.get(example["vg_url"], stream=True).raw)  # noqa

            # Also plot the objects and relationships if the objects and relationship files exist
            if os.path.exists(VG_OBJECTS_FILE) and os.path.exists(VG_RELATIONSHIPS_FILE) and os.path.exists(
                VG_OBJ_REL_IDX_FILE
            ):
                # Find the object annotations for the selected image
                image_objects = objs[obj_rel_idx_file[str(example["vg_image_id"])]["objs"]]["objects"][0]
                # Find the relationship annotations for the selected image
                image_relationships = rels[
                    obj_rel_idx_file[str(example["vg_image_id"])]["rels"]
                ]["relationships"][0]
                image = draw_objs_and_rels(image, image_objects, image_relationships)

                st.image(image, caption=f"{example['sentences_raw']}")
                doc = nlp(example["sentences_raw"])
                st.empty()
                # Generate the dependency parse tree and display it
                visualize_parser(
                    doc,
                    key=f"{str(example[id_col])}_parser_split_sents",
                    displacy_options={"compact": True, "distance": 65},
                    title=None,
                )
            st.divider()
