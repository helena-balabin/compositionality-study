"""Visualize the stimuli chosen with the data/select_stimuli.py script based on streamlit.

To see the stimuli, run the following command: streamlit run visualize_selected_stimuli.py
"""
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

if os.path.exists(VG_OBJECTS_FILE) and os.path.exists(VG_RELATIONSHIPS_FILE):
    objs = load_dataset("json", data_files=VG_OBJECTS_FILE, split="train")
    rels = load_dataset("json", data_files=VG_RELATIONSHIPS_FILE, split="train")

for cell, comp in zip([col1, col2, col3, col4], complexities):
    with cell:
        header = comp.replace("_", " ")  # noqa
        st.header(f"{header}")
        examples = [example for example in dataset if example["complexity"] == comp]
        for example in examples:
            try:
                image = Image.open(requests.get(example["vg_url"], stream=True).raw)  # noqa

                # Also plot the objects and relationships if the objects and relationship files exist
                if objs and rels:
                    # Find the object annotations for the selected image
                    image_objects = [
                        obj for obj in objs if obj["image_id"] == example["vg_image_id"]
                    ]
                    # Find the relationship annotations for the selected image
                    image_relationships = [
                        rel for rel in rels if rel["image_id"] == example["vg_image_id"]
                    ]
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
            except:  # noqa
                st.warning(
                    f"Failed to load image for example with caption: {example['sentences_raw']}"
                )
            st.divider()
