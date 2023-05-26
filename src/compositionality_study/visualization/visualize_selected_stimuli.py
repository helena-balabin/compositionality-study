"""Visualize the stimuli chosen with the data/select_stimuli.py script based on streamlit.

To see the stimuli, run the following command: streamlit run visualize_selected_stimuli.py
"""
# Imports
import requests
import spacy
import streamlit as st
from datasets import load_from_disk
from PIL import Image
from spacy_streamlit import visualize_parser

from compositionality_study.constants import VG_COCO_SELECTED_STIMULI_DIR

# Set up the streamlit page
st.set_page_config(layout="wide")

# Load a spacy model to display the dependency trees
nlp = spacy.load("en_core_web_sm")

# Load the dataset
dataset = load_from_disk(VG_COCO_SELECTED_STIMULI_DIR)
# Get the complexities
complexities = list(set([example["complexity"] for example in dataset]))

# Create table-like layout
col1, col2, col3, col4 = st.columns(4, gap="medium")

# Retrieve the id column
id_col = "cocoid" if "cocoid" in dataset.features else "id"

for cell, comp in zip([col1, col2, col3, col4], complexities):
    with cell:
        header = comp.replace('_', ' ')  # noqa
        st.header(f"{header}")
        examples = [example for example in dataset if example["complexity"] == comp]
        for example in examples:
            try:
                image = Image.open(requests.get(example["vg_url"], stream=True).raw)  # noqa
                st.image(image, caption=f"{example['sentences_raw']}")
                doc = nlp(example['sentences_raw'])
                st.empty()
                # Generate the dependency parse tree and display it
                visualize_parser(
                    doc,
                    key=f"{str(example[id_col])}_parser_split_sents",
                    displacy_options={"compact": True, "distance": 65},
                    title=None,
                )
            except:  # noqa
                st.warning(f"Failed to load image for example with caption: {example['sentences_raw']}")
            st.divider()
