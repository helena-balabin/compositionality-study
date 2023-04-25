"""Visualize the stimuli chosen with the data/select_stimuli.py script based on streamlit.

To see the stimuli, run the following command: streamlit run visualize_selected_stimuli.py
"""
# Imports
import requests
import streamlit as st
from datasets import load_from_disk
from PIL import Image

from compositionality_study.constants import VG_COCO_SELECTED_STIMULI_DIR

# Load the dataset
dataset = load_from_disk(VG_COCO_SELECTED_STIMULI_DIR)

# TODO visualize sorted by complexity
# Iterate over the examples and display them
for example in dataset:
    # Display the image
    try:
        image = Image.open(requests.get(example["vg_url"], stream=True).raw)  # noqa
        st.image(image, caption=example["sentences_raw"])
    except:  # noqa
        st.warning(f"Failed to load image for example with caption: {example['sentences_raw']}")
