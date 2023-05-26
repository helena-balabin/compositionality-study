"""Constants for the compositionality study repository."""

import os

from dotenv import load_dotenv

# Use dotenv to properly load device-dependent constants
load_dotenv()

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

# Directory for data, logs, models, notebooks
DATA_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "data")
LOGS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "logs")
MODELS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "models")
NOTEBOOKS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "notebooks")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# Directory for visualizations
VISUALIZATIONS_DIR = os.path.join(DATA_DIR, "visualizations")
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Huggingface cache dirs
LARGE_DATASET_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_STORAGE_PATH") or "~/.cache/huggingface", "huggingface/datasets"
)
LARGE_MODELS_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_STORAGE_PATH") or "~/.cache/huggingface", "huggingface/transformers"
)
LARGE_HUB_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_STORAGE_PATH") or "~/.cache/huggingface", "huggingface/hub"
)

# Directory for local Visual Genome files
VG_DIR = os.path.join(os.getenv("LARGE_STORAGE_PATH") or DATA_DIR, "vg")
os.makedirs(VG_DIR, exist_ok=True)
# When using local VG files, place the image_data.json file in the VG_DIR
VG_METADATA_FILE = os.path.join(VG_DIR, "image_data.json")
# File paths for objects and relationships
VG_OBJECTS_FILE = os.path.join(VG_DIR, "data_release", "objects.json")
VG_RELATIONSHIPS_FILE = os.path.join(VG_DIR, "data_release", "relationships.json")
# Directory for object segmentation annotation files
VG_COCO_OBJ_SEG_DIR = os.path.join(VG_DIR, "coco_annotations")
# This is the directory where the filtered VG/COCO dataset with captions will be stored
VG_COCO_OVERLAP_DIR = os.path.join(VG_DIR, "vg_coco_overlap")
VG_COCO_PREPROCESSED_TEXT_DIR = os.path.join(VG_DIR, "vg_coco_preprocessed_text")
VG_COCO_PREP_TEXT_GRAPH_DIR = os.path.join(VG_DIR, "vg_coco_preprocessed_graph")
VG_COCO_PREP_TEXT_IMG_SEG_DIR = os.path.join(VG_DIR, "vg_coco_preprocessed_img_seg")
VG_COCO_SELECTED_STIMULI_DIR = os.path.join(VG_DIR, "vg_coco_80_stimuli")
