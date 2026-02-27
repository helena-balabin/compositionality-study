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
    os.getenv("LARGE_STORAGE_PATH") or "~/.cache/huggingface", "huggingface/hub"  # noqa
)

# Directory for local Visual Genome files
COCO_DIR = os.path.join(os.getenv("LARGE_STORAGE_PATH") or DATA_DIR, "coco")
os.makedirs(COCO_DIR, exist_ok=True)
# Directory for object segmentation annotation files
COCO_OBJ_SEG_DIR = os.path.join(COCO_DIR, "coco_annotations")
COCO_IMAGE_DIR = os.path.join(COCO_DIR, "coco_images")
COCO_PREP_TEXT_GRAPH_DIR = os.path.join(COCO_DIR, "coco_preprocessed_text_graphs")
COCO_PREP_IMAGE_GRAPH_DIR = os.path.join(COCO_DIR, "coco_preprocessed_text_image_graphs")
COCO_PREP_ALL = os.path.join(COCO_DIR, "coco_a_preprocessed_all")
COCO_SELECTED_STIMULI_DIR = os.path.join(COCO_DIR, "coco_328_stimuli")
THINGS_IMAGE_DIR = os.path.join(COCO_DIR, "images_things")
IMAGES_COCO_SELECTED_STIMULI_DIR = os.path.join(COCO_DIR, "coco_328_stimuli")

HF_DATASET_NAME = "coco_a_preprocessed_all"
# Path to COCO action annotations
COCO_A_ANNOT_FILE = os.path.join(COCO_DIR, "coco_annotations", "cocoa_beta2015.json")
# Directory for stimuli converted into local files for psychopy
COCO_LOCAL_STIMULI_DIR = os.path.join(COCO_DIR, "coco_252_stimuli_local")
COCO_PRACTICE_STIMULI_DIR = os.path.join(COCO_DIR, "coco_practice_stimuli")
# Directory for the behavioral experiment
BEHAV_DIR = os.path.join(DATA_DIR, "behavioral")
BEHAV_INPUT_DIR = os.path.join(BEHAV_DIR, "input")
BEHAV_OUTPUT_DIR = os.path.join(BEHAV_DIR, "output")
BEHAV_OUTPUT_IMAGES_DIR = os.path.join(BEHAV_OUTPUT_DIR, "image")
BEHAV_OUTPUT_TEXT_DIR = os.path.join(BEHAV_OUTPUT_DIR, "text")
MEMORY_TEST_DIR = os.path.join(BEHAV_DIR, "memory_test")
os.makedirs(BEHAV_DIR, exist_ok=True)
os.makedirs(BEHAV_INPUT_DIR, exist_ok=True)
os.makedirs(BEHAV_OUTPUT_DIR, exist_ok=True)
os.makedirs(BEHAV_OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(BEHAV_OUTPUT_TEXT_DIR, exist_ok=True)
os.makedirs(MEMORY_TEST_DIR, exist_ok=True)

# Directory for the image complexity models and outputs
IC9000_MODEL_DIR = os.path.join(os.getenv("LARGE_STORAGE_PATH") or MODELS_DIR, "IC9600")
IC9000_IMG_COM_OUTPUT_DIR = os.path.join(IC9000_MODEL_DIR, "img_com_output")
IC_SCORES_FILE = os.path.join(IC9000_IMG_COM_OUTPUT_DIR, "ic_scores.json")
IMG_DUMMY_DIR = os.path.join(DATA_DIR, "dummy_images")
os.makedirs(IC9000_IMG_COM_OUTPUT_DIR, exist_ok=True)

# Excluded WordNet verb synsets for action-based filtering of relationships
# Consumption had to be excluded because of food-related stimuli that had a high number of verbs but no actions
# Body verbs mostly describe wearing clothes, which is static
WN_EXCLUDED_CATEGORIES = ["verb.body", "verb.consumption"]
WN_SYNSET_FILTER = ["be", "have", "along"]
WN_PREDICATE_FILTER = ["with", "on", "in"]
# "verb.contact", "verb.change", "verb.weather", "verb.stative", "verb.possession"

# Directory for raw and preprocessed MRI data
MRI_DIR = os.path.join(os.getenv("LARGE_STORAGE_PATH") or DATA_DIR, "comp_fmri_study_2025")
BIDS_DIR = os.path.join(MRI_DIR, "bids")
PREPROC_MRI_DIR = os.path.join(MRI_DIR, "fmriprep")
MRIQC_DIR = os.path.join(MRI_DIR, "mriqc")
BETAS_DIR = os.path.join(MRI_DIR, "betas")
MEMORY_TEST_DIR = os.path.join(MRI_DIR, "memory_test")
MVPA_DIR = os.path.join(MRI_DIR, "mvpa")
os.makedirs(BIDS_DIR, exist_ok=True)
os.makedirs(PREPROC_MRI_DIR, exist_ok=True)