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
