"""Generate images from the text stimuli in the dataset using a text-to-image model (as a qualitative evaluation)."""
import os

import click
import torch
from datasets import load_from_disk
from diffusers import DiffusionPipeline
from tqdm import tqdm

from compositionality_study.constants import (
    LARGE_DATASET_STORAGE_PATH,
    LARGE_MODELS_STORAGE_PATH,
    TEXT_TO_IMAGE_OUTPUT_DIR,
    VG_COCO_SELECTED_STIMULI_DIR,
)


@click.command()
@click.option(
    "--stimuli_ds_path",
    default=VG_COCO_SELECTED_STIMULI_DIR,
    help="Path to the stimuli dataset to use",
    type=str,
)
@click.option(
    "--model_name",
    default="segmind/SSD-1B",
    help="Name of the text-to-image model to use",
    type=str,
)
@click.option("--prompt_prefix", default="", type=str)
@click.option("--n_images", default=5, type=int)
@click.option("--large_model_dir", default=LARGE_MODELS_STORAGE_PATH, type=str)
@click.option("--large_dataset_dir", default=LARGE_DATASET_STORAGE_PATH, type=str)
@click.option("--output_dir", default=TEXT_TO_IMAGE_OUTPUT_DIR, type=str)
def generate_images_from_text(
    stimuli_ds_path: str,
    model_name: str,
    prompt_prefix: str = "",
    n_images: int = 5,
    large_model_dir: str = LARGE_MODELS_STORAGE_PATH,
    large_dataset_dir: str = LARGE_DATASET_STORAGE_PATH,
    output_dir: str = TEXT_TO_IMAGE_OUTPUT_DIR,
) -> None:
    """Generate images from the text stimuli in the dataset using a text-to-image model.

    :param stimuli_ds_path: Path to the stimuli dataset to use
    :type stimuli_ds_path: str
    :param model_name: Name of the text-to-image model to use
    :type model_name: str
    :param prompt_prefix: Prefix to add to the text-to-image prompt, defaults to ""
    :type prompt_prefix: str
    :param n_images: Number of images to generate per sentence, defaults to 5
    :type n_images: int
    :param large_model_dir: Directory for saving large models, defaults to LARGE_MODELS_STORAGE_PATH
    :type large_model_dir: str
    :param large_dataset_dir: Directory for saving large datasets, defaults to LARGE_DATASET_STORAGE_PATH
    :type large_dataset_dir: str
    :param output_dir: Output directory for saving the generated images, defaults to TEXT_TO_IMAGE_OUTPUT_DIR
    :type output_dir: str
    """
    # 1. Load the stimuli dataset
    stimuli_ds = load_from_disk(stimuli_ds_path)

    # 2. Load the text-to-image model
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=large_model_dir,
    )
    pipe.to("cuda")

    # 3. Create the output directory
    model_specific_output_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    os.makedirs(model_specific_output_dir, exist_ok=True)

    # 4. Generate an image for each sentence
    for example in tqdm(stimuli_ds, desc=f"Generating images for {model_name}"):
        images = pipe(
            prompt=prompt_prefix + example["sentences_raw"],
            num_images_per_prompt=n_images,
        ).images

        for i, image in enumerate(images):
            # Save the generated image
            image.save(os.path.join(model_specific_output_dir, f"tti_{example['cocoid']}_{i}.png"))


@click.group()
def cli() -> None:
    """Generate images from the text stimuli in the dataset using a text-to-image model."""


if __name__ == "__main__":
    cli.add_command(generate_images_from_text)
    cli()
