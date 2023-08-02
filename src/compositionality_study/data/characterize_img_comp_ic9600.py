"""Characterization of image complexity based on https://github.com/tinglyfeng/IC9600."""
import io
import json
import os
from typing import Dict, Union

import click
import cv2
import torch
import torch.nn.functional as F  # noqa
import numpy as np
from datasets import Dataset, load_from_disk
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from compositionality_study.constants import IC9000_IMG_COM_OUTPUT_DIR, IC9000_MODEL_DIR, IMG_DUMMY_DIR
from compositionality_study.models.icnet import ICNet

# Load the ICNet model
model = ICNet()
model.load_state_dict(torch.load(os.path.join(IC9000_MODEL_DIR, "ck.pth"), map_location=torch.device('cpu')))
model.eval()
# Define inference transform
inference_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


def blend(ori_img, ic_img, alpha=0.8, cm=plt.get_cmap("magma")):
    """Blend the original image with the IC map."""
    cm_ic_map = cm(ic_img)
    heatmap = Image.fromarray((cm_ic_map[:, :, -2::-1] * 255).astype(np.uint8))
    ori_img = Image.fromarray(ori_img)
    blend = Image.blend(ori_img, heatmap, alpha=alpha)  # noqa
    blend = np.array(blend)  # noqa
    return blend


def infer_one_image(
    input_img_data: Union[str, Dict],
    output_dir: str,
    device: str = "cpu",
    save_ic_map: bool = False,
) -> float:
    """Infer the image complexity of one image.

    :param input_img_data: The path to the image, or an image data example from a preprocessed hf dataset
    :type input_img_data: Union[str, Dict]
    :param output_dir: The output_dir directory
    :type output_dir: str
    :param device: The device to use for inference, defaults to "cpu"
    :type device: str, optional
    :param save_ic_map: Whether to save the IC and blend maps, defaults to False
    :type save_ic_map: bool, optional
    :return: The predicted image complexity score (between 0 and 1)
    :rtype: float
    """
    # Define the image and the output paths
    if isinstance(input_img_data, Dict):
        img_data = io.BytesIO(input_img_data["bytes"])
        out_name = input_img_data["path"].strip(".jpg")
    else:
        img_data = input_img_data  # type: ignore
        out_name = os.path.basename(input_img_data).split('.')[0]

    with torch.no_grad():
        # Load the image
        ori_img = Image.open(img_data).convert("RGB")
        ori_height = ori_img.height
        ori_width = ori_img.width
        img = inference_transform(ori_img)
        img = img.to(device)
        img = img.unsqueeze(0)
        # Get the model predictions
        if save_ic_map:
            ic_score, ic_map = model(img)
        else:
            ic_score, _ = model(img)
        ic_score = ic_score.item()

        # Save the IC/blend maps
        if save_ic_map:
            ic_map = F.interpolate(ic_map, (ori_height, ori_width), mode='bilinear')

            # gene ic map
            ic_map_np = ic_map.squeeze().detach().cpu().numpy()
            out_ic_map_name = out_name + '_' + str(ic_score)[:7] + '.npy'
            out_ic_map_path = os.path.join(output_dir, out_ic_map_name)
            np.save(out_ic_map_path, ic_map_np)

            # gene blend map
            ic_map_img = (ic_map * 255).round().squeeze().detach().cpu().numpy().astype('uint8')
            blend_img = blend(np.array(ori_img), ic_map_img)  # noqa
            out_blend_img_name = out_name + '.png'
            out_blend_img_path = os.path.join(output_dir, out_blend_img_name)
            cv2.imwrite(out_blend_img_path, blend_img)  # noqa

        return ic_score


def infer_img_source(
    img_source: str,
    output_dir: str,
    device: str = "cpu",
    save_ic_map: bool = False,
) -> Dict[str, float]:
    """Infer image complexity for a directory or hf dataset consisting of images.

    :param img_source: Path to directory of images or to a saved hf dataset.
    :type img_source: str
    :param output_dir: Path to output_dir directory.
    :type output_dir: str
    :param device: Device to use for inference, defaults to "cpu".
    :type device: str, optional
    :param save_ic_map: Whether to save the IC and blend maps, defaults to False.
    :type save_ic_map: bool, optional
    """
    # Get images (either as hf dataset or from directory)
    try:
        imgs = load_from_disk(img_source)
    except FileNotFoundError:
        imgs = os.listdir(img_source)

    col_name = ""
    # Define a col name in the case of a dataset
    if isinstance(imgs, Dataset):
        col_name = "cocoid" if "cocoid" in imgs.column_names else "__index_level_0__"

    # Define a results dict
    results = {}

    # Loop over images
    for img in tqdm(imgs, desc="Infering images"):
        # Get the image path/data
        if isinstance(img, Dict):
            img_path = img["image"]
        else:
            img_path = os.path.join(img_source, img)  # noqa
        # Infer the image complexity
        ic_score = infer_one_image(img_path, output_dir, device=device, save_ic_map=save_ic_map)
        if isinstance(img, Dict):
            results[img[col_name]] = ic_score
        else:
            results[img.strip(".jpg")] = ic_score  # noqa

    return results


@click.command()
@click.option("--input_dir", default=IMG_DUMMY_DIR, type=str, help="Path to the directory of images.")
@click.option("--output_dir", default=IC9000_IMG_COM_OUTPUT_DIR, type=str, help="Output directory.")
@click.option("--device", default="cpu", type=str, help="Device to use for inference.")
@click.option("--save_ic_map", default=False, type=bool, help="Whether to save the IC and blend maps.")
def characterize_img_comp(
    input_dir: str = IMG_DUMMY_DIR,
    output_dir: str = IC9000_IMG_COM_OUTPUT_DIR,
    device: str = "cpu",
    save_ic_map: bool = False,
):
    """Characterize image complexity for a given image or directory of images.

    :param input_dir: Path to directory of images.
    :type input_dir: str
    :param output_dir: Path to output_dir directory.
    :type output_dir: str
    :param device: Device to use for inference, defaults to "cpu".
    :type device: str
    :param save_ic_map: Whether to save the IC and blend maps, defaults to False.
    :type save_ic_map: bool
    """
    model.to(torch.device(device))

    results = infer_img_source(input_dir, output_dir, device=device, save_ic_map=save_ic_map)

    # Save the results to a json file
    results_path = os.path.join(output_dir, "ic_scores.json")
    with open(results_path, "w") as f:
        json.dump(results, f)


@click.group()
def cli() -> None:
    """Characterize image complexity."""


if __name__ == "__main__":
    cli.add_command(characterize_img_comp)
    cli()
