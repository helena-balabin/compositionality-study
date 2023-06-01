"""Characterization of image complexity taken from https://github.com/tinglyfeng/IC9600."""
import argparse
import os

import cv2
import torch
import torch.nn.functional as F  # noqa
import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from compositionality_study.constants import IC9000_IMG_COM_OUTPUT_DIR, IC9000_MODEL_DIR, IMG_DUMMY_DIR
from compositionality_study.models.icnet import ICNet


# TODO change to click
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default=IMG_DUMMY_DIR)
parser.add_argument('-o', '--output', type=str, default=IC9000_IMG_COM_OUTPUT_DIR)
parser.add_argument('-d', '--device', type=str, default="cpu")


def blend(ori_img, ic_img, alpha=0.8, cm=plt.get_cmap("magma")):
    cm_ic_map = cm(ic_img)
    heatmap = Image.fromarray((cm_ic_map[:, :, -2::-1] * 255).astype(np.uint8))
    ori_img = Image.fromarray(ori_img)
    blend = Image.blend(ori_img, heatmap, alpha=alpha)  # noqa
    blend = np.array(blend)  # noqa
    return blend


def infer_one_image(img_path):
    with torch.no_grad():
        ori_img = Image.open(img_path).convert("RGB")
        ori_height = ori_img.height
        ori_width = ori_img.width
        img = inference_transform(ori_img)
        img = img.to(device)
        img = img.unsqueeze(0)
        ic_score, ic_map = model(img)
        ic_score = ic_score.item()
        ic_map = F.interpolate(ic_map, (ori_height, ori_width), mode='bilinear')

        # gene ic map
        ic_map_np = ic_map.squeeze().detach().cpu().numpy()
        out_ic_map_name = os.path.basename(img_path).split('.')[0] + '_' + str(ic_score)[:7] + '.npy'
        out_ic_map_path = os.path.join(args.output, out_ic_map_name)
        np.save(out_ic_map_path, ic_map_np)

        # gene blend map
        ic_map_img = (ic_map * 255).round().squeeze().detach().cpu().numpy().astype('uint8')
        blend_img = blend(np.array(ori_img), ic_map_img)
        out_blend_img_name = os.path.basename(img_path).split('.')[0] + '.png'
        out_blend_img_path = os.path.join(args.output, out_blend_img_name)
        cv2.imwrite(out_blend_img_path, blend_img)
        return ic_score


def infer_directory(img_dir):
    imgs = os.listdir(img_dir)
    for img in tqdm(imgs):
        img_path = os.path.join(img_dir, img)
        infer_one_image(img_path)


if __name__ == "__main__":
    args = parser.parse_args()

    model = ICNet()
    model.load_state_dict(torch.load(os.path.join(IC9000_MODEL_DIR, "ck.pth"), map_location=torch.device('cpu')))
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    inference_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    if os.path.isfile(args.input):
        infer_one_image(args.input)
    else:
        infer_directory(args.input)









