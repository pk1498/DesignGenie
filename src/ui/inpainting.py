from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import cv2
import numpy as np
import torch


def preprocess_image(image_path, mask_path, target_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image = image.resize(target_size)
    mask = mask.resize(target_size)

    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0

    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

    return image.unsqueeze(0), mask.unsqueeze(0)


def inpaint_image(model, image, mask):
    # Concatenate image and mask for input
    input_data = torch.cat((image, mask), dim=1)  # Assuming model expects (N, 4, H, W)
    with torch.no_grad():
        inpainted = model(input_data)
    return inpainted.squeeze().permute(1, 2, 0).numpy()  # Convert back to HWC
