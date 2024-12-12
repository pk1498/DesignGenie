from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image
import cv2
import numpy as np
from lama import LamaPredictor


depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")


def estimate_depth(image_path):
    image = Image.open(image_path)
    inputs = depth_processor(images=image, return_tensors="pt")
    depth_map = depth_model(**inputs).logits.squeeze().detach().numpy()
    return depth_map

# LaMa model setup
lama = LamaPredictor(model_dir='path_to_lama_model')

def inpaint_with_lama(image, mask):
    inpainted_image = lama.predict(image, mask)
    return inpainted_image



from rembg import remove
import streamlit as st

@st.cache_resource
def load_bg_model():
    from rembg.bg import new_session
    return new_session(model_name="u2netp")

bg_model = load_bg_model()

def load_lama_model(ckpt_path):
    model = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    return model

lama_model = load_lama_model("path_to_your_model.ckpt")