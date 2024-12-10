import sys
sys.path.append("../src")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_transformations():
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])
    return transformations


def predict_image(img, model, device, dataset):
    xb = img.unsqueeze(0).to(device)         # Convert to a batch of 1
    yb = model(xb)                           # Get predictions from model
    prob, preds  = torch.max(yb, dim=1)      # Pick index with highest probability
    return dataset.classes[preds[0].item()]  # Retrieve the class label


def prepare_image(image_path): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB") # Open image 
    image_tensor = F.to_tensor(image).unsqueeze(0) #Convert image to tensor and add batch dimension 
    return image_tensor.to(device) 


def get_class_name(class_id): 
    COCO_CLASSES = {0: "Background", 1: "Bathroom shelf", 2: "Bed", 3: "Chair", 4: "Desk", 5: "Dining Set", 6: "Floor lamp", 
                    7: "Kitchen shelf", 8: "Photo Frame", 9: "Sofa", 10: "Table", 11: "Vase", 12: "Wardrobe"} 
    return COCO_CLASSES.get(class_id, "Unknown") 


#Draw bounding boxes with the correct class names and increase image size 
def draw_boxes (image, prediction, fig_size=(10, 10)): 
    boxes = prediction [0]['boxes'].cpu().numpy() # Get predicted bounding boxes 
    labels = prediction [0]['labels'].cpu().numpy() # Get predicted labels 
    scores = prediction[0]['scores'].cpu().numpy() #Get predicted scores 
    
    #Set a threshold for showing boxes (e.g., score > 0.5) 
    threshold = 0.3
    
    #Set up the figure size to control the image size 
    plt.figure(figsize=fig_size) # Adjust the figure size here 
    
    for box, label, score in zip(boxes, labels, scores): 
        if score > threshold: 
            x_min, y_min, x_max, y_max = box 
            class_name = get_class_name(label) # Get the class name 
            plt.imshow(image) # Display the image 
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')) 
            plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r') 
    plt.axis('off') #Turn off axis 
    plt.show() 

def get_faster_rcnn_model():
    # number of decor items is equal to number of classes for object detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_decor_items = 13
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_decor_items)

    model.load_state_dict(torch.load("../FasterRCNN/fasterrcnn_resnet50_epoch_(epoch + 1).pth"))
    model.to(device)
    model.eval()
    return model

