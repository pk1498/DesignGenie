import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt


# Step 1: Load the Room Image
def load_image(image_path):
    """
    Load and preprocess the input room image.
    """
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Step 2: Detect Bed Position
def detect_bed_position(image_tensor):
    """
    Detect bed positions in the room image using Faster R-CNN.
    """
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Perform object detection
    with torch.no_grad():
        predictions = model([image_tensor])[0]

    # Extract bounding boxes for 'bed' (label 65 in COCO dataset)
    bed_boxes = []
    for idx, label in enumerate(predictions['labels']):
        if label == 65:  # 'bed' class in COCO dataset
            score = predictions['scores'][idx]
            if score > 0.8:  # Confidence threshold
                bed_boxes.append(predictions['boxes'][idx].tolist())
                print(bed_boxes)

    return bed_boxes


# Step 3: Load Furniture Images
def load_furniture_image(file_path, size=(64, 64)):
    """
    Load and resize a furniture image with transparency support.
    """
    furniture = Image.open(file_path).convert('RGBA')
    furniture = furniture.resize(size)
    return furniture


# Step 4: Place Furniture Near the Bed
def place_furniture_near_bed(room_image, bed_boxes, furniture_images):
    """
    Place furniture images near detected bed positions.
    """
    room = T.ToPILImage()(room_image.squeeze(0)).convert('RGBA')

    for idx, bed_box in enumerate(bed_boxes):
        x_min, y_min, x_max, y_max = map(int, bed_box)
        bed_center_x = (x_min + x_max) // 2
        bed_center_y = (y_min + y_max) // 2
        print(x_min, y_min, x_max, y_max, bed_center_x, bed_center_y)

        furniture = furniture_images[0]
        room.paste(furniture, (bed_center_x - 100, bed_center_y - 30), furniture)
        furniture = furniture_images[1]
        room.paste(furniture, (bed_center_x + 40, bed_center_y - 30), furniture)

    return room


# Step 5: Main Workflow
def main():
    # Input room image
    room_image_path = "bedroom_sample_image.jpg"  # Replace with your room image path
    room_image = load_image(room_image_path)

    # Detect bed positions
    bed_boxes = detect_bed_position(room_image.squeeze(0))

    if not bed_boxes:
        print("No bed detected in the image.")
        return

    # Load furniture images
    bedside_table_path = "bedside_table.png"  # Replace with bedside table image
    lamp_path = "bedside_table.png"  # Replace with lamp image

    furniture_images = [
        load_furniture_image(bedside_table_path),
        load_furniture_image(lamp_path)
    ]

    # Place furniture in the room
    inpainted_room = place_furniture_near_bed(room_image, bed_boxes, furniture_images)

    # Display the final result
    plt.imshow(inpainted_room)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
