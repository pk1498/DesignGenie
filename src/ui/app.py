import random
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.scene_classification.FinalModel import ConvClassifier
import os
import zipfile
import urllib.request
from src.utils.common_functions import get_faster_rcnn_model, prepare_image, draw_boxes_ui, get_file_count

# GitHub Release URL for the zipped model files
MODEL_ZIP_URL = "https://github.com/pk1498/final-models/releases/download/FinalModels/FinalModels.zip"


def download_and_extract_once():
    persistent_cache_dir = os.path.expanduser("~/.streamlit_model_cache")
    os.makedirs(persistent_cache_dir, exist_ok=True)

    zip_path = os.path.join(persistent_cache_dir, "models.zip")
    extract_dir = os.path.join(persistent_cache_dir, "models")

    if not os.path.exists(extract_dir):
        st.info("Downloading model files...")
        urllib.request.urlretrieve(MODEL_ZIP_URL, zip_path)
        st.info("Extracting model files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    return extract_dir


with st.spinner("Loading Models, this should take a while..."):
    cache_dir = download_and_extract_once()

# Paths to individual model files
sc_model_path = os.path.join(cache_dir, "FinalModels/vit_b_16.pth")
od_model_path = os.path.join(cache_dir, "FinalModels/fasterrcnn_resnet50_epoch4.pth")

# Title and description
st.title("Design Genie - Virtual Interior Design Tool")
st.markdown(
    """
    **Upload a photo of your room to get personalized furniture and decor recommendations!**
    """
)

# File upload section
st.markdown("### Upload a Room Image")
uploaded_file = st.file_uploader("Drag and drop file here (Max 200MB, JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Room Image")

    # Simulating Room Identification (replace this with model inference later)
    st.markdown("### Room Identification")

    image = Image.open(uploaded_file)
    device = torch.device("cpu")
    sc_model = ConvClassifier().to(device)
    sc_model.load_state_dict(torch.load(sc_model_path, map_location=device))
    sc_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by the ViT model
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(  # Normalize using ImageNet's mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    with st.spinner("Analyzing the room..."):
        with torch.no_grad():
            output = sc_model(image_tensor)
            probabilities = torch.softmax(output[0], dim=0)
            confidence_score, predicted_class_idx = torch.max(probabilities, dim=0)

    class_labels = ['artstudio', 'bathroom', 'bedroom', 'children_room', 'closet', 'computerroom', 'dining_room',
                    'gameroom', 'kitchen', 'livingroom', 'locker_room', 'meeting_room']
    room_type = class_labels[predicted_class_idx]

    # room_type = "Bedroom"  # Example prediction
    # confidence_score = 0.9123  # Example confidence score
    st.success(f"This is a **{room_type}**! [Confidence score – {confidence_score:.4f}]")

    # Simulating Object Detection (replace this with actual detection model)
    st.markdown("### Detected Objects")

    od_model = get_faster_rcnn_model(od_model_path)
    image_tensor = prepare_image(uploaded_file)

    room_decor = {
        "livingroom": ["Sofa", "Table", "Dining Set", "Desk", "Photo Frame", "Floor lamp", "Chair", "Vase"],
        "bedroom": ["Bed", "Desk", "Wardrobe", "Photo Frame", "Floor lamp", "Vase"],
        "kitchen": ["Kitchen shelf", "Vase"],
        "bathroom": ["Bathroom shelf"],
        "children_room": ["Bed", "Desk", "Table", "Wardrobe", "Photo Frame", "Chair"],
        "dining_room": ["Desk", "Dining Set", "Photo Frame", "Vase"]
    }

    with st.spinner("Detecting Objects..."):
        with torch.no_grad():
            prediction = od_model(image_tensor)
        predicted_objects, annotated_image = draw_boxes_ui(image, prediction, fig_size=(12, 10))

    st.image(annotated_image, caption="Detected Objects")
    total_required_objects = room_decor[room_type]
    actual_required_objects = list(set(total_required_objects) - set(predicted_objects))

    # detected_objects = ["Bed", "Cupboard", "Side Table"]  # Example objects
    detected_objects = predicted_objects
    st.info(f"The current image has these objects – {', '.join(detected_objects)}")

    # Recommendations
    st.markdown("### Recommendations")
    # recommendations = ["Dressing Table", "Lamp", "Rug", "Bedside Runner"]  # Example recommendations
    recommendations = actual_required_objects

    st.write("Select the items you would like to add to your room:")
    selected_recommendations = st.multiselect(
        "Available Recommendations:",
        options=recommendations,
        default=[],
        help="Choose the items you want to add to your room."
    )

    if selected_recommendations:
        st.success(f"You selected: {', '.join(selected_recommendations)}")
    else:
        st.warning("No items selected yet!")

    # # Theme Selection
    # st.markdown("### Choose a Theme for Further Recommendations")
    # themes = ["Modern", "Minimalist", "Bohemian", "Vintage"]
    # selected_theme = st.radio("Select a Theme:", themes)

    # Generate button
    if st.button("Generate Recommendations"):
        base_image_dir = "../../scraper/ikea_images_new"
        st.markdown("### Generated Images")
        # st.write(f"Here are 3 placeholder images generated for the **{selected_theme}** theme with your selections: "
        #          f"{', '.join(selected_recommendations) if selected_recommendations else 'No items selected.'}")
        st.write(f"Here are images for the **{selected_recommendations}:**")

        if selected_recommendations:
            with st.spinner("Generating Images..."):
                for decor_item in selected_recommendations:
                    st.subheader(f"Images for: {decor_item}")
                    decor_item_dir = os.path.join(os.path.dirname(__file__), base_image_dir, decor_item)
                    if os.path.exists(decor_item_dir):
                        image_files = [f for f in os.listdir(decor_item_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                        if len(image_files) >= 5:
                            selected_images = random.sample(image_files, 5)
                        else:
                            selected_images = image_files

                        cols = st.columns(5)
                        # num_images_decor = get_file_count(decor_item)
                        for i, img_file in enumerate(selected_images):
                            with cols[i]:
                                try:
                                    img_path = os.path.join(decor_item_dir, img_file)
                                    img = Image.open(img_path)
                                    st.image(img, caption=f"{decor_item} Image {i + 1}")
                                except Exception as e:
                                    st.error(f"Could not load image {i + 1} for {decor_item}: {e}")
                    else:
                        st.warning(f"No images found for '{decor_item}' in the directory: {decor_item_dir}")

        else:
            st.warning("No recommendations selected!")

