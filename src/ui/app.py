import streamlit as st

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
    room_type = "Bedroom"  # Example prediction
    confidence_score = 0.9123  # Example confidence score
    st.success(f"This is a **{room_type}**! [Confidence score – {confidence_score:.4f}]")

    # Simulating Object Detection (replace this with actual detection model)
    st.markdown("### Detected Objects")
    detected_objects = ["Bed", "Cupboard", "Side Table"]  # Example objects
    st.info(f"The current image has these objects – {', '.join(detected_objects)}")

    # Recommendations
    st.markdown("### Recommendations")
    recommendations = ["Dressing Table", "Lamp", "Rug", "Bedside Runner"]  # Example recommendations
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

    # Theme Selection
    st.markdown("### Choose a Theme for Further Recommendations")
    themes = ["Modern", "Minimalist", "Bohemian", "Vintage"]
    selected_theme = st.radio("Select a Theme:", themes)

    # Generate button
    if st.button("Generate Recommendations"):
        # Placeholder for generation process
        st.markdown("### Generated Images")
        st.write(
            f"Here are 3 placeholder images generated for the **{selected_theme}** theme with your selections: {', '.join(selected_recommendations) if selected_recommendations else 'No items selected.'}")

        # Display uploaded image 3 times as placeholder
        st.markdown("Click on the images to view them in full size.")
        cols = st.columns(3)
        for i in range(3):
            with cols[i]:
                st.image(uploaded_file, caption=f"Generated Image {i + 1}")