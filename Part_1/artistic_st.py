import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import ntpath
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from vis

# Define the StyleModel class and style-related functions here (as previously shown)

# Load the pre-trained style model
model = tf.keras.models.load_model('style_model.h5', custom_objects={'StyleModel': StyleModel})

# Function to search for images by style
def search_by_style(image_style_embeddings, images, reference_image, max_results=10):
    v0 = image_style_embeddings[reference_image]
    distances = {}
    for k, v in image_style_embeddings.items():
        d = cosine_similarity([v0], [v])[0][0]
        distances[k] = d

    sorted_neighbors = sorted(distances.items(), key=lambda x: x[1], reverse=False)

    return sorted_neighbors[:max_results]

# Streamlit app
st.title("Image Search by Artistic Style")

# Upload a reference image
st.write("Upload a reference image to search for similar images by style:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Compute style embeddings for the uploaded image
    image_tensor = load_image(image)
    style_embeddings = model(image_tensor)
    style_vector = style_to_vec(style_embeddings['style'])

    # Search for similar images by style
    st.write("Searching for similar images by style...")
    similar_images = search_by_style(image_style_embeddings, images, ntpath.basename(uploaded_file.name))

    # Display the results
    st.write("Top 10 similar images by style:")
    st.subheader("Reference Image:")
    st.image(images[ntpath.basename(uploaded_file.name)], caption="Reference Image", use_column_width=True)
    col1, col2, col3 = st.columns(3)
    for i, (img_name, _) in enumerate(similar_images[:9]):
        if i < 3:
            with col1:
                st.image(images[img_name], caption=f"Image {i + 1}", use_column_width=True)
        elif i < 6:
            with col2:
                st.image(images[img_name], caption=f"Image {i + 1}", use_column_width=True)
        else:
            with col3:
                st.image(images[img_name], caption=f"Image {i + 1}", use_column_width=True)
