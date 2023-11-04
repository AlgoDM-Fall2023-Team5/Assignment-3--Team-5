import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load your VAE model
model = tf.keras.models.load_model('visual_search_vae.keras')

# Function to generate images from embeddings
def generate_images(embeddings):
    images = model.decode(embeddings).numpy()
    return images

# Create a Streamlit web app
st.title('Fashion MNIST Visual Search')

st.sidebar.title('Search Options')

# Add a slider for choosing latent space coordinates
st.sidebar.subheader('Choose Latent Space Coordinates')
latent_space_x = st.sidebar.slider('X', -3.0, 3.0, 0.0)
latent_space_y = st.sidebar.slider('Y', -3.0, 3.0, 0.0)
latent_coordinates = np.array([[latent_space_x, latent_space_y]])

# Generate and display the image from latent coordinates
st.subheader('Generated Image')
generated_image = generate_images(latent_coordinates)
st.image(generated_image.squeeze(), use_column_width=True, caption='Generated Image')

# Nearest Neighbor Search
st.sidebar.subheader('Nearest Neighbor Search')

# Add a slider to choose a query image
query_image_id = st.sidebar.slider('Select Query Image', 0, 59999, 15)
k = 6

# Function to perform nearest neighbor search
def query_nearest_neighbors(image_id, k):
    # Query the VAE model for nearest neighbors
    query_embeddings = model.enc(train_images[image_id:image_id + 1])[0]
    distances = np.linalg.norm(embeddigns - query_embeddings, axis=1)
    nearest_neighbor_indices = np.argpartition(distances, k + 1)[:k + 1]
    return nearest_neighbor_indices[1:]  # Exclude the query image itself

# Get nearest neighbors
nearest_neighbor_indices = query_nearest_neighbors(query_image_id, k)

# Display nearest neighbors
st.subheader('Nearest Neighbors')
st.image(train_images[nearest_neighbor_indices], caption=['NN 1', 'NN 2', 'NN 3', 'NN 4', 'NN 5', 'NN 6'], width=84)

# Display original query image
st.image(1 - train_images[query_image_id], caption='Query Image', width=84)

st.sidebar.text('By Your Name')

# Manifold Visualization
st.sidebar.subheader('Manifold Visualization')
st.subheader('Manifold Visualization')
st.image(1 - canvas, use_column_width=True, caption='Manifold Visualization')

