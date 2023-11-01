import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("visual_search_similarity.keras")

# Define the target image size expected by the model
target_image_size = (10,10)

st.title("Visual Similarity Finder")

uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    try:
        # Read the uploaded image using PIL
        image = Image.open(uploaded_image)

        # Resize the image to match the model's expected input size
        image = image.resize(target_image_size)

        # Convert the image to a NumPy array
        image = image.convert('L')
        image = np.array(image)
        
        # Normalize the pixel values to the [0, 1] range
        image = image.astype('float32') / 255.0

        # Add a batch dimension to the image
        image = np.expand_dims(image, axis=0)

        # Make predictions with the model
        predictions = model.predict(image)

        #class probabilites
        predicted_class = np.argmax(predictions, axis=1)

        # Display the class and corresponding probability
        st.write(f'Predicted Class: {predicted_class[0]}')
        st.write(f'Class Probabilities: {predictions[0]}')

        st.write(predictions)
    except Exception as e:
        st.error(f'Failed to load or process the image: {str(e)}')
