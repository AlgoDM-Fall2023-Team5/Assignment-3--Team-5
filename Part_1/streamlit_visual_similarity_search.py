import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import io
import cv2

# Load the model
model = tf.keras.models.load_model("visual_search_similarity.keras")
label_set = ['Hat', 'Shoes', 'T-Shirt', 'Longsleeve', 'Dress']

def decode_labels(y, mlb):
  labels = np.array(mlb.inverse_transform(np.atleast_2d(y)))[:, 0]
  return labels

def decode_label_prob(y, classes):
  labels = []
  for i, c in enumerate(classes):
    labels.append(f'{c}: {y[i]:.2%}')
  return labels

st.title("Visual Similarity Finder")

uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
st.write(uploaded_image)

if uploaded_image is not None:
  try:
    # Convert the PIL image object to a bytes object
    # image_bytes = io.BytesIO(uploaded_image.read())

    # Read the image using tf.keras.preprocessing.image.load_img()
    # image = tf.keras.preprocessing.image.load_img(uploaded_image, color_mode='rgb', target_size=(224, 224))

    img = cv2.resize(uploaded_image,(224,224))     # resize image to match model's expected sizing
    image = img.reshape(1,224,224,3) # return the image with shaping that TF wants.

    # Convert the image to a NumPy array
    image = np.array(image)

    # Preprocess the image
    x = image.astype('float32')

    # Make a prediction
    mlb = MultiLabelBinarizer()
    mlb.fit([label_set])

    class_probs = model.predict(x)[0]

    # Decode the prediction results
    labels = decode_label_prob(class_probs, mlb.classes_)

    # Display the results
    st.write('\n'.join(labels))

  except Exception as e:
    st.error(f'Failed to load or process the image: {str(e)}')
