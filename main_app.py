import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the full model (architecture + weights)
model = load_model('dog_breed.h5')

CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

st.title("Dog Breed Prediction")
st.markdown("Upload an image of a dog")

dog_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
submit = st.button('Predict')

if submit:
    if dog_image is not None:
        # Convert uploaded file to OpenCV image
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Show image
        st.image(opencv_image, channels="BGR")

        # Preprocess for model
        opencv_image = cv2.resize(opencv_image, (224,224))
        opencv_image = opencv_image / 255.0  # Normalize
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Prediction
        Y_pred = model.predict(opencv_image)
        predicted_class = CLASS_NAMES[np.argmax(Y_pred)]

        st.title(f"The Dog Breed is {predicted_class}")
