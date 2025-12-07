import streamlit as st
from PIL import Image
import numpy as np
import cv2

@st.cache_resource
def Model_load(name):
    from tensorflow.keras.models import load_model
    return load_model(name)

def preprocess_image(image):
    img_arr = np.array(image)
    img = cv2.resize(img_arr, (224,224))

    if img.shape != (224,224,3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img / 255.0
    img = np.expand_dims(img, 0)
    return img

model = Model_load("mask_Model_01.keras")

st.set_page_config(page_title="FaceMask Detector", page_icon="😷")

st.title("Face Mask Detection System !!")
st.write("( Upload an image of a human and model will predict weather he/she is wearing mask or not.. )")
st.text('')
st.text('')
uploaded_file = st.file_uploader("Upload an image (😷):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            if prediction > 0.6: 
                st.error("No Mask")
            else:
                st.success("Mask Detected")