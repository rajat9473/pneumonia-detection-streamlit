import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="sachin2042yadav/Pneumonia-detection", filename="pneumonia_model.h5")
    return load_model(model_path)

model = load_model()

st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = img.resize((150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    prediction = model.predict(x)
    if prediction[0][0] > 0.5:
        st.write("Prediction: **Normal**")
    else:
        st.write("Prediction: **Pneumonia**")
