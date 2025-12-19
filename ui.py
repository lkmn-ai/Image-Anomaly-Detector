import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Anomaly Detector")

st.title("🧠 Image Anomaly Detection")
st.write("Upload an image and click Detect")

@st.cache_resource
def load_model_and_threshold():
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model = load_model("autoencoder.h5", compile=False)
    threshold = np.load("threshold.npy")
    return model, threshold

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

def preprocess(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image).astype("float32") / 255.0

    if image.mean() > 0.5:
        image = 1.0 - image

    return image.reshape(1, 28, 28, 1) # batchsize, w, h, channels(first it will be RGB 3 -> 1 channel coz the model accepts)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if st.button("Detect"):
        with st.spinner("Loading model..."):
            model, threshold = load_model_and_threshold()

        img = preprocess(image)
        recon = model.predict(img)
        error = np.mean((img - recon) ** 2)

        if error > threshold:
            st.error(" ANOMALOUS, YOU ARE NOT GOOD")
        else:
            st.success(" NORMAL, YOUR MODEL IS PERFECT")

        st.metric("Reconstruction Error", f"{error:.6f}")
        st.metric("Threshold", f"{threshold:.6f}")
