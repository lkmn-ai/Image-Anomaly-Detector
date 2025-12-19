from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Image Anomaly Detection API")

# Load trained model and threshold
autoencoder = load_model("autoencoder.h5")
threshold = np.load("threshold.npy")

def preprocess_image(image_bytes):
    """
    Convert uploaded image to 28x28 grayscale tensor
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.post("/detect")
async def detect_anomaly(file: UploadFile = File(...)):
    """
    Upload an image and detect anomaly
    """
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)

    reconstructed = autoencoder.predict(image)
    mse = float(np.mean((image - reconstructed) ** 2))

    result = "ANOMALOUS" if mse > threshold else "NORMAL"

    return {
        "result": result,
        "reconstruction_error": mse,
        "threshold": float(threshold)
    }
