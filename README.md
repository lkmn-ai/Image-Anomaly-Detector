# Image-Anomaly-Detector
Autoencoder-based image anomaly detection using FastAPI and TensorFlow

Image Anomaly Detection API

-> How it works
1) Model trained only on normal images
2) Reconstructs input image
3) Calculates reconstruction error (MSE)
4) If error > threshold → ANOMALOUS

-> Tech Stack
1) FastAPI
2) TensorFlow
3) Numpy
4) Streamlit (UI)

Run API
bash: uvicorn app:app --reload
Run UI
bash: streamlit run ui.py

API Endpoint
POST /detect

Example Response
{
  "result": "ANOMALOUS",
  "reconstruction_error": 0.021,
  "threshold": 0.015
}
