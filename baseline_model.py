import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import joblib  # For loading the scaler
import json
import timeit


# Load the baseline model
MODEL_PATH = "baseline_model_40epoch.h5"  # Ensure this file exists
SCALER_PATH = "scaler.pkl"  # Ensure the scaler exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file '{SCALER_PATH}' not found.")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)  # Load the same scaler used in training


# Load label dictionary (assuming you saved it in JSON format)
LABEL_DICT_PATH = "label_dict.json"  # Replace with actual path
with open(LABEL_DICT_PATH, "r") as json_file:
    label_dict = json.load(json_file)
    print("Loaded labels:", label_dict)


# Function to extract audio features
def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    
    # Time-Domain Features
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    
    # Pitch-Based Features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # Frequency-Domain Features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    
    # Energy-Based Features
    sub_band_energy = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return np.hstack([rms, zcr, pitch_mean, spectral_centroid, spectral_bandwidth, sub_band_energy, mfccs_mean])

# Function to classify audio using the baseline model
def classify_baseline(file_path):
    start_time = timeit.default_timer()  # Start timing
    
    features = extract_features(file_path)
    features = scaler.transform([features])  # Normalize using the same scaler
    
    # Predict genre probabilities
    prediction_prob = model.predict(features)[0]

    end_time = timeit.default_timer()  # End timing
    prediction_time = end_time - start_time  # Calculate time taken

    # Map predictions to genre names
    genre_names = {idx: genre for genre, idx in label_dict.items()}
    predicted_percentages = {genre_names[idx]: float(prob) * 100 for idx, prob in enumerate(prediction_prob)}

    # Get the most probable genre
    predicted_genre = max(predicted_percentages, key=predicted_percentages.get)

    return {
        "Predicted Genre": predicted_genre,
        "Prediction Time (seconds)": round(prediction_time, 4),
        "Predicted Percentages": {genre: round(prob, 2) for genre, prob in predicted_percentages.items()}
    }
