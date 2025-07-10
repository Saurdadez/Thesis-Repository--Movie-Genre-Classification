import librosa
import numpy as np
import json
import timeit
from tensorflow.keras.models import load_model 
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

# Load the saved model
try:
    model = load_model("Proposedmodel_30_true.h5")  # Ensure the correct path
    print("✅ Model loaded successfully!")  # Debugging print
except Exception as e:  
    print("❌ Error loading model:", e)

# Load label dictionary (assuming you saved it in JSON format)
LABEL_DICT_PATH = "label_dict.json"  # Replace with actual path
with open(LABEL_DICT_PATH, "r") as json_file:
    label_dict = json.load(json_file)
    print("📂 Loaded labels:", label_dict)

# Feature extraction function
def extract_mel_spectrogram(audio_path, n_mels=128, hop_length=512):
    print(f"🔍 Processing file: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)
    print(f"🎵 Sample rate: {sr}, Audio length: {len(y)/sr:.2f} sec")
    y = y.astype(np.float32)
    
    # Trim to 60 sec like training
    target_samples = sr * 60
    if len(y) > target_samples:
        y = y[:target_samples]
    print(f"✂️ Trimmed audio length: {len(y)/sr:.2f} sec")

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
    print(f"📊 Mel-Spectrogram shape: {mel_spec_db.shape}")

    # Save as image & reload (same as training pipeline)
    img_path = "temp_spec.png"
    fig, ax = plt.subplots(figsize=(1.28, 1.28), dpi=100)
    ax.set_axis_off()
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, cmap='magma')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = Image.open(img_path).convert("L")  # Convert to grayscale like training
    img = img.resize((128, 128))
    os.remove(img_path)
    print(f"🖼️ Processed image shape: {np.array(img).shape}")

    return np.array(img) / 255.0  # Normalize pixel values

# Function to classify music with prediction time
def classify_proposed(file_path):
    start_time = timeit.default_timer()  # Start timing

    # Extract features
    mel_spectrogram = extract_mel_spectrogram(file_path)
    if mel_spectrogram is None:
        return {"Error": "Feature extraction failed"}

    # Reshape for model input (assuming [batch, height, width, channels])
    input_features = np.expand_dims(mel_spectrogram, axis=-1)  # Add channel dimension
    input_features = np.expand_dims(input_features, axis=0)  # Add batch dimension
    print(f"📏 Final input shape before model: {input_features.shape}")

    # Predict genre probabilities
    prediction_prob = model.predict(input_features)[0]
    end_time = timeit.default_timer()  # End timing
    prediction_time = end_time - start_time  # Calculate time taken

    # Map predictions to genre names
    genre_names = {idx: genre for genre, idx in label_dict.items()}
    predicted_percentages = {genre_names[idx]: float(prob) * 100 for idx, prob in enumerate(prediction_prob)}

    # Get the most probable genre
    predicted_genre = max(predicted_percentages, key=predicted_percentages.get)

    print(f"🎯 Predicted Genre: {predicted_genre}")
    return {
        "Predicted Genre": predicted_genre,
        "Prediction Time (seconds)": prediction_time,
        "Predicted Percentages": {genre: round(prob, 2) for genre, prob in predicted_percentages.items()}
    }