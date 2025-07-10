from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from proposed_model import classify_proposed
from baseline_model import classify_baseline
app = Flask(__name__)
CORS(app)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/howtouse')
def howtouse():
    return render_template('howtouse.html')

@app.route('/predict', methods=['POST'])

def predict():
    try:
        if "file" not in request.files:
            print("❌ No file uploaded")  # Debugging
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            print("❌ No file selected")  # Debugging
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file temporarily
        temp_path = "temp_audio.mp3"
        file.save(temp_path)
        print(f"✅ File saved at: {temp_path}")  # Debugging

        # Classify the saved audio file
        proposed_result = classify_proposed(temp_path)
        baseline_result = classify_baseline(temp_path)

        # Clean up: Remove the temp file after processing
        os.remove(temp_path)
        print(f"🗑️ Temp file deleted: {temp_path}")  # Debugging
        print(f"✅ Prediction result: {proposed_result}")
        print(f"✅ Prediction result: {baseline_result}")

        return jsonify({
            "Proposed Model": proposed_result,
            "Baseline Model": baseline_result
        })

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")  # Debugging
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)