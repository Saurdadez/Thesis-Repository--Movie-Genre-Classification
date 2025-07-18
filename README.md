
## 🚀 Deployment

The model is deployed as a Flask web service. Users can upload background music clips, and the service returns the predicted movie genre.

To run locally:

```bash
cd app
pip install -r ../requirements.txt
python app.py
| Model                 | Accuracy | Precision | Recall  | F1-Score |
| --------------------- | -------- | --------- | ------- | -------- |
| MLFNN + MFCC          | 79%      | 78%       | 77%     | 77%      |
| **CNN + MLFNN + Mel** | **89%**  | **88%**   | **89%** | **88%**  |
🧪 Dataset
A balanced dataset with 6000 samples (1000 per genre). Each sample is a 10-second audio clip extracted from background scores of various movies.

📌 Note: We used custom audio segmentation and ensured genre purity.

🛠 Requirements
Python 3.8+

TensorFlow / PyTorch

Librosa

NumPy, Pandas

Matplotlib, Seaborn

Flask (for deployment)

pip install -r requirements.txt
