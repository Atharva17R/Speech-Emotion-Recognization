
from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels (update if needed)
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

app = Flask(__name__)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = 'temp.wav'
    file.save(file_path)

    try:
        features = extract_features(file_path)
        prediction = model.predict(features)
        predicted_label = emotion_labels[np.argmax(prediction)]
        return jsonify({'emotion': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
