
import os
import numpy as np
import tensorflow as tf
import librosa

# Load model
model = tf.keras.models.load_model("emotion_model.h5")
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

test_folder = "test_audio"  # Folder with test .wav files

for fname in os.listdir(test_folder):
    if fname.endswith('.wav'):
        path = os.path.join(test_folder, fname)
        features = extract_features(path)
        prediction = model.predict(features)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        print(f"{fname} -> {predicted_emotion}")
