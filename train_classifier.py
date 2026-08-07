"""
train_classifier.py — train SoundSight's classifier on YOUR sounds.

This is transfer learning, the same technique used across computer
vision: we keep YAMNet's pretrained convolutional layers (trained on
millions of AudioSet clips) as a fixed feature extractor, and train
only a small classifier head on top of its embeddings using your own
recorded examples. You need far fewer samples this way than training
a model from scratch.

Expected folder structure:

  data/
    adhan/       *.wav
    doorbell/    *.wav
    fire_alarm/  *.wav
    call/        *.wav
    ambulance/   *.wav
    other/       *.wav   <- IMPORTANT: background noise / random
                             household sounds, so the model learns
                             what NOT to alert on.

Aim for at least 20-30 clips per category (2-3 seconds each) to start.
Use record_samples.py to record them from your microphone.

Run:
    python train_classifier.py
"""

import os
import glob
import json

import numpy as np
import soundfile as sf
import resampy
import tensorflow_hub as hub
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib

DATA_DIR = "data"
MODEL_DIR = "models"
SAMPLE_RATE = 16000


def load_wav(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = resampy.resample(wav, sr, SAMPLE_RATE)
    return wav


def main():
    print("Loading YAMNet...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    categories = sorted(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    if not categories:
        print(f"No category folders found inside '{DATA_DIR}/'. "
              f"Run record_samples.py first, or create the folders manually.")
        return
    if "other" not in categories:
        print("WARNING: no 'other' folder found. Add background/negative "
              "examples so the model learns what NOT to alert on.")

    print("Categories found:", categories)

    X, y = [], []
    for label in categories:
        files = glob.glob(os.path.join(DATA_DIR, label, "*.wav")) + glob.glob(os.path.join(DATA_DIR, label, "*.mp3"))
        print(f"  {label}: {len(files)} files")
        for f in files:
            wav = load_wav(f)
            if len(wav) < SAMPLE_RATE * 0.5:
                continue
            _, embeddings, _ = yamnet(wav)
            X.append(np.mean(embeddings.numpy(), axis=0))
            y.append(label)

    if len(X) < 10:
        print("Not enough samples yet to train. Record more clips and try again.")
        return

    X = np.array(X)
    y = np.array(y)
    print(f"Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.2%}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.joblib"))
    with open(os.path.join(MODEL_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump([str(c) for c in clf.classes_], f, ensure_ascii=False)

    print("\nSaved model to models/classifier.joblib")
    print("Restart app.py — it will automatically load your custom classifier.")


if __name__ == "__main__":
    main()
