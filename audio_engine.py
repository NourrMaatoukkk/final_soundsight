"""
audio_engine.py — the actual "brain" of SoundSight.

Computer-vision approach to audio (this is the core idea):
Every ~1 second of microphone audio is converted into a log-mel
spectrogram — literally a 2D image of "how loud each frequency is,
over time". That image is fed into YAMNet, a convolutional neural
network (the same architecture family used for image classification,
based on MobileNet) pretrained on millions of labelled sound clips
from Google's AudioSet. YAMNet's last hidden layer gives us a
1024-number "fingerprint" (embedding) of the sound — exactly the way
a CV model produces an embedding of an image. A small classifier
head then turns that fingerprint into one of SoundSight's categories.

This file runs continuously in a background thread, listening to the
microphone and updating `engine.latest` / `engine.log` in real time.
"""

import time
import threading
import collections
import csv
import os
import json

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import joblib

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

CATEGORIES = {
    "adhan":      {"label_ar": "الأذان",           "color": "#ffc857"},
    "doorbell":   {"label_ar": "جرس الباب / خبط",   "color": "#35d0c0"},
    "fire_alarm": {"label_ar": "إنذار حريق",         "color": "#ff5a45"},
    "call":       {"label_ar": "حد بينادي عليك",     "color": "#7ed394"},
    "ambulance":  {"label_ar": "صفارة إسعاف",         "color": "#8c7cff"},
    "other":      {"label_ar": "صوت عادي",           "color": "#5f6672"},
}

# Fallback mapping from YAMNet/AudioSet class names to our 5 categories.
# Used automatically until you train a custom classifier (see train_classifier.py).
GENERIC_MAP = {
    "Doorbell": "doorbell",
    "Ding-dong": "doorbell",
    "Knock": "doorbell",
    "Smoke detector, smoke alarm": "fire_alarm",
    "Fire alarm": "fire_alarm",
    "Alarm": "fire_alarm",
    "Siren": "ambulance",
    "Ambulance (siren)": "ambulance",
    "Civil defense siren": "ambulance",
    "Speech": "call",
    "Shout": "call",
    "Male speech, man speaking": "call",
    "Female speech, woman speaking": "call",
    "Chant": "adhan",
    "Singing": "adhan",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")


class SoundEngine:
    def __init__(self, on_update=None):
        print("Loading YAMNet (first run downloads ~15MB, then it's cached locally)...")
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self.yamnet.class_map_path().numpy().decode("utf-8")
        self.class_names = self._load_class_names(class_map_path)

        self.custom_clf = None
        self.custom_labels = None
        self._load_custom_classifier()

        self.buffer = collections.deque(maxlen=WINDOW_SAMPLES)
        self.lock = threading.Lock()
        self.on_update = on_update
        self.running = False

        self.latest = {
            "category": "other",
            "label_ar": CATEGORIES["other"]["label_ar"],
            "color": CATEGORIES["other"]["color"],
            "confidence": 0.0,
            "mode": "generic",
            "level": 0.0,
            "ts": time.strftime("%H:%M:%S"),
        }
        self.log = collections.deque(maxlen=30)
        self.mode = "custom" if self.custom_clf is not None else "generic"

    def _load_class_names(self, path):
        names = []
        with tf.io.gfile.GFile(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["display_name"])
        return names

    def _load_custom_classifier(self):
        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(LABELS_PATH):
            self.custom_clf = joblib.load(CLASSIFIER_PATH)
            with open(LABELS_PATH, encoding="utf-8") as f:
                self.custom_labels = json.load(f)
            print(f"Loaded custom classifier. Categories: {self.custom_labels}")
        else:
            print("No custom classifier found — using generic AudioSet mapping. "
                  "See train_classifier.py to train one on your own sounds.")

    def _audio_callback(self, indata, frames, time_info, status):
        mono = indata[:, 0]
        with self.lock:
            self.buffer.extend(mono)

    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=self._audio_callback,
        )
        self.stream.start()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()

    def _loop(self):
        while self.running:
            time.sleep(HOP_SECONDS)
            with self.lock:
                if len(self.buffer) < WINDOW_SAMPLES:
                    continue
                waveform = np.array(self.buffer, dtype=np.float32)

            level = float(np.sqrt(np.mean(waveform ** 2)))
            if level < 0.01:
                self._set_result("other", 0.0, level)
                continue

            scores, embeddings, spectrogram = self.yamnet(waveform)
            mean_scores = np.mean(scores.numpy(), axis=0)
            mean_embedding = np.mean(embeddings.numpy(), axis=0)

            if self.custom_clf is not None:
                probs = self.custom_clf.predict_proba([mean_embedding])[0]
                idx = int(np.argmax(probs))
                category = self.custom_labels[idx]
                confidence = float(probs[idx])
            else:
                category, confidence = self._generic_predict(mean_scores)

            self._set_result(category, confidence, level)

    def _generic_predict(self, mean_scores):
        top_idx = np.argsort(mean_scores)[::-1][:5]
        for i in top_idx:
            name = self.class_names[i]
            if name in GENERIC_MAP and mean_scores[i] > 0.15:
                return GENERIC_MAP[name], float(mean_scores[i])
        return "other", float(mean_scores[top_idx[0]])

    def _set_result(self, category, confidence, level):
        info = CATEGORIES.get(category, CATEGORIES["other"])
        result = {
            "category": category,
            "label_ar": info["label_ar"],
            "color": info["color"],
            "confidence": round(confidence, 2),
            "mode": self.mode,
            "level": round(min(level * 8, 1.0), 2),
            "ts": time.strftime("%H:%M:%S"),
        }
        self.latest = result
        if category != "other" and confidence > 0.25:
            self.log.appendleft(result)
        if self.on_update:
            self.on_update(result)
