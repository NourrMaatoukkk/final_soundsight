# SoundSight — Working Prototype

This is the real, running AI engine behind SoundSight — not a mockup.
It listens to your computer's microphone, classifies the sound in real
time, and shows the result on a live dashboard styled like the product
concept. It's the same pipeline that would eventually run on the
wristband's chip; here it runs on your laptop so you (and anyone you
demo it to) can see it actually work today.

## How the AI works (in plain terms)

1. The mic captures a rolling ~1 second of audio.
2. That audio is converted into a **log-mel spectrogram** — a 2D image
   of frequency vs. time. This is the "computer vision" trick: instead
   of analyzing raw sound math, we turn sound into a picture.
3. That picture is fed into **YAMNet**, a convolutional neural network
   (Google, pretrained on ~2 million labelled sound clips) — the same
   type of architecture used for image classification, applied to
   audio spectrograms. Its output is a 1024-number "fingerprint"
   (embedding) plus a rough guess at 521 general sound classes.
4. A small classifier sits on top of that fingerprint and maps it to
   SoundSight's 5 categories (أذان / جرس الباب / حريق / نداء / إسعاف).
   - **Out of the box**, this uses a simple lookup from YAMNet's
     general classes (e.g. "Doorbell", "Siren") — good enough to demo
     immediately, not perfectly tuned.
   - **Once you record your own samples**, `train_classifier.py`
     trains a proper classifier on YAMNet's embeddings using your
     data. This is transfer learning — you need dozens of samples per
     category, not thousands.

## 1. Install

Requires **Python 3.9–3.11** (TensorFlow doesn't yet support 3.12+ everywhere).

```bash
cd soundsight_app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Microphone library note:** `sounddevice` needs the system audio
library PortAudio.
- macOS: `brew install portaudio` (then `pip install sounddevice` again if it failed)
- Linux: `sudo apt install libportaudio2`
- Windows: usually works with no extra steps
- Apple Silicon (M1/M2/M3): if plain `tensorflow` fails to install, use
  `pip install tensorflow-macos tensorflow-metal` instead.

## 2. Run it

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Grant the terminal/app
mic permission if your OS asks. Try:
- Playing a doorbell sound or clip from your phone near the mic
- Playing an adhan or siren clip from YouTube
- Clapping / knocking on your desk

The dashboard will light up in that sound's color, show the Arabic
label, a confidence %, and log it. The first run will download YAMNet
(~15MB) — after that it's cached and works offline.

## 3. Train it on your own sounds (recommended before any demo)

The generic mode is a rough approximation — it doesn't know "adhan" as
a specific class, and it will over-trigger on speech. Training your
own classifier fixes this and takes ~20 minutes:

```bash
python record_samples.py
```

Run it once per category (`adhan`, `doorbell`, `fire_alarm`, `call`,
`ambulance`, and — important — `other` for background noise/silence/TV/
talking, so the model learns what to ignore). 20–40 clips per category
is a good starting point.

Then:

```bash
python train_classifier.py
```

This prints a validation accuracy and saves `models/classifier.joblib`.
Restart `app.py` — it auto-detects and loads your trained model, and
the dashboard will show `MODE: CUSTOM TRAINED MODEL`.

## 4. Project structure

```
soundsight_app/
  app.py                 → Flask server, serves the live dashboard
  audio_engine.py         → mic capture + YAMNet + classification logic
  train_classifier.py     → trains your custom classifier from data/
  record_samples.py       → records training clips from your mic
  static/index.html       → live dashboard UI (same visual language as the pitch deck)
  data/                    → your recorded training clips (you create this)
  models/                  → saved custom classifier (created after training)
```

## 5. From this prototype to the real wearable

This app proves the AI approach works — it's the software validation
step before hardware. The path from here to the wristband, matching
the two-team split from the project plan:

- **Team B (AI & DevOps):** convert `classifier.joblib` + YAMNet into a
  single quantized **TensorFlow Lite** model sized for a low-power
  microcontroller, package it with a versioned build (Docker + CI/CD),
  and re-train/redeploy as more samples come in.
- **Team A (Software Engineering):** replace the Flask/browser
  dashboard with the on-device firmware loop — same input (audio
  window) and output (category + confidence) contract, but driving the
  haptic motor and LED ring directly instead of a webpage.

Because the interface between "AI result" and "what the device does
with it" is already a simple `{category, confidence}` object here, that
swap doesn't require re-architecting anything — only the last step
(webpage vs. motor/LED) changes.

## Known limitations (be upfront about these in a demo)

- Laptop mic ≠ final embedded hardware — expect different sensitivity.
- "Adhan" isn't a built-in AudioSet class, so generic mode is weak
  there specifically; training with a handful of adhan clips is what
  makes it reliable.
- This demo has no wake-word/noise-gating tuning yet — in a loud room,
  expect more false positives until you train on real environment
  noise as "other".
