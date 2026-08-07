"""
record_samples.py — record training clips for your own sound categories.

Run:
    python record_samples.py

Then follow the prompts. Recorded clips are saved into data/<category>/,
ready for train_classifier.py.

Tips for good recordings:
  - 2-3 second clips work well.
  - Record from a few different distances/angles from the mic.
  - For "other", record background noise, TV, talking, traffic, etc —
    anything the wristband should learn to IGNORE.
"""

import os
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CLIP_SECONDS = 2.5


def main():
    print("Suggested categories: adhan, doorbell, fire_alarm, call, ambulance, other\n")
    category = input("Category name: ").strip().lower().replace(" ", "_")
    count = int(input("How many clips to record? (suggest 20-40): ").strip())

    out_dir = os.path.join("data", category)
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])

    for i in range(count):
        input(f"\n[{i + 1}/{count}] Press Enter, then make/play the sound "
              f"for about {CLIP_SECONDS}s...")
        print("Recording...")
        audio = sd.rec(
            int(CLIP_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        )
        sd.wait()
        path = os.path.join(out_dir, f"clip_{existing + i + 1:03d}.wav")
        sf.write(path, audio, SAMPLE_RATE)
        print(f"Saved {path}")

    print(f"\nDone. Recorded {count} clips into data/{category}/")
    print("Repeat for each category, then run: python train_classifier.py")


if __name__ == "__main__":
    main()
