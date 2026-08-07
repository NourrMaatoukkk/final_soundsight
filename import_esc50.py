"""
import_esc50.py — Automatically import audio samples from ESC-50 dataset into SoundSight's data/ directory.
"""

import os
import shutil
import csv

# Source path of unzipped ESC-50
ESC50_DIR = os.path.expanduser("~/Downloads/ESC-50-master")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ESC50_MAP = {
    # ESC-50 category name -> SoundSight category
    "door_wood_knock": "doorbell",
    "doorbell": "doorbell",
    "siren": "ambulance",
    "clock_alarm": "fire_alarm",
    "crying_baby": "call",
    "laughing": "call",
    "coughing": "call",
    "sneezing": "call",
    "clapping": "call",
    # Environmental / Background sounds for 'other'
    "rain": "other",
    "sea_waves": "other",
    "crackling_fire": "other",
    "vacuum_cleaner": "other",
    "washing_machine": "other",
    "keyboard_typing": "other",
    "footsteps": "other",
    "snoring": "other",
    "wind": "other",
    "engine": "other",
    "car_horn": "other",
    "insects": "other",
    "chirping_birds": "other",
    "drinking_sipping": "other",
    "mouse_click": "other",
}

def main():
    csv_path = os.path.join(ESC50_DIR, "meta", "esc50.csv")
    audio_dir = os.path.join(ESC50_DIR, "audio")

    if not os.path.exists(csv_path) or not os.path.exists(audio_dir):
        print(f"Error: ESC-50 not found at '{ESC50_DIR}'. Please check the folder path.")
        return

    counts = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            esc_cat = row["category"]
            filename = row["filename"]

            if esc_cat in ESC50_MAP:
                target_cat = ESC50_MAP[esc_cat]
                dest_dir = os.path.join(DATA_DIR, target_cat)
                os.makedirs(dest_dir, exist_ok=True)

                src_file = os.path.join(audio_dir, filename)
                dest_file = os.path.join(dest_dir, f"esc50_{esc_cat}_{filename}")

                if os.path.exists(src_file):
                    shutil.copy2(src_file, dest_file)
                    counts[target_cat] = counts.get(target_cat, 0) + 1

    print("\n✅ Successfully imported ESC-50 samples into SoundSight data/ directory:")
    for cat, count in sorted(counts.items()):
        print(f"  • {cat}: {count} clips")

    # Ensure adhan folder exists
    os.makedirs(os.path.join(DATA_DIR, "adhan"), exist_ok=True)
    adhan_count = len([f for f in os.listdir(os.path.join(DATA_DIR, "adhan")) if f.endswith(".wav") or f.endswith(".mp3")])
    print(f"  • adhan: {adhan_count} clips (Add Adhan clips here to complete the dataset!)")

if __name__ == "__main__":
    main()
