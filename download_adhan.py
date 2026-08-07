import os
import ssl
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adhan")
os.makedirs(OUT_DIR, exist_ok=True)

ssl_context = ssl._create_unverified_context()

BASE_URL = "https://archive.org/download/adhan.notifications/"
FILES = [
    "Ahmed_al_Imadi_Adhan.mp3",
    "Majed_al_Hamathani_Adhan.mp3",
    "Mishary_Rashid_al_Afasy_Fajr_Adhan.mp3",
    "Mokhtar_Hadj_Slimane_Adhan.mp3",
    "Nasser_al_Qatami_Adhan.mp3"
]

def main():
    print("Downloading Adhan MP3 files directly into data/adhan/...")
    headers = {"User-Agent": "Mozilla/5.0"}

    count = 0
    for idx, fname in enumerate(FILES):
        url = BASE_URL + fname
        dest_file = os.path.join(OUT_DIR, fname)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_context) as response, open(dest_file, "wb") as f:
                f.write(response.read())

            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1000:
                print(f"  ✅ Saved {fname} ({round(os.path.getsize(dest_file)/1024, 1)} KB)")
                count += 1
        except Exception as e:
            print(f"  ❌ Error downloading {fname}: {e}")

    print(f"\n✅ Total Adhan MP3 files in data/adhan/: {count}")

if __name__ == "__main__":
    main()
