"""
app.py — SoundSight local demo server.

Runs the real audio-classification engine (audio_engine.py) against
your computer's microphone and serves a live dashboard, styled the
same way as the product presentation, at http://127.0.0.1:5000

This is the real AI pipeline that would run on the wearable's chip —
here it's just running on your laptop/desktop so you can see and test
it working today, before any hardware exists.
"""

from flask import Flask, jsonify, send_from_directory
from audio_engine import SoundEngine

app = Flask(__name__, static_folder="static", static_url_path="")

print("Starting SoundSight engine...")
engine = SoundEngine()
engine.start()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/status")
def status():
    return jsonify({
        "current": engine.latest,
        "log": list(engine.log),
        "mode": engine.mode,
    })


if __name__ == "__main__":
    print("\nSoundSight is listening. Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
