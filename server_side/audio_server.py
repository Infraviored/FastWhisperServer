from flask import Flask, request
from faster_whisper import WhisperModel
import os
import logging
import sys
import gc
import wave
import time
import atexit

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SERVER_CONFIG as CONFIG


def cleanup():
    """Cleanup function"""
    gc.collect()


app = Flask(__name__)
model = WhisperModel(
    CONFIG["MODEL_SIZE"], device=CONFIG["DEVICE"], compute_type=CONFIG["COMPUTE_TYPE"]
)

atexit.register(cleanup)

# Disable Flask development server warnings
cli = sys.modules["flask.cli"]
cli.show_server_banner = lambda *x: None

# Disable all logging except errors
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    start_time = time.time()
    print(f"\nStarting new transcription at {time.strftime('%H:%M:%S')}")

    # Check API key
    if request.headers.get("X-API-Key") != CONFIG["API_KEY"]:
        return "Unauthorized", 401

    # Check file presence and size
    if "file" not in request.files:
        return "No file", 400
    if request.content_length > CONFIG["MAX_FILE_SIZE"]:
        return "File too large", 413

    file = request.files["file"]
    temp_path = CONFIG["TEMP_FILE"]
    file.save(temp_path)

    try:
        # Basic WAV validation
        with wave.open(temp_path, "rb") as wav:
            if wav.getnchannels() > 2:
                return "Invalid audio format", 400

        print("Starting model.transcribe...")
        print(
            f"Model config: device={CONFIG['DEVICE']}, compute_type={CONFIG['COMPUTE_TYPE']}"
        )

        # Enable debug logging for the model
        logging.basicConfig()
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

        segments, info = model.transcribe(
            temp_path, beam_size=CONFIG["BEAM_SIZE"], without_timestamps=True
        )

        print(
            f"Transcription complete. Language: {info.language} (probability: {info.language_probability})"
        )

        # Force evaluation of generator to catch any issues
        segments = list(segments)
        print(f"Number of segments: {len(segments)}")

        text = " ".join(seg.text for seg in segments)
        duration = time.time() - start_time
        print(f"Completed transcription in {duration:.1f} seconds")

        return text.strip()

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback

        print(traceback.format_exc())
        return str(e), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("\nStarting Whisper transcription server...")
    print("Configuration:")
    print(f"- Model: {CONFIG['MODEL_SIZE']}")
    print(f"- Device: {CONFIG['DEVICE']}")
    print(f"- Compute Type: {CONFIG['COMPUTE_TYPE']}")
    print(f"- Max File Size: {CONFIG['MAX_FILE_SIZE'] / 1024 / 1024:.1f} MB")
    print(f"- Host: {CONFIG['HOST']}")
    print(f"- Port: {CONFIG['PORT']}")

    try:
        app.run(host=CONFIG["HOST"], port=CONFIG["PORT"], debug=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
