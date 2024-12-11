from flask import Flask, request
from flask_sock import Sock
from faster_whisper import WhisperModel
import os
import logging
import sys
import gc
import wave
import time
import atexit
import numpy as np
import json
import queue
import threading
import io

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SERVER_CONFIG as CONFIG


def cleanup():
    """Cleanup function"""
    gc.collect()


app = Flask(__name__)
sock = Sock(app)
model = WhisperModel(
    CONFIG["MODEL_SIZE"], device=CONFIG["DEVICE"], compute_type=CONFIG["COMPUTE_TYPE"]
)

atexit.register(cleanup)

# Disable Flask development server warnings
cli = sys.modules["flask.cli"]
cli.show_server_banner = lambda *x: None

# Disable all logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)


class StreamingTranscriber:
    def __init__(self, model):
        self.model = model
        self.buffer = queue.Queue()
        self.is_processing = False
        self.current_audio = []
        self.min_audio_length = 1.0  # Process at least 1 second of audio

    def add_audio(self, audio_data):
        """Add audio data to buffer"""
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
            
            self.current_audio.extend(audio_data.tolist())
            audio_length = len(self.current_audio) / CONFIG["RATE"]
            
            if audio_length >= self.min_audio_length and not self.is_processing:
                self.is_processing = True
                return self._process_audio()
            return None
        except Exception as e:
            print(f"Error adding audio data: {e}")
            return None

    def _process_audio(self):
        """Process accumulated audio"""
        try:
            # Convert audio data to temporary file
            with wave.open(CONFIG["TEMP_FILE"], 'wb') as wf:
                wf.setnchannels(CONFIG["CHANNELS"])
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(CONFIG["RATE"])
                audio_bytes = np.array(self.current_audio, dtype=np.int16).tobytes()
                wf.writeframes(audio_bytes)

            # Transcribe
            segments, info = self.model.transcribe(
                CONFIG["TEMP_FILE"],
                beam_size=CONFIG["BEAM_SIZE"],
                without_timestamps=True
            )

            # Get text and clear buffer
            text = " ".join(segment.text for segment in segments)
            self.current_audio = []
            return text

        except Exception as e:
            print(f"Error in streaming transcription: {e}")
            return None
        finally:
            self.is_processing = False
            if os.path.exists(CONFIG["TEMP_FILE"]):
                os.remove(CONFIG["TEMP_FILE"])


@sock.route('/stream')
def stream(ws):
    """Handle streaming transcription requests"""
    try:
        # Verify API key
        api_key = ws.receive()  # First message should be API key
        if api_key != CONFIG["API_KEY"]:
            ws.send(json.dumps({"error": "Unauthorized"}))
            return

        print("\nStarting streaming transcription session...")
        transcriber = StreamingTranscriber(model)
        
        while True:
            try:
                data = ws.receive()  # Receive audio data
                if data == "END_STREAM":
                    break
                
                # Process audio
                result = transcriber.add_audio(data)
                if result:
                    print(f"Transcribed: {result}")
                    ws.send(json.dumps({"text": result}))
                    
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
                break
                
    except Exception as e:
        print(f"Streaming error: {e}")
    finally:
        print("Streaming session ended")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    start_time = time.time()
    print(f"\nStarting transcription at {time.strftime('%H:%M:%S')}")

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

        segments, info = model.transcribe(
            temp_path, beam_size=CONFIG["BEAM_SIZE"], without_timestamps=True
        )

        print(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        # Force evaluation of generator to catch any issues
        text = " ".join(segment.text for segment in segments)
        duration = time.time() - start_time
        print(f"Completed transcription in {duration:.1f} seconds\n")

        return text.strip()

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
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
