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
import traceback
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SERVER_CONFIG as CONFIG

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def cleanup():
    """Cleanup function"""
    logger.info("Cleaning up resources...")
    gc.collect()

app = Flask(__name__)
sock = Sock(app)

logger.info("Initializing Whisper model...")
model = WhisperModel(
    CONFIG["MODEL_SIZE"], device=CONFIG["DEVICE"], compute_type=CONFIG["COMPUTE_TYPE"]
)

atexit.register(cleanup)

# Disable Flask development server warnings but keep other logging
cli = sys.modules["flask.cli"]
cli.show_server_banner = lambda *x: None
logging.getLogger("werkzeug").setLevel(logging.WARNING)


class StreamingTranscriber:
    def __init__(self, model):
        self.model = model
        self.buffer = queue.Queue()
        self.is_processing = False
        self.current_audio = []
        self.min_audio_length = 1.0
        self.rate = CONFIG["RATE"]
        self.channels = CONFIG["CHANNELS"]
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ws = None  # Store websocket reference
        self.total_audio_duration = 0  # Track total audio duration
        self.segment_count = 0  # Track number of segments processed
        self.expected_segments = 1  # Initial estimate, will be updated
        logger.info(f"New streaming session initialized: {self.session_id}")

    def add_audio(self, audio_data):
        """Add audio data to buffer"""
        try:
            # Log incoming data size
            logger.debug(f"Session {self.session_id}: Received audio chunk of size {len(audio_data)} bytes")
            
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
                logger.debug(f"Session {self.session_id}: Converted {len(audio_data)} samples to numpy array")
            
            self.current_audio.extend(audio_data.tolist())
            audio_length = len(self.current_audio) / self.rate
            logger.debug(f"Session {self.session_id}: Current buffer length: {audio_length:.2f}s")
            
            # Update total audio duration and estimate expected segments
            self.total_audio_duration += len(audio_data) / self.rate
            self.expected_segments = max(1, int(self.total_audio_duration / 30) + 1)
            
            if audio_length >= self.min_audio_length and not self.is_processing:
                self.is_processing = True
                logger.info(f"Session {self.session_id}: Processing audio chunk of {audio_length:.2f}s")
                return self._process_audio()
            return None
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error adding audio data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _process_audio(self):
        """Process accumulated audio"""
        try:
            temp_file = f"{CONFIG['TEMP_FILE']}.{self.session_id}"
            logger.debug(f"Session {self.session_id}: Writing {len(self.current_audio)} samples to {temp_file}")
            
            # Convert audio data to temporary file
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.rate)
                audio_bytes = np.array(self.current_audio, dtype=np.int16).tobytes()
                wf.writeframes(audio_bytes)

            logger.info(f"Session {self.session_id}: Starting transcription of {len(audio_bytes)/2/self.rate:.2f}s audio")
            
            # Transcribe
            segments, info = self.model.transcribe(
                temp_file,
                beam_size=CONFIG["BEAM_SIZE"],
                without_timestamps=True
            )

            # Get text and clear buffer
            text = " ".join(segment.text for segment in segments)
            logger.info(f"Session {self.session_id}: Transcribed text: {text}")
            
            # Increment segment count
            self.segment_count += 1
            
            # Send segment progress notification
            if self.ws:
                self.ws.send(json.dumps({
                    "segment_progress": {
                        "current": self.segment_count,
                        "total": self.expected_segments
                    }
                }))
                logger.debug(f"Session {self.session_id}: Sent segment progress notification {self.segment_count}/{self.expected_segments}")
            
            self.current_audio = []
            return text

        except Exception as e:
            logger.error(f"Session {self.session_id}: Error in streaming transcription: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            self.is_processing = False
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Session {self.session_id}: Cleaned up temporary file")


@sock.route('/stream')
def stream(ws):
    """Handle streaming transcription requests"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"New WebSocket connection established: {session_id}")
    
    try:
        # Verify API key
        api_key = ws.receive()
        logger.debug(f"Session {session_id}: Received API key")
        
        if api_key != CONFIG["API_KEY"]:
            logger.warning(f"Session {session_id}: Unauthorized access attempt")
            ws.send(json.dumps({"error": "Unauthorized"}))
            return

        logger.info(f"Session {session_id}: Starting streaming transcription")
        transcriber = StreamingTranscriber(model)
        transcriber.ws = ws  # Pass websocket to transcriber for segment notifications
        
        while True:
            try:
                data = ws.receive()
                if data == "END_STREAM":
                    logger.info(f"Session {session_id}: Received end stream signal")
                    break
                
                # Process audio
                result = transcriber.add_audio(data)
                if result:
                    logger.debug(f"Session {session_id}: Sending transcription: {result}")
                    ws.send(json.dumps({"text": result}))
                    
            except Exception as e:
                logger.error(f"Session {session_id}: Error processing audio chunk: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                break
                
    except Exception as e:
        logger.error(f"Session {session_id}: Streaming error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info(f"Session {session_id}: Streaming session ended")


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
            
            # Calculate expected segments based on audio duration
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / rate
            expected_segments = max(1, int(duration / 30) + 1)
            
            print(f"Audio duration: {duration:.2f}s, expecting {expected_segments} segments")

        # Create a subclass to monitor segment processing
        class SegmentMonitor:
            def __init__(self):
                self.segment_count = 0
                self.last_start_time = 0
                
            def check_segment(self, segment_start):
                # If this is a new 30-second segment
                if segment_start >= self.last_start_time + 30:
                    self.segment_count += 1
                    self.last_start_time = segment_start
                    print(f"Processed segment {self.segment_count}/{expected_segments-1}")
                    return True
                return False
        
        monitor = SegmentMonitor()
        
        # Enable logging to capture segment processing
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
        
        # Original handler to capture log messages
        original_debug = logging.getLogger("faster_whisper").debug
        
        # Override debug method to detect segment processing
        def custom_debug(msg, *args, **kwargs):
            original_debug(msg, *args, **kwargs)
            
            # Check if this is a segment processing message
            if "Processing segment at" in msg:
                # Extract the timestamp
                try:
                    timestamp = float(msg.split("Processing segment at ")[1].split(":")[0]) * 60 + \
                               float(msg.split("Processing segment at ")[1].split(":")[1])
                    
                    # If this is a new segment, add to metadata
                    if monitor.check_segment(timestamp):
                        pass  # We'll use the print statement for monitoring
                except:
                    pass
        
        # Replace the debug method
        logging.getLogger("faster_whisper").debug = custom_debug

        segments, info = model.transcribe(
            temp_path, beam_size=CONFIG["BEAM_SIZE"], without_timestamps=True, log_progress=True
        )

        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        # Force evaluation of generator to catch any issues
        text = " ".join(segment.text for segment in segments)
        duration = time.time() - start_time
        print(f"Completed transcription in {duration:.1f} seconds\n")

        # Add metadata about the transcription
        result = {
            "text": text.strip(),
            "metadata": {
                "duration": duration,
                "expected_updates": expected_segments
            }
        }

        return json.dumps(result)

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return str(e), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Restore original debug method
        if 'original_debug' in locals():
            logging.getLogger("faster_whisper").debug = original_debug


@app.route("/transcribe_with_progress", methods=["POST"])
def transcribe_with_progress():
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

    def generate():
        try:
            # Basic WAV validation
            with wave.open(temp_path, "rb") as wav:
                if wav.getnchannels() > 2:
                    yield json.dumps({"error": "Invalid audio format"})
                    return
                
                # Calculate expected segments based on audio duration
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / rate
                expected_segments = max(1, int(duration / 30) + 1)
                
                print(f"Audio duration: {duration:.2f}s, expecting {expected_segments} segments")

            # Create a subclass to monitor segment processing
            class SegmentMonitor:
                def __init__(self):
                    self.segment_count = 0
                    self.last_start_time = 0
                    
                def check_segment(self, segment_start):
                    # If this is a new 30-second segment
                    if segment_start >= self.last_start_time + 30:
                        self.segment_count += 1
                        self.last_start_time = segment_start
                        print(f"Processed segment {self.segment_count}/{expected_segments-1}")
                        
                        # Send progress update
                        yield json.dumps({
                            "segment_progress": {
                                "current": self.segment_count,
                                "total": expected_segments
                            }
                        }) + "\n"
                        return True
                    return False
            
            monitor = SegmentMonitor()
            
            # Enable logging to capture segment processing
            logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
            
            # Original handler to capture log messages
            original_debug = logging.getLogger("faster_whisper").debug
            
            # Override debug method to detect segment processing
            def custom_debug(msg, *args, **kwargs):
                original_debug(msg, *args, **kwargs)
                
                # Check if this is a segment processing message
                if "Processing segment at" in msg:
                    # Extract the timestamp
                    try:
                        timestamp = float(msg.split("Processing segment at ")[1].split(":")[0]) * 60 + \
                                  float(msg.split("Processing segment at ")[1].split(":")[1])
                        
                        # If this is a new segment, yield progress update
                        monitor.check_segment(timestamp)
                    except:
                        pass
            
            # Replace the debug method
            logging.getLogger("faster_whisper").debug = custom_debug

            segments, info = model.transcribe(
                temp_path, beam_size=CONFIG["BEAM_SIZE"], without_timestamps=True, log_progress=True
            )

            print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

            # Force evaluation of generator to catch any issues
            text = " ".join(segment.text for segment in segments)
            duration = time.time() - start_time
            print(f"Completed transcription in {duration:.1f} seconds\n")

            yield json.dumps({"text": text.strip()}) + "\n"

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Restore original debug method
            if 'original_debug' in locals():
                logging.getLogger("faster_whisper").debug = original_debug

    return app.response_class(generate(), mimetype='text/event-stream')


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
