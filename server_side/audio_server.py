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
        self.min_audio_length = 1.0  # Reduced to 1.0 second for faster processing
        self.rate = CONFIG["RATE"]
        self.channels = CONFIG["CHANNELS"]
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_audio_received = 0  # Track total audio received
        self.chunks_received = 0  # Track number of chunks received
        self.last_process_time = time.time()
        self.accumulated_audio = []  # Store all audio for final processing
        self.processing_thread = None
        logger.info(f"New streaming session initialized: {self.session_id}")

    def add_audio(self, audio_data):
        """Add audio data to buffer and process in background if needed"""
        try:
            # Update counters
            self.chunks_received += 1
            self.total_audio_received += len(audio_data)
            
            # Log incoming data size (only every 50 chunks to reduce log spam)
            if self.chunks_received % 50 == 0:
                logger.debug(f"Session {self.session_id}: Received {self.chunks_received} chunks, total {self.total_audio_received} bytes")
            
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
            
            self.current_audio.extend(audio_data.tolist())
            self.accumulated_audio.extend(audio_data.tolist())  # Store for final processing
            
            audio_length = len(self.current_audio) / self.rate
            
            # Process audio if we have enough and not currently processing
            current_time = time.time()
            if (audio_length >= self.min_audio_length and not self.is_processing and 
                current_time - self.last_process_time >= 2.0):  # Process every 2 seconds
                
                # Start processing in a background thread
                self.is_processing = True
                self.last_process_time = current_time
                
                # Create a copy of current audio for processing
                audio_to_process = self.current_audio.copy()
                audio_length = len(audio_to_process) / self.rate
                logger.info(f"Session {self.session_id}: Starting background processing of {audio_length:.2f}s audio")
                
                # Clear current audio buffer for next chunk
                self.current_audio = []
                
                # Process in background thread
                self.processing_thread = threading.Thread(
                    target=self._process_audio_background,
                    args=(audio_to_process,)
                )
                self.processing_thread.daemon = True
                self.processing_thread.start()
            
            return None
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error adding audio data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _process_audio_background(self, audio_data):
        """Process audio in background thread and send result to client"""
        try:
            temp_file = f"{CONFIG['TEMP_FILE']}.{self.session_id}.{time.time()}"
            
            # Convert audio data to temporary file
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.rate)
                audio_bytes = np.array(audio_data, dtype=np.int16).tobytes()
                wf.writeframes(audio_bytes)
            
            audio_length = len(audio_data) / self.rate
            logger.info(f"Session {self.session_id}: Processing {audio_length:.2f}s audio in background")
            
            # Transcribe audio
            segments, info = self.model.transcribe(
                temp_file, 
                beam_size=CONFIG["BEAM_SIZE"],
                without_timestamps=True,
                language="en"  # Force English for better results
            )
            
            # Get transcription text
            text = " ".join(segment.text for segment in segments)
            logger.info(f"Session {self.session_id}: Background transcription result: '{text}'")
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Session {self.session_id}: Cleaned up temporary file")
            
            # Store result in queue for sending to client
            if text.strip():
                self.buffer.put(text)
            
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error in background processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.is_processing = False

    def get_result(self):
        """Get any available transcription result"""
        if not self.buffer.empty():
            return self.buffer.get()
        return None

    def process_final_audio(self):
        """Process all accumulated audio for final transcription"""
        try:
            # Wait for any background processing to complete
            if self.processing_thread and self.processing_thread.is_alive():
                logger.info(f"Session {self.session_id}: Waiting for background processing to complete")
                self.processing_thread.join(timeout=5.0)
            
            # Process any remaining audio in current_audio
            if self.current_audio and len(self.current_audio) > 0:
                self.accumulated_audio.extend(self.current_audio)
            
            if not self.accumulated_audio:
                logger.warning(f"Session {self.session_id}: No accumulated audio to process")
                return None
                
            temp_file = f"{CONFIG['TEMP_FILE']}.{self.session_id}.final"
            
            # Convert all accumulated audio to temporary file
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.rate)
                audio_bytes = np.array(self.accumulated_audio, dtype=np.int16).tobytes()
                wf.writeframes(audio_bytes)
            
            audio_length = len(self.accumulated_audio) / self.rate
            logger.info(f"Session {self.session_id}: Processing final audio of {audio_length:.2f}s")
            
            # Transcribe audio
            segments, info = self.model.transcribe(
                temp_file, 
                beam_size=CONFIG["BEAM_SIZE"],
                without_timestamps=True,
                language="en"  # Force English for better results
            )
            
            # Get transcription text
            text = " ".join(segment.text for segment in segments)
            logger.info(f"Session {self.session_id}: Final transcription: '{text}'")
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Session {self.session_id}: Cleaned up final temporary file")
            
            return text
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error processing final audio: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None


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
        
        # Start a background thread to check for results and send them to client
        results_thread = threading.Thread(
            target=_check_and_send_results,
            args=(ws, transcriber, session_id)
        )
        results_thread.daemon = True
        results_thread.start()
        
        while True:
            try:
                data = ws.receive()
                
                # Check for end signal
                if data == "END_STREAM":
                    logger.info(f"Session {session_id}: Received end stream signal")
                    
                    # Process all accumulated audio for final transcription
                    result = transcriber.process_final_audio()
                    if result:
                        logger.info(f"Session {session_id}: Sending final transcription: '{result}'")
                        ws.send(json.dumps({"text": result, "final": True}))
                    else:
                        logger.warning(f"Session {session_id}: Final transcription returned no result")
                        
                    # Log summary
                    logger.info(f"Session {session_id}: Streaming session summary - Received {transcriber.chunks_received} chunks, {transcriber.total_audio_received} bytes total")
                    break
                    
                # Add audio to transcriber
                transcriber.add_audio(data)
                    
            except Exception as e:
                logger.error(f"Session {session_id}: Error processing audio chunk: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                break
                
    except Exception as e:
        logger.error(f"Session {session_id}: Streaming error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logger.info(f"Session {session_id}: Streaming session ended")

def _check_and_send_results(ws, transcriber, session_id):
    """Background thread to check for results and send them to client"""
    try:
        while True:
            result = transcriber.get_result()
            if result:
                logger.info(f"Session {session_id}: Sending transcription: '{result}'")
                try:
                    ws.send(json.dumps({"text": result}))
                except Exception as e:
                    logger.error(f"Session {session_id}: Error sending result: {str(e)}")
                    break
            time.sleep(0.1)  # Check for new results every 100ms
    except Exception as e:
        logger.error(f"Session {session_id}: Error in results thread: {str(e)}")


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
