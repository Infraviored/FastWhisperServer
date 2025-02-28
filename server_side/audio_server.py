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
        self.min_audio_length = CONFIG.get("STREAMING_MIN_AUDIO_LENGTH", 10.0)  # Use config value or default to 5.0
        self.process_frequency = CONFIG.get("STREAMING_PROCESS_FREQUENCY", 10.0)  # Use config value or default to 3.0
        self.rate = CONFIG["RATE"]
        self.channels = CONFIG["CHANNELS"]
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_audio_received = 0
        self.chunks_received = 0
        self.last_process_time = time.time()
        self.accumulated_transcription = ""  # Store the complete transcription
        self.processing_thread = None
        logger.info(f"New streaming session initialized: {self.session_id}")
        logger.info(f"Using min_audio_length={self.min_audio_length}s, process_frequency={self.process_frequency}s")

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
            
            audio_length = len(self.current_audio) / self.rate
            
            # Process audio if we have enough and not currently processing
            current_time = time.time()
            if (audio_length >= self.min_audio_length and not self.is_processing and 
                current_time - self.last_process_time >= self.process_frequency):
                
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
        """Process audio in background thread and put result in queue"""
        try:
            temp_file = f"{CONFIG['TEMP_FILE']}.{self.session_id}.{time.time()}"
            
            # Convert audio data to temporary file
            audio_length = len(audio_data) / self.rate
            logger.info(f"Session {self.session_id}: Processing {audio_length:.2f}s audio in background")
            
            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.rate)
                wf.writeframes(np.array(audio_data, dtype=np.int16).tobytes())
            
            # Transcribe audio
            logger.info(f"Session {self.session_id}: Starting transcription of {audio_length:.2f}s audio")
            
            segments, info = self.model.transcribe(
                temp_file, 
                beam_size=CONFIG["BEAM_SIZE"],
                without_timestamps=True
            )
            
            # Get transcription text
            text = " ".join(segment.text for segment in segments)
            logger.info(f"Session {self.session_id}: Background transcription result: '{text}'")
            
            # Add to results queue
            self.buffer.put(text)
            
            # Add to accumulated transcription
            if text.strip():
                self.accumulated_transcription += text
            
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Session {self.session_id}: Cleaned up temporary file")
            
            self.is_processing = False
            
        except Exception as e:
            logger.error(f"Session {self.session_id}: Error in background processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.is_processing = False

    def get_result(self):
        """Get transcription result from queue if available"""
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None

    def process_final_audio(self):
        """Process any remaining audio and return final accumulated result"""
        # Wait for any ongoing processing to complete
        logger.info(f"Session {self.session_id}: Waiting for background processing to complete")
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10)
        
        # Process any remaining audio in the buffer
        if self.current_audio and len(self.current_audio) > 0:
            audio_length = len(self.current_audio) / self.rate
            logger.info(f"Session {self.session_id}: Processing final chunk of {audio_length:.2f}s audio")
            
            temp_file = f"{CONFIG['TEMP_FILE']}.{self.session_id}.final"
            
            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.rate)
                wf.writeframes(np.array(self.current_audio, dtype=np.int16).tobytes())
            
            # Transcribe final audio chunk
            segments, info = self.model.transcribe(
                temp_file, 
                beam_size=CONFIG["BEAM_SIZE"],
                without_timestamps=True
            )
            
            # Get transcription text
            final_chunk_text = " ".join(segment.text for segment in segments)
            logger.info(f"Session {self.session_id}: Final chunk transcription: '{final_chunk_text}'")
            
            # Add final chunk to accumulated transcription
            if final_chunk_text.strip():
                self.accumulated_transcription += final_chunk_text
            
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"Session {self.session_id}: Cleaned up final temporary file")
        
        # Return the complete accumulated transcription
        return self.accumulated_transcription


@sock.route('/stream')
def stream_handler(ws):
    """WebSocket handler for streaming audio"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"New streaming connection from {request.remote_addr}, session {session_id}")
    
    try:
        # Authenticate
        auth_message = ws.receive()
        auth_data = json.loads(auth_message)
        
        if auth_data.get("api_key") != CONFIG["API_KEY"]:
            logger.warning(f"Session {session_id}: Authentication failed")
            ws.send(json.dumps({"error": "Authentication failed"}))
            return
        
        logger.info(f"Session {session_id}: Authentication successful")
        ws.send(json.dumps({"status": "authenticated"}))
        
        # Initialize transcriber
        transcriber = StreamingTranscriber(model)
        
        # Start a background thread to check for results and send them to client
        results_thread = threading.Thread(
            target=_check_and_send_results,
            args=(ws, transcriber, session_id)
        )
        results_thread.daemon = True
        results_thread.start()
        
        # Process incoming audio data
        while True:
            try:
                message = ws.receive()
                
                # Check if client wants to end the stream
                if message == "END_STREAM":
                    logger.info(f"Session {session_id}: Received end stream signal")
                    break
                
                # Process binary audio data
                if message:
                    data = message
                    
                    # Add audio to transcriber
                    transcriber.add_audio(data)
                    
            except Exception as e:
                logger.error(f"Session {session_id}: Error processing audio chunk: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                break
                
        # Process final audio and send complete transcription
        final_transcription = transcriber.process_final_audio()
        if final_transcription:
            logger.info(f"Session {session_id}: Sending final transcription: '{final_transcription}'")
            ws.send(json.dumps({"text": final_transcription, "final": True}))
        
        logger.info(f"Session {session_id}: Streaming session summary - Received {transcriber.chunks_received} chunks, {transcriber.total_audio_received} bytes total")
                
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
