import sys
import os
import signal
import atexit

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CLIENT_CONFIG as CONFIG

import pyaudio
import wave
import requests
import subprocess
import os
import argparse
from pynput import keyboard
import numpy as np
import sounddevice as sd
import time
import signal
from websocket import WebSocketApp, ABNF
import json
import threading
import traceback

recorder = None  # Global variable to store recorder instance


def generate_beep(frequency, duration):
    """Generate a simple beep sound with smooth envelope"""
    rate = CONFIG["RATE"]
    t = np.linspace(0, duration, int(rate * duration), False)

    # Create the basic sine wave
    samples = np.sin(2 * np.pi * frequency * t)

    # Create smooth envelope
    attack_time = 0.02
    release_time = 0.15

    attack_len = int(attack_time * rate)
    release_len = int(release_time * rate)

    # Apply attack (fade in)
    samples[:attack_len] *= np.linspace(0, 1, attack_len)

    # Apply release (fade out)
    samples[-release_len:] *= np.linspace(1, 0, release_len) ** 2

    # Normalize
    samples = samples * 0.7

    return samples.astype(np.float32)


def play_sound(sound_type):
    """Play a notification beep"""
    try:
        frequency, duration = CONFIG["SOUNDS"][sound_type]
        samples = generate_beep(frequency, duration)

        if sound_type in ["error", "empty"]:
            for _ in range(2):
                sd.play(samples, CONFIG["RATE"])
                sd.wait()
                time.sleep(0.15)
        else:
            sd.play(samples, CONFIG["RATE"])
            sd.wait()
    except Exception as e:
        print(f"Error playing {sound_type} sound: {e}")
        # Don't let sound errors stop the program
        pass


def check_server():
    try:
        requests.get(CONFIG["SERVER_URL"], timeout=1)
        return True
    except:
        play_sound("error")
        print("Error: Cannot connect to transcription server")
        return False


def write_pid():
    """Write current process ID to file"""
    with open(CONFIG["PID_FILE"], "w") as f:
        f.write(str(os.getpid()))


def read_pid():
    """Read process ID from file"""
    try:
        with open(CONFIG["PID_FILE"], "r") as f:
            return int(f.read().strip())
    except:
        return None


def cleanup_pid():
    """Remove PID file"""
    if os.path.exists(CONFIG["PID_FILE"]):
        os.remove(CONFIG["PID_FILE"])


# Add new signal handler
def handle_stop_signal(signum, frame):
    """Signal handler for stop request"""
    print("\nStop signal received, finishing recording...")
    global recorder
    if recorder and hasattr(recorder, "recording"):
        recorder.recording = False


def cleanup_resources():
    """Cleanup function to ensure all resources are released"""
    print("Cleaning up resources...")
    try:
        # Stop any active recording
        if 'recorder' in globals() and recorder:
            recorder.recording = False
            recorder.streaming = False
            
            # Close WebSocket if it exists
            if hasattr(recorder, 'ws') and recorder.ws:
                try:
                    recorder.ws.close()
                except:
                    pass
            
            # Terminate PyAudio
            if hasattr(recorder, 'p'):
                try:
                    recorder.p.terminate()
                except:
                    pass
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    # Always remove PID file
    cleanup_pid()


# Register cleanup functions
atexit.register(cleanup_resources)
signal.signal(signal.SIGTERM, lambda signo, frame: cleanup_resources())
signal.signal(signal.SIGINT, lambda signo, frame: cleanup_resources())


class AudioRecorder:
    def __init__(self):
        self.recording = True
        self.frames = []
        self.p = pyaudio.PyAudio()
        signal.signal(signal.SIGUSR1, handle_stop_signal)

    def on_press(self, key):
        if key == keyboard.Key.space:
            print("\nStopping recording...")
            self.recording = False
            return False

    def record(self, auto_stop=False):
        """Record audio with optional auto-stop"""
        if not check_server():
            return None

        write_pid()

        try:
            stream = self.p.open(
                format=CONFIG["AUDIO_FORMAT"],
                channels=CONFIG["CHANNELS"],
                rate=CONFIG["RATE"],
                input=True,
                frames_per_buffer=CONFIG["CHUNK"],
            )

            play_sound("start")
            print("\nRecording..." + (" Press SPACE to stop" if not auto_stop else ""))

            if not auto_stop:
                listener = keyboard.Listener(on_press=self.on_press)
                listener.start()

            while self.recording:
                try:
                    data = stream.read(CONFIG["CHUNK"], exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Error during recording: {e}")
                    break

        except Exception as e:
            print(f"Error setting up audio stream: {e}")
            return
        finally:
            if "stream" in locals():
                stream.stop_stream()
                stream.close()
            self.p.terminate()
            cleanup_pid()

        if self.frames:
            play_sound("stop")
            return self._save_and_transcribe()
        return None

    def _save_and_transcribe(self):
        """Save recording and send to transcription server"""
        try:
            wf = wave.open(CONFIG["TEMP_FILE"], "wb")
            wf.setnchannels(CONFIG["CHANNELS"])
            wf.setsampwidth(self.p.get_sample_size(CONFIG["AUDIO_FORMAT"]))
            wf.setframerate(CONFIG["RATE"])
            wf.writeframes(b"".join(self.frames))
            wf.close()

            print("Sending to server for transcription...")
            url = f"{CONFIG['SERVER_URL']}/transcribe"
            headers = {"X-API-Key": CONFIG["API_KEY"]}

            with open(CONFIG["TEMP_FILE"], "rb") as audio_file:
                response = requests.post(
                    url, files={"file": audio_file}, headers=headers
                )

                if response.status_code == 200:
                    text = response.text
                    if not text.strip():
                        play_sound("empty")
                        print("\nTranscription was empty!")
                        return None
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=text.encode(),
                        check=True,
                    )
                    play_sound("complete")
                    print("\nTranscription copied to clipboard!")
                    print(f"Text: {text}")
                    return text
                else:
                    print(f"Server error: {response.text}")
                    return None

        except Exception as e:
            print(f"Error during save/transcribe: {e}")
            return None
        finally:
            if os.path.exists(CONFIG["TEMP_FILE"]):
                try:
                    os.remove(CONFIG["TEMP_FILE"])
                except Exception as e:
                    print(f"Error removing temporary file: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        if hasattr(self, 'p'):
            try:
                self.p.terminate()
            except:
                pass


class StreamingRecorder(AudioRecorder):
    def __init__(self):
        super().__init__()
        self.ws = None
        self.streaming = False
        self.accumulated_text = []
        self.connection_established = False
        self.connection_timeout = 5  # seconds
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.min_recording_time = 2.0  # Ensure at least 2 seconds of audio
        self.start_time = None
        self.audio_chunks_sent = 0  # Track how many chunks we've sent
        self.total_audio_bytes = 0  # Track total audio bytes sent
        self.stream = None
        self.stream_active = False
        self.final_transcription = None
        self.debug_mode = False  # Set to False to disable debug messages

    def _connect_websocket(self):
        """Establish WebSocket connection"""
        ws_url = CONFIG["WS_URL"] + "/stream"
        print(f"\nConnecting to WebSocket at {ws_url}...")
        
        try:
            self.ws = WebSocketApp(
                ws_url,
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection with timeout
            start_time = time.time()
            while not self.connection_established:
                if time.time() - start_time > self.connection_timeout:
                    play_sound("error")
                    print("\nError: Could not establish WebSocket connection")
                    return False
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            print(f"\nError creating WebSocket connection: {e}")
            return False

    def _on_ws_open(self, ws):
        """Send API key on connection"""
        print("WebSocket connected, authenticating...")
        # Send API key as JSON
        auth_message = json.dumps({"api_key": CONFIG["API_KEY"]})
        ws.send(auth_message)
        self.connection_established = True
        print("Connection established")

    def _on_ws_message(self, ws, message):
        """Handle incoming transcription"""
        try:
            if self.debug_mode:
                print(f"\nReceived message from server: {message}")
                
            data = json.loads(message)
            if "error" in data:
                print(f"\nServer error: {data['error']}")
                self.streaming = False
            elif "text" in data:
                # Check if this is the final transcription
                if data.get("final", False):
                    self.final_transcription = data["text"]
                    # Don't print here - we'll print once at the end
                else:
                    self.accumulated_text.append(data["text"])
                    print(f"\rPartial transcription: {data['text']}", end="", flush=True)
        except json.JSONDecodeError:
            print(f"\nError: Invalid message from server")
        except Exception as e:
            print(f"\nError processing message: {e}")

    def _on_ws_error(self, ws, error):
        play_sound("error")
        print(f"\nWebSocket error: {error}")
        self.streaming = False

    def _on_ws_close(self, ws, close_status_code, close_msg):
        if self.streaming:  # Only show error if not intentionally closed
            print(f"\nWebSocket connection closed")
        self.streaming = False

    def record_stream(self):
        """Record audio and stream to server for real-time transcription"""
        try:
            print("\nInitializing streaming mode...")
            
            # Check if server is available
            if not check_server():
                return None
                
            # Write PID for external control
            write_pid()
            
            # Register signal handler
            signal.signal(signal.SIGUSR1, handle_stop_signal)
            
            # Connect to WebSocket
            if not self._connect_websocket():
                return None
                
            # Initialize audio
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=CONFIG["AUDIO_FORMAT"],
                channels=CONFIG["CHANNELS"],
                rate=CONFIG["RATE"],
                input=True,
                frames_per_buffer=CONFIG["CHUNK"]
            )
            self.stream_active = True
            
            # Start recording
            self.frames = []
            self.recording = True
            self.streaming = True
            self.start_time = time.time()
            
            play_sound("start")
            print("\nStreaming... Press SPACE to stop\n")
            
            # Start keyboard listener
            keyboard_thread = threading.Thread(target=self._keyboard_listener)
            keyboard_thread.daemon = True
            keyboard_thread.start()
            
            # Main loop - keep running until stopped
            while self.streaming and self.recording:
                if not self.ws or not self.ws.sock or not self.ws.sock.connected:
                    play_sound("error")
                    print("\nLost connection to server")
                    break
                
                # Read audio data directly
                if self.stream_active:
                    try:
                        audio_data = self.stream.read(CONFIG["CHUNK"], exception_on_overflow=False)
                        if audio_data:
                            self.ws.send(audio_data, ABNF.OPCODE_BINARY)
                            self.frames.append(audio_data)  # Also store locally for backup
                            
                            # Update counters
                            self.audio_chunks_sent += 1
                            self.total_audio_bytes += len(audio_data)
                            
                            # Print debug info only in debug mode
                            if self.debug_mode and self.audio_chunks_sent % 50 == 0:
                                print(f"\nSent {self.audio_chunks_sent} audio chunks ({self.total_audio_bytes} bytes)")
                    except Exception as e:
                        print(f"\nError reading/sending audio: {e}")
                
                # Ensure minimum recording time
                if time.time() - self.start_time < self.min_recording_time:
                    continue
                
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
            # Clean up stream
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                except Exception as e:
                    print(f"\nError closing audio stream: {e}")
            
            # Close WebSocket connection gracefully
            if self.ws and hasattr(self.ws, 'sock') and self.ws.sock:
                try:
                    # Send end signal to server
                    print("\nSending end signal to server...")
                    self.ws.send("END_STREAM")
                    
                    # Wait for final transcription
                    print("Waiting for final transcription...")
                    wait_start = time.time()
                    while time.time() - wait_start < 10.0 and self.final_transcription is None:
                        time.sleep(0.1)  # Wait for final transcription with timeout
                    
                    self.ws.close()
                except Exception as e:
                    print(f"\nError closing WebSocket: {e}")

            # Use final transcription if available, otherwise use accumulated text
            final_text = self.final_transcription if self.final_transcription else " ".join(self.accumulated_text)
            
            if final_text.strip():
                try:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=final_text.encode(),
                        check=True,
                    )
                    play_sound("complete")
                    print(f"\nFinal transcription: {final_text}")
                    return final_text
                except Exception as e:
                    print(f"\nError copying to clipboard: {e}")
                    return final_text
            else:
                play_sound("empty")
                print("\nNo transcription received")
                return None

        except Exception as e:
            print(f"\nError in streaming session: {str(e)}")
            return None
        finally:
            # Ensure everything is cleaned up
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
            if hasattr(self, 'p') and self.p:
                try:
                    self.p.terminate()
                except:
                    pass
            cleanup_pid()
            print("Session ended")

    # Add the keyboard listener method
    def _keyboard_listener(self):
        """Listen for keyboard input to stop recording"""
        try:
            with keyboard.Listener(on_press=self._on_key_press) as listener:
                listener.join()
        except Exception as e:
            print(f"\nError in keyboard listener: {e}")
            self.recording = False
            
    def _on_key_press(self, key):
        """Handle key press events"""
        try:
            # Stop on space key
            if key == keyboard.Key.space:
                print("\nStopping recording...")
                self.recording = False
                return False  # Stop listener
        except Exception as e:
            print(f"\nError processing key press: {e}")
        return True  # Continue listening


def test_sounds():
    """Test all notification sounds"""
    print("Testing all sounds...")
    for sound_type in CONFIG["SOUNDS"]:
        print(f"Playing {sound_type} sound...")
        play_sound(sound_type)
        time.sleep(2)


def stop_recording():
    """Signal the recording process to stop"""
    pid = read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGUSR1)  # Send USR1 signal instead of TERM
            print("Stop signal sent to recording process")
            return True
        except ProcessLookupError:
            cleanup_pid()
            print("No recording process found")
        except Exception as e:
            print(f"Error signaling recording process: {e}")
    return False


def main():
    global recorder
    try:
        parser = argparse.ArgumentParser(description="Audio recording and transcription client")
        parser.add_argument("--start", action="store_true", help="Start recording")
        parser.add_argument("--stop", action="store_true", help="Stop recording")
        parser.add_argument("--test-sounds", action="store_true", help="Test all notification sounds")
        parser.add_argument("--stream", action="store_true", help="Use streaming mode")
        args = parser.parse_args()

        if args.test_sounds:
            test_sounds()
            return

        if args.stop:
            stop_recording()
            return

        if args.start:
            if read_pid():
                print("Recording already in progress")
                return
            recorder = AudioRecorder()
            recorder.record(auto_stop=True)
        elif args.stream:
            recorder = StreamingRecorder()
            recorder.record_stream()
        else:
            recorder = AudioRecorder()
            recorder.record()
    except Exception as e:
        print(f"Error in main: {e}")
        cleanup_resources()
    finally:
        cleanup_resources()


if __name__ == "__main__":
    try:
        # Suppress ALSA error messages
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)

        main()
    finally:
        # Restore stderr and ensure cleanup
        os.dup2(old_stderr, 2)
        cleanup_resources()
