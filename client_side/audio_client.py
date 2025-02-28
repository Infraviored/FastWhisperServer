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
                    
        # Ensure sounddevice is properly cleaned up
        try:
            sd.stop()
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
            url = f"{CONFIG['SERVER_URL']}/transcribe_with_progress"
            headers = {"X-API-Key": CONFIG["API_KEY"]}

            with open(CONFIG["TEMP_FILE"], "rb") as audio_file:
                # Use a streaming response to get progress updates
                with requests.post(
                    url, files={"file": audio_file}, headers=headers, stream=True
                ) as response:
                    
                    if response.status_code != 200:
                        print(f"Server error: {response.text}")
                        return None
                    
                    # Process streaming response for progress updates
                    final_text = ""
                    segment_count = 0
                    expected_segments = 1
                    
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        data = json.loads(line.decode('utf-8'))
                        
                        if "segment_progress" in data:
                            # Play a sound with pitch based on progress
                            progress = data["segment_progress"]
                            current_segment = progress["current"]
                            total_segments = progress["total"]
                            expected_segments = total_segments
                            
                            # Only play for segments before the final one
                            if current_segment < total_segments:
                                # Calculate pitch - start low and increase with each segment
                                base_freq = 220  # A3 note
                                freq = base_freq * (1 + 0.5 * (current_segment / total_segments))
                                
                                # Create and play a custom pitched sound
                                samples = generate_beep(freq, 0.15)
                                sd.play(samples, CONFIG["RATE"])
                                sd.wait()
                                
                                print(f"\nSegment {current_segment}/{total_segments} processed")
                        
                        elif "text" in data:
                            final_text = data["text"]
                    
                    if not final_text.strip():
                        play_sound("empty")
                        print("\nTranscription was empty!")
                        return None
                    
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=final_text.encode(),
                        check=True,
                    )
                    play_sound("complete")
                    print("\nTranscription copied to clipboard!")
                    print(f"Text: {final_text}")
                    return final_text

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
        ws.send(CONFIG["API_KEY"])
        self.connection_established = True

    def _on_ws_message(self, ws, message):
        """Handle incoming transcription"""
        try:
            data = json.loads(message)
            if "error" in data:
                print(f"\nServer error: {data['error']}")
                self.streaming = False
            elif "segment_progress" in data:
                # Play a sound with pitch based on progress
                progress = data["segment_progress"]
                current_segment = progress["current"]
                total_segments = progress["total"]
                
                # Only play for segments before the final one
                if current_segment < total_segments:
                    # Calculate pitch - start low and increase with each segment
                    base_freq = 220  # A3 note
                    freq = base_freq * (1 + 0.5 * (current_segment / total_segments))
                    
                    # Create and play a custom pitched sound
                    samples = generate_beep(freq, 0.15)
                    sd.play(samples, CONFIG["RATE"])
                    sd.wait()
                    
                    print(f"\nSegment {current_segment}/{total_segments} processed")
            elif "text" in data:
                self.accumulated_text.append(data["text"])
                print(f"\rPartial transcription: {data['text']}", end="", flush=True)
        except json.JSONDecodeError:
            print(f"\nError: Invalid JSON message from server: {message}")
        except Exception as e:
            print(f"\nError processing message: {e}")

    def _on_ws_error(self, ws, error):
        play_sound("error")
        print(f"\nWebSocket error: {error}")
        self.streaming = False

    def _on_ws_close(self, ws, close_status_code, close_msg):
        if self.streaming:  # Only show error if not intentionally closed
            play_sound("error")
            print(f"\nWebSocket connection closed: {close_status_code} - {close_msg}")
        self.streaming = False

    def record_stream(self):
        """Record and stream audio"""
        print("\nInitializing streaming mode...")
        
        if not check_server():
            print("Server check failed!")
            return None

        write_pid()
        self.streaming = True
        
        print("Attempting to establish WebSocket connection...")
        if not self._connect_websocket():
            print("Failed to establish WebSocket connection!")
            cleanup_pid()
            return None

        try:
            stream = self.p.open(
                format=CONFIG["AUDIO_FORMAT"],
                channels=CONFIG["CHANNELS"],
                rate=CONFIG["RATE"],
                input=True,
                frames_per_buffer=CONFIG["CHUNK"],
                stream_callback=self._stream_callback
            )
            self.stream = stream  # Store reference for cleanup

            play_sound("start")
            print("\nStreaming... Press SPACE to stop")

            listener = keyboard.Listener(on_press=self.on_press)
            listener.start()

            while self.streaming and self.recording:
                if not self.ws or not self.ws.sock or not self.ws.sock.connected:
                    play_sound("error")
                    print("\nLost connection to server")
                    break
                time.sleep(0.1)

            stream.stop_stream()
            stream.close()
            
            if self.ws:
                self.ws.close()

            # Combine all transcriptions
            final_text = " ".join(self.accumulated_text)
            if final_text.strip():
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=final_text.encode(),
                    check=True,
                )
                play_sound("complete")
                print(f"\nFinal transcription: {final_text}")
                return final_text
            else:
                play_sound("empty")
                print("\nNo transcription received")
                return None

        except Exception as e:
            print(f"\nSession {self.session_id}: Stream error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            # Ensure everything is cleaned up
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
            if self.p:
                try:
                    self.p.terminate()
                except:
                    pass
            cleanup_pid()
            print(f"Session {self.session_id}: Cleanup complete")

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """Handle audio stream data"""
        if status:
            print(f"\nStream error: {status}")
        
        if self.streaming and self.ws and hasattr(self.ws, 'sock') and self.ws.sock and self.ws.sock.connected:
            try:
                self.ws.send(in_data, ABNF.OPCODE_BINARY)
            except Exception as e:
                print(f"\nError sending audio data: {str(e)}")
                self.streaming = False
                return (None, pyaudio.paComplete)
        return (in_data, pyaudio.paContinue)


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
