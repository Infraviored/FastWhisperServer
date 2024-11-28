import sys
import os

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
    global recorder  # Add this line
    parser = argparse.ArgumentParser(
        description="Audio recording and transcription client"
    )
    parser.add_argument("--start", action="store_true", help="Start recording")
    parser.add_argument("--stop", action="store_true", help="Stop recording")
    parser.add_argument(
        "--test-sounds", action="store_true", help="Test all notification sounds"
    )
    args = parser.parse_args()

    if args.test_sounds:
        test_sounds()
        return

    if args.stop:
        stop_recording()
        return

    if args.start:
        # Check if already running
        if read_pid():
            print("Recording already in progress")
            return
        recorder = AudioRecorder()  # Now assigns to global variable
        recorder.record(auto_stop=True)
    else:
        recorder = AudioRecorder()  # Now assigns to global variable
        recorder.record()


if __name__ == "__main__":
    # Suppress ALSA error messages
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)

    main()

    # Restore stderr
    os.dup2(old_stderr, 2)
