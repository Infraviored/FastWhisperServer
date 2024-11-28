# Server Configuration
SERVER_CONFIG = {
    "MODEL_SIZE": "base",
    "DEVICE": "cuda",
    "COMPUTE_TYPE": "float32",
    "PORT": 2024,
    "HOST": "0.0.0.0",
    "TEMP_FILE": "temp_audio.wav",
    "BEAM_SIZE": 5,
    "MAX_FILE_SIZE": 30 * 1024 * 1024,  # 30MB max
    "API_KEY": "your_api_key_here",
}

# Client Configuration
CLIENT_CONFIG = {
    "SERVER_URL": "http://127.0.0.1:2024",
    "AUDIO_FORMAT": 8,  # pyaudio.paInt16
    "CHANNELS": 1,
    "RATE": 44100,
    "CHUNK": 1024,
    "TEMP_FILE": "temp_recording.wav",
    "PID_FILE": "/tmp/whisper_recorder.pid",
    "API_KEY": "your_api_key_here",
    "SOUNDS": {
        "start": (523.25, 0.15),
        "stop": (659.25, 0.15),
        "complete": (783.99, 0.15),
        "error": (311.13, 0.15),
        "empty": (392.00, 0.15),
    },
}
