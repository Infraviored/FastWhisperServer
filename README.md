# Whisper Cube

A client-server application for real-time audio transcription using OpenAI's Whisper model.

## Prerequisites

- Python 3.8+
- `screen` (for server management)
- CUDA-capable GPU (recommended)

## Installation

### Server Setup

1. Clone the repository on the server:
```bash
git clone https://github.com/yourusername/whisper_cube.git
cd whisper_cube
```

2. Create and activate virtual environment:
```bash
python -m venv whisper_env
source whisper_env/bin/activate  # On Linux/Mac
# or
.\whisper_env\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create server configuration:
```bash
cp config/server_config.template.sh config/server_config.sh
nano config/server_config.sh
```

Required server configurations:
- `PORT`: Port number for the server (default: 2024)
- `MODEL_SIZE`: Whisper model size (tiny, base, small, medium, large)
- `COMPUTE_TYPE`: CPU/GPU computation type (float16, float32)

### Client Setup

1. Clone the repository on the client machine:
```bash
git clone https://github.com/yourusername/whisper_cube.git
cd whisper_cube
```

2. Create client configuration:
```bash
cp config/client_config.template.sh config/client_config.sh
nano config/client_config.sh
```

Required client configurations:
- `SERVER_IP`: IP address of the server
- `SERVER_PORT`: Port number matching server configuration
- `NOTIFICATION_SOUND`: Path to sound file for notifications (optional)

## Usage

### Server-side Commands

Start the server:
```bash
./server_scripts/start_server.sh
```

Stop the server:
```bash
./server_scripts/stop_server.sh
```

### Client-side Commands

Start transcription:
```bash
./client_scripts/whisper_start.sh
```

Stop transcription:
```bash
./client_scripts/whisper_stop.sh
```

View server output:
```bash
./client_scripts/whisper_reattach.sh
```

## Troubleshooting

- If the server won't start, check if the port is already in use:
  ```bash
  lsof -i :2024  # Replace with your port number
  ```

- Ensure the virtual environment is activated before running scripts
- Check server logs for CUDA/GPU-related errors if using GPU acceleration

## License

[Your license information here]

