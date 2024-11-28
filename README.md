# FastWhisperServer

## Introduction

FastWhisperServer enables instant voice-to-text transcription using OpenAI's Whisper model. It's designed for developers who want GPU-accelerated transcription without requiring a GPU on their local machine.

### Key Benefits

- **Instant Voice-to-Clipboard**: Record with keyboard shortcuts, get text in your clipboard
- **Remote GPU Processing**: Use a powerful server while running lightweight clients (tested: GTX 1050, ~2s for 30s audio)
- **Secure Communication**: All traffic tunneled through SSH
- **Perfect for AI Development**: Quickly dictate prompts to your AI assistants

### How It Works

1. **Server Side**: 
   - GPU machine runs the Whisper transcription server
   - Uses faster-whisper for quick processing

2. **Client Side**:
   - Records audio with sound notifications
   - Automatically copies transcription to clipboard
   - Uses SSH tunnel for security

### Typical Workflow

1. Press shortcut (e.g., `Ctrl+Alt+R`) → start sound plays
2. Speak your text
3. Press stop shortcut (`Ctrl+Alt+S`) → stop sound plays
4. Wait for completion sound (~2 seconds)
5. Press `Ctrl+V` to paste anywhere

## Prerequisites

Server:
- Python 3.8+
- CUDA-capable GPU
- faster-whisper
- CUDA toolkit

Client:
- Python 3.8+
- `xclip`
- `python3-pyaudio`
- `ssh`

## Installation

### 1. Client Setup

1. Clone and prepare:
```bash
git clone git@github.com:Infraviored/FastWhisperServer.git
cd FastWhisperServer
python3 -m venv whisper_env
source whisper_env/bin/activate
```

2. Install dependencies:
```bash
sudo apt-get install xclip python3-pyaudio
pip3 install -r requirements.txt
```

3. Configure:
```bash
cp config/config.template.py config/config.py
cp config/config.template.sh config/config.sh
```

Edit both files with your settings (server address, API key, etc.)

### 2. Server Setup

1. Clone and prepare:
```bash
git clone git@github.com:Infraviored/FastWhisperServer.git
cd FastWhisperServer
python3 -m venv whisper_env
source whisper_env/bin/activate
pip3 install -r requirements.txt
```

2. Transfer configuration:
```bash
./client_side/shell/transfer_config.sh
```

### 3. Setup Shortcuts

Add keyboard shortcuts in your system settings:
```
Start Recording: python3 /path/to/FastWhisperServer/client_side/audio_client.py --start
Stop Recording: python3 /path/to/FastWhisperServer/client_side/audio_client.py --stop
```

## Usage

### Server

```bash
./server_scripts/start_server.sh
./server_scripts/stop_server.sh
```

### Client

```bash
# Test setup
python3 client_side/audio_client.py --test-sounds

# Or use keyboard shortcuts configured above
```

## Troubleshooting

- **No Connection**: Check SSH tunnel
- **Audio Issues**: Test with `arecord -l`
- **Already Running**: Delete `/tmp/whisper_recorder.pid`
