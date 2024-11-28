# Whisper Cube

A client-server application for real-time audio transcription using OpenAI's Whisper model.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whisper_cube.git
cd whisper_cube
```

2. Create configuration:
```bash
cp config/config.template.sh config/config.sh
```

3. Edit the configuration:
```bash
nano config/config.sh
```

4. Set up virtual environment:
```bash
python -m venv whisper_env
source whisper_env/bin/activate
pip install -r requirements.txt
```

## Usage

### Start Server
```bash
./scripts/client/whisper_start.sh
```

### Stop Server
```bash
./scripts/client/whisper_stop.sh
```

### Reattach to Server Output
```bash
./scripts/client/whisper_reattach.sh
```

