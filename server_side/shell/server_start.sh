#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../config/config.sh"

cd $REMOTE_DIR

# Check if server is already running
if pgrep -f "python3.*audio_server.py" > /dev/null; then
    echo "Whisper server is already running."
    exit 1
fi

# Activate virtual environment and start server
source whisper_env/bin/activate

# Start in a detached screen session and immediately attach to it
screen -S $SCREEN_SESSION_NAME -d -m python3 server_side/audio_server.py
screen -r $SCREEN_SESSION_NAME