#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../config/config.sh"

cd $REMOTE_DIR

# Find and kill the screen session
if screen -list | grep -q "$SCREEN_SESSION_NAME"; then
    screen -S $SCREEN_SESSION_NAME -X quit
fi

# Double-check and kill any remaining server process
if pgrep -f "python3.*audio_server.py" > /dev/null; then
    pkill -f "python3.*audio_server.py"
fi

echo "Whisper server stopped."