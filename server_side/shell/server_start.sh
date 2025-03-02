#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

# Stop display manager before starting server
sudo systemctl stop display-manager

cd $REMOTE_DIR

# Check if server is already running and handle screen session
if pgrep -f "python3.*audio_server.py" > /dev/null; then
    echo "Whisper server is already running."
    
    # Check if screen session exists
    if ! screen -list | grep -q "$SCREEN_SESSION_NAME"; then
        echo "Screen session not found. Cleaning up and restarting..."
        pkill -f "python3.*audio_server.py"
    else
        # Try to reattach
        if ! screen -r $SCREEN_SESSION_NAME; then
            echo "Cannot reattach to screen. Cleaning up and restarting..."
            screen -S $SCREEN_SESSION_NAME -X quit
            pkill -f "python3.*audio_server.py"
        else
            exit 0
        fi
    fi
fi

# Start fresh
source whisper_env/bin/activate
screen -S $SCREEN_SESSION_NAME -d -m python3 server_side/audio_server.py
screen -r $SCREEN_SESSION_NAME