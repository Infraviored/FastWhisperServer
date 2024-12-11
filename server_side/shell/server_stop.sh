#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

cd $REMOTE_DIR

# Kill screen session if it exists
if screen -list | grep -q "$SCREEN_SESSION_NAME"; then
    echo "Killing screen session..."
    screen -S $SCREEN_SESSION_NAME -X quit
    sleep 1  # Give it a moment to clean up
fi

# Kill any remaining server processes
if pgrep -f "python3.*audio_server.py" > /dev/null; then
    echo "Killing remaining server process..."
    pkill -f "python3.*audio_server.py"
    sleep 1  # Give it a moment to clean up
fi

# Double check nothing is left
if pgrep -f "python3.*audio_server.py" > /dev/null; then
    echo "Warning: Server process still running. Forcing kill..."
    pkill -9 -f "python3.*audio_server.py"
fi

echo "Whisper server stopped."