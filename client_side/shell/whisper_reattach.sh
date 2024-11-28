#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

# Check if server is running
if ! ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "pgrep -f 'python3.*audio_server.py'" > /dev/null; then
    echo "Whisper server is not running."
    exit 1
fi

# Check/create SSH tunnel if needed
if ! pgrep -f "ssh.*$SERVER_PORT:127.0.0.1:$SERVER_PORT" > /dev/null; then
    echo "Creating SSH tunnel..."
    ssh -f -N -L $SERVER_PORT:127.0.0.1:$SERVER_PORT -p $SSH_PORT $SSH_USER@$SSH_HOST
    echo "SSH tunnel created."
fi

# Reattach to the screen session
echo "Reattaching to server output..."
ssh -t -p $SSH_PORT $SSH_USER@$SSH_HOST "screen -r $SCREEN_SESSION_NAME"