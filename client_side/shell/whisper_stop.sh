#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

echo "Stopping Whisper server..."

# Execute the stop script on the server
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "cd $REMOTE_DIR && ./server_side/shell/server_stop.sh"

# Kill the local SSH tunnel
if pgrep -f "ssh.*$SERVER_PORT:127.0.0.1:$SERVER_PORT" > /dev/null; then
    echo "Closing SSH tunnel..."
    pkill -f "ssh.*$SERVER_PORT:127.0.0.1:$SERVER_PORT"
    echo "SSH tunnel closed."
fi