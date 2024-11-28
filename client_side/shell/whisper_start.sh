#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../config/config.sh"

# Function to check if server is running on remote host
check_remote_server() {
    ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "pgrep -f 'python3.*audio_server.py'" > /dev/null
    return $?
}

# Function to check if local tunnel exists
check_local_tunnel() {
    pgrep -f "ssh.*$SERVER_PORT:127.0.0.1:$SERVER_PORT" > /dev/null
    return $?
}

# Main logic
if check_remote_server; then
    echo "Whisper server is already running."
    
    if ! check_local_tunnel; then
        echo "Creating SSH tunnel..."
        ssh -f -N -L $SERVER_PORT:127.0.0.1:$SERVER_PORT -p $SSH_PORT $SSH_USER@$SSH_HOST
        echo "SSH tunnel created."
    else
        echo "SSH tunnel already exists."
    fi
else
    echo "Starting Whisper server..."
    # Create tunnel if it doesn't exist
    if ! check_local_tunnel; then
        echo "Creating SSH tunnel..."
        ssh -f -N -L $SERVER_PORT:127.0.0.1:$SERVER_PORT -p $SSH_PORT $SSH_USER@$SSH_HOST
    fi
    
    # Start the server and show live output
    ssh -t -p $SSH_PORT $SSH_USER@$SSH_HOST "cd $REMOTE_DIR && ./server_side/shell/server_start.sh"
fi