#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

# Function to cleanup local processes
cleanup_local() {
    echo "Cleaning up local processes..."
    
    # Kill any hanging audio client processes
    pkill -f "python3.*audio_client.py"
    
    # Remove PID file if it exists
    rm -f /tmp/whisper_recorder.pid
    
    # Kill any hanging SSH tunnels
    pkill -f "ssh.*$SERVER_PORT:127.0.0.1:$SERVER_PORT"
}

# Register cleanup on script exit
trap cleanup_local EXIT

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
    echo "Whisper server is already running. Reattaching..."
    
    # Ensure tunnel exists
    if ! check_local_tunnel; then
        echo "Creating SSH tunnel..."
        ssh -f -N -L $SERVER_PORT:127.0.0.1:$SERVER_PORT -p $SSH_PORT $SSH_USER@$SSH_HOST
        echo "SSH tunnel created."
    fi
    
    # Reattach to the screen session
    ssh -t -p $SSH_PORT $SSH_USER@$SSH_HOST "screen -r $SCREEN_SESSION_NAME"
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