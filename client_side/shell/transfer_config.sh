#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../config/config.sh"

echo "Transferring configuration to server..."

# Create config directory on server if it doesn't exist
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p $REMOTE_DIR/config"

# Transfer all configuration files
scp -P $SSH_PORT "$SCRIPT_DIR/../../config/config.py" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/config/"
scp -P $SSH_PORT "$SCRIPT_DIR/../../config/config.sh" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/config/"

echo "Configuration transfer complete!"

# Verify transfer
echo "Verifying configuration files on server..."
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "ls -l $REMOTE_DIR/config/config.*" 