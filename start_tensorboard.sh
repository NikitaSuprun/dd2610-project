#!/bin/bash

# Complete TensorBoard Setup Script with Nginx and Authentication
# Usage: ./start_tensorboard.sh [port] [setup]
#
# Examples:
#   ./start_tensorboard.sh              # Just start TensorBoard
#   ./start_tensorboard.sh 6006 setup   # Full setup with nginx

set -e  # Exit on error

PORT=${1:-6006}
LOGDIR="runs"
SETUP_MODE=${2:-""}

echo "=========================================="
echo "TensorBoard Setup & Start"
echo "=========================================="
echo ""

# Function to check if running as root when needed
check_sudo() {
    if [ "$EUID" -ne 0 ] && [ "$SETUP_MODE" == "setup" ]; then
        echo "Error: Setup mode requires sudo privileges"
        echo "Please run: sudo ./start_tensorboard.sh $PORT setup"
        exit 1
    fi
}

# Function to setup nginx
setup_nginx() {
    echo "Setting up Nginx..."

    # Install nginx if not present
    if ! command -v nginx &> /dev/null; then
        echo "Installing nginx..."
        apt update
        apt install -y nginx
    fi

    # Install apache2-utils for htpasswd
    if ! command -v htpasswd &> /dev/null; then
        echo "Installing apache2-utils..."
        apt install -y apache2-utils
    fi

    # Create password file if it doesn't exist
    if [ ! -f /etc/nginx/.htpasswd ]; then
        echo ""
        echo "Creating authentication for TensorBoard..."
        echo "Enter username for TensorBoard access:"
        read -r USERNAME
        htpasswd -c /etc/nginx/.htpasswd "$USERNAME"
        echo "Password file created at /etc/nginx/.htpasswd"
    else
        echo "Password file already exists at /etc/nginx/.htpasswd"
        echo "To add users: sudo htpasswd /etc/nginx/.htpasswd username"
    fi

    # Copy nginx config
    echo "Installing nginx configuration..."
    cp nginx-tensorboard.conf /etc/nginx/sites-available/tensorboard

    # Enable site if not already enabled
    if [ ! -L /etc/nginx/sites-enabled/tensorboard ]; then
        ln -s /etc/nginx/sites-available/tensorboard /etc/nginx/sites-enabled/tensorboard
    fi

    # Test nginx config
    echo "Testing nginx configuration..."
    nginx -t

    # Open firewall ports
    echo "Configuring firewall..."
    if command -v ufw &> /dev/null; then
        ufw allow 80/tcp
        ufw --force enable
        echo "Firewall configured (port 80 open)"
    fi

    # Restart nginx
    echo "Restarting nginx..."
    systemctl restart nginx
    systemctl enable nginx

    echo ""
    echo "✓ Nginx setup complete!"
    echo ""
}

# Main execution
if [ "$SETUP_MODE" == "setup" ]; then
    check_sudo
    setup_nginx
fi

# Create runs directory if it doesn't exist
mkdir -p "$LOGDIR"

# Kill any existing tensorboard processes
echo "Stopping any existing TensorBoard processes..."
pkill -f "tensorboard --logdir" 2>/dev/null || true
sleep 1

# Find tensorboard command
# First try in virtual environment, then in PATH
TENSORBOARD_CMD=""
if [ -f ".venv/bin/tensorboard" ]; then
    TENSORBOARD_CMD=".venv/bin/tensorboard"
elif command -v tensorboard &> /dev/null; then
    TENSORBOARD_CMD="tensorboard"
else
    echo "✗ Error: tensorboard command not found!"
    echo "Please install tensorboard: uv pip install tensorboard"
    echo "Or activate your virtual environment first"
    exit 1
fi

# Start tensorboard in background
echo "Starting TensorBoard..."
echo "  Port: $PORT"
echo "  Log directory: $LOGDIR"
echo "  Command: $TENSORBOARD_CMD"
echo ""

nohup $TENSORBOARD_CMD --logdir=$LOGDIR --host=0.0.0.0 --port=$PORT > tensorboard.log 2>&1 &
TB_PID=$!

# Wait a moment and check if it started
sleep 2
if ps -p $TB_PID > /dev/null; then
    echo "✓ TensorBoard started successfully (PID: $TB_PID)"
else
    echo "✗ TensorBoard failed to start. Check tensorboard.log for errors."
    exit 1
fi

echo ""
echo "=========================================="
echo "Access Information"
echo "=========================================="

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

if [ "$SETUP_MODE" == "setup" ] || [ -f /etc/nginx/sites-enabled/tensorboard ]; then
    echo "TensorBoard URL (with nginx):"
    echo "  http://$SERVER_IP/tensorboard/"
    echo ""
    echo "Login credentials: Check /etc/nginx/.htpasswd"
    echo "To add users: sudo htpasswd /etc/nginx/.htpasswd username"
else
    echo "Direct access (no nginx):"
    echo "  http://localhost:$PORT"
    echo "  http://$SERVER_IP:$PORT"
    echo ""
    echo "To setup nginx with authentication:"
    echo "  sudo ./start_tensorboard.sh $PORT setup"
fi

echo ""
echo "=========================================="
echo "Management Commands"
echo "=========================================="
echo "View logs:      tail -f tensorboard.log"
echo "Stop:           pkill -f tensorboard"
echo "Check status:   ps aux | grep tensorboard"
echo "Restart nginx:  sudo systemctl restart nginx"
echo "=========================================="
echo ""
