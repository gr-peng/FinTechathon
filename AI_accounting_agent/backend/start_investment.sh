#!/bin/bash

# Kill any process running on port 8020
PID=$(lsof -ti:8020)
if [ ! -z "$PID" ]; then
    echo "Killing process on port 8020 (PID: $PID)..."
    kill -9 $PID
fi

# Activate venv
source .venv/bin/activate

# Set env vars
export QWEN_MODEL_DIR="/home/bld/data/data4/admin/fintechcom/Qwen3-4B"
export PORT=8020

# Start server
echo "Starting Investment Agent Backend on port 8020..."
python investment_app.py
