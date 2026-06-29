#!/bin/bash
set -e

echo "========================================="
echo " Starting SafeChat Hugging Face Space... "
echo "========================================="

# 1. Start the ML Service Backend (FastAPI) in the background
echo "Starting FastAPI backend on port 8000..."
cd ml-service
# Use 1 worker to ensure models are only loaded into memory once
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 &
BACKEND_PID=$!

# 2. Wait for the backend to initialize
echo "Waiting for models to load..."
# Give it 15 seconds for HingBERT and the gatekeeper to download/load into memory
sleep 15

# 3. Start the Streamlit Frontend in the foreground
echo "Starting Streamlit UI on port 7860..."
cd ../python-frontend
# Set environment variable so Streamlit knows where the local backend is
export SAFECHAT_API_URL="http://127.0.0.1:8000"

# HF Spaces requires the app to bind to 0.0.0.0 and port 7860
streamlit run app.py --server.port=7860 --server.address=0.0.0.0
