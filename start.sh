#!/usr/bin/env bash

# Stop on error and clean up background processes on exit
set -e
trap 'echo "Shutting down servers..."; kill 0' SIGINT SIGTERM EXIT

echo "=========================================="
echo "🚀 Starting Search Engine Playground..."
echo "=========================================="

echo "Starting FastAPI Backend (Port 8000)..."
# Start the backend in the background
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend to initialize (ML models may take ~15 seconds to load)..."
# A quick loop to check if the backend health endpoint is up
until curl -s http://localhost:8000/docs > /dev/null; do
    sleep 2
done
echo "✅ Backend is up and running!"

echo "Starting Streamlit Frontend..."
# Start the frontend
uv run streamlit run src/frontend/app.py &
FRONTEND_PID=$!

echo "=========================================="
echo "✨ Everything is running!"
echo "Press Ctrl+C to stop both servers."
echo "=========================================="

# Wait indefinitely until interrupted
wait
