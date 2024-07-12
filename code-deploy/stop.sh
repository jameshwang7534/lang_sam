#!/usr/bin/env bash

# Find the process ID (PID) of the running FastAPI server and terminate it
PID=$(pgrep -f "uvicorn main:app")
if [ -z "$PID" ]; then
  echo "FastAPI server is not running."
else
  kill $PID
  echo "FastAPI server stopped."
fi