#!/usr/bin/env bash

sudo apt update
sudo apt install -y python3-pip

pip3 install -r $PROJECT_ROOT/requirements.txt
# Run the FastAPI server
nohup uvicorn main:app --host 0.0.0.0 --port 8080 > server.log 2>&1 &

echo "FastAPI server started on port 8080."
