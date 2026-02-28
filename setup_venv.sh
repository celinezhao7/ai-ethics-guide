#!/usr/bin/env bash
set -euo pipefail

# Creates a virtual environment in .venv and installs requirements
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Virtual environment created at .venv and dependencies installed."
echo "Copy .env.example to .env and add your GENERATIVEAI_API_KEY before running streamlit."
