#!/bin/bash

echo "Starting Movie Recommendation System..."
echo "=========================================="
echo ""

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created!"
    echo ""
fi

echo "Activating virtual environment..."
source venv/bin/activate

if [ ! -f "venv/.requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch venv/.requirements_installed
    echo "Requirements installed!"
    echo ""
else
    echo "Requirements already installed!"
    echo ""
fi

echo "Launching Streamlit dashboard..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
