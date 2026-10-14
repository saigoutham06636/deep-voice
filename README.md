# DeepFake Voice Phishing (Vishing) Detection System

## Overview
This project is an AI-powered security application designed to detect Voice Phishing (Vishing) scams in real-time. It analyzes audio from phone calls to determine if the voice is AI-generated (DeepFake) and transcribes the speech to scan for social engineering and cybersecurity threat markers.

## Features
*   **Synthetic Voice Detection:** Uses a PyTorch Convolutional Neural Network (CNN) to classify audio as "Human" or "AI Generated" based on acoustic features.
*   **Speech-to-Text Transcription:** Integrates with Google Speech Recognition to transcribe the spoken audio into text.
*   **NLP Threat Analysis:** Scans the transcript using heuristic rules for common phishing tactics, categorizing threats into:
    *   Financial Requests (e.g., wire transfers, crypto)
    *   Identity & Credentials (e.g., SSN, passwords, OTPs)
    *   Urgency & Fear (e.g., immediate action, arrest threats)
    *   Authority Pressure (e.g., posing as police or bank officials)
*   **Audio Digital Signal Processing (DSP):** Computes and visualizes waveforms and Mel-spectrograms using `librosa` and `scipy`.
*   **Call Simulator:** Includes an interactive simulator with preset scenarios (Bank Scam, Grandchild Emergency, Tech Support, Normal Call) to test the system in real-time.
*   **Auto-Setup Model:** The system automatically bootstraps a training model with mock data if a pre-trained PyTorch model isn't found.

## Project Structure
*   `backend/app.py`: The main Flask server application handling API requests, simulations, and routing.
*   `backend/audio_processor.py`: Contains DSP logic for loading audio and extracting features (Waveforms, Mel-Spectrograms, Spectral Features).
*   `backend/classifier.py`: The PyTorch CNN model definition, training loop, and inference logic for classifying AI vs. Human voices.
*   `backend/phishing_detector.py`: Handles audio transcription and NLP threat analysis on the transcribed text.
*   `backend/mock_data_generator.py`: Generates synthetic WAV data to bootstrap the model for demonstration purposes.
*   `backend/templates/`: HTML templates for the frontend UI.
*   `backend/static/`: Static assets (CSS, JS) for the frontend.

## Requirements
*   Python 3.x
*   PyTorch
*   Librosa
*   Flask
*   SpeechRecognition
*   NumPy & SciPy

*(See `requirements.txt` for full dependencies)*

## Setup & Installation

1.  **Navigate to the project folder:**
    Open your terminal or command prompt and go to the project directory.

2.  **Run the Setup Script:**
    Double-click or run `setup.bat`.
    This script will:
    *   Create a Python virtual environment (`venv`).
    *   Activate the environment.
    *   Install all required packages from `requirements.txt`.

## Running the Application

1.  **Start the Server:**
    Double-click or run `run.bat`.
    *   It will automatically check if the virtual environment exists and activate it.
    *   It starts the Flask backend server.
    *   On the first run, the system will auto-setup and train a baseline CNN model if one doesn't exist.

2.  **Access the UI:**
    Open your web browser and navigate to:
    ```
    http://127.0.0.1:5000
    ```

## Usage
*   **Upload Audio:** You can upload a `.wav` file through the interface to analyze a pre-recorded call.
*   **Simulate a Call:** Use the built-in simulator on the frontend to generate and analyze specific scenarios (e.g., Tech Support Scam). The system will return a risk score, severity level, and a detailed breakdown of detected threats.
