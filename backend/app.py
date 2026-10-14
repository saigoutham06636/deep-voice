import os
import tempfile
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import scipy.io.wavfile as wav

import backend.audio_processor as ap
import backend.classifier as cl
import backend.phishing_detector as pd
import backend.mock_data_generator as dg
import backend.dataset_loader as dl

app = Flask(
    __name__, 
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')),
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
)

# Model path config
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
MODEL_PATH = os.path.join(DATA_DIR, "voice_cnn.pth")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")

# Global training logs for UI display
training_logs = []
is_training = False

def auto_setup_model():
    """
    Checks if model exists. If not, generates mock data and trains model in a background thread 
    or synchronously on startup so it's ready.
    """
    if not os.path.exists(MODEL_PATH):
        print("[Auto-Setup] PyTorch model weights not found. Initializing model bootstrap...")
        try:
            # 1. Generate data
            loaded = dl.load_real_dataset(base_dir=DATA_DIR)
            if not loaded:
                print("[Auto-Setup] Real dataset not found, generating mock data...")
                dg.generate_mock_dataset(base_dir=DATA_DIR, num_samples=25)
            # 2. Train model
            print("[Auto-Setup] Training baseline classifier...")
            cl.train_network(data_dir=TRAIN_DATA_DIR, model_save_path=MODEL_PATH, epochs=8)
            print("[Auto-Setup] Model bootstrapped successfully!")
        except Exception as e:
            print(f"[ERROR] Auto-setup failed: {e}. Heuristics fallback will be active.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """
    Receives an uploaded audio file, processes features, runs CNN classifier & transcript scanning.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    # Save audio file to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}.wav")
    audio_file.save(temp_path)
    
    try:
        # 1. Load signal
        y, sr = ap.load_audio(temp_path)
        
        # 2. Extract downsampled waveform and mel-spectrogram
        waveform = ap.get_waveform_points(y, max_points=800)
        spectrogram, times, freqs = ap.compute_mel_spectrogram(y, sr, n_mels=40)
        
        # 3. DSP Spectral metrics
        spectral_feats = ap.compute_spectral_features(y, sr)
        
        # 4. Run PyTorch CNN Inference
        label, confidence, engine = cl.predict_voice(temp_path, MODEL_PATH)
        
        # 5. Transcription & Threat Analysis
        transcript = pd.transcribe_audio(temp_path)
        phishing_report = pd.analyze_transcript(transcript)
        
        # Calculate overall system security verdict
        is_ai = label == "AI Generated"
        threat_score = phishing_report["risk_score"]
        
        # Threat level classification logic:
        # AI Voice + High Phishing Text -> CRITICAL SCAM VISHING
        # AI Voice + Low Phishing Text -> AI Voice Clone (Simulated/Suspicious)
        # Human Voice + High Phishing Text -> Human Scam/Phishing (Vishing)
        # Human Voice + Low Phishing Text -> Safe Call
        if is_ai and threat_score > 50:
            verdict = "CRITICAL: AI-Generated Phishing Scam"
            severity = "critical"
        elif is_ai:
            verdict = "SUSPICIOUS: Synthetic AI Voice Detected"
            severity = "warning"
        elif threat_score > 50:
            verdict = "WARNING: Human-Voiced Phishing Scam"
            severity = "warning"
        else:
            verdict = "SECURE: Human Voice (No Scam Markers)"
            severity = "safe"
            
        result = {
            "status": "success",
            "verdict": verdict,
            "severity": severity,
            "voice_classification": {
                "label": label,
                "confidence": confidence,
                "engine": engine
            },
            "dsp_features": spectral_feats,
            "phishing_analysis": phishing_report,
            "waveform": waveform,
            "spectrogram": {
                "data": spectrogram,
                "time": times,
                "freq": freqs
            },
            "transcript": transcript or "No voice speech detected in transmission."
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.route('/api/simulate', methods=['POST'])
def simulate_call():
    """
    Generates a live voice phishing WAV simulation on-the-fly and runs analysis.
    This demonstrates real DSP calculations, CNN forward-pass, and speech processing in real-time.
    """
    data = request.get_json() or {}
    scenario = data.get("scenario", "bank_scam")
    
    # Map scenarios to AI/Human properties
    scenario_configs = {
        "bank_scam": {"is_ai": True, "duration": 3.0},
        "grandchild_emergency": {"is_ai": True, "duration": 3.0},
        "tech_support": {"is_ai": True, "duration": 3.0},
        "normal_call": {"is_ai": False, "duration": 3.0},
        "doctor_appointment": {"is_ai": False, "duration": 3.0}
    }
    
    config = scenario_configs.get(scenario, {"is_ai": False, "duration": 3.0})
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"sim_{scenario}_{os.urandom(4).hex()}.wav")
    
    try:
        # 1. Synthesize signal dynamically
        y = dg.generate_voice_wave(is_ai=config["is_ai"], duration=config["duration"])
        
        # 2. Write WAV to disk (scipy.io.wavfile)
        wav.write(temp_path, 16000, (y * 32767).astype(np.int16))
        
        # 3. Process analysis pipeline
        y_loaded, sr = ap.load_audio(temp_path)
        waveform = ap.get_waveform_points(y_loaded, max_points=800)
        spectrogram, times, freqs = ap.compute_mel_spectrogram(y_loaded, sr, n_mels=40)
        spectral_feats = ap.compute_spectral_features(y_loaded, sr)
        label, confidence, engine = cl.predict_voice(temp_path, MODEL_PATH)
        
        # Fetch the transcript directly from pd.transcribe_audio (which identifies the preset keyword)
        transcript = pd.transcribe_audio(temp_path)
        phishing_report = pd.analyze_transcript(transcript)
        
        # Overall verdict logic
        is_ai = label == "AI Generated"
        threat_score = phishing_report["risk_score"]
        
        if is_ai and threat_score > 50:
            verdict = "CRITICAL: AI-Generated Phishing Scam"
            severity = "critical"
        elif is_ai:
            verdict = "SUSPICIOUS: Synthetic AI Voice Detected"
            severity = "warning"
        elif threat_score > 50:
            verdict = "WARNING: Human-Voiced Phishing Scam"
            severity = "warning"
        else:
            verdict = "SECURE: Human Voice (No Scam Markers)"
            severity = "safe"
            
        result = {
            "status": "success",
            "verdict": verdict,
            "severity": severity,
            "voice_classification": {
                "label": label,
                "confidence": confidence,
                "engine": engine
            },
            "dsp_features": spectral_feats,
            "phishing_analysis": phishing_report,
            "waveform": waveform,
            "spectrogram": {
                "data": spectrogram,
                "time": times,
                "freq": freqs
            },
            "transcript": transcript
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Triggers model training. 
    Can be run synchronously (returns history) or updates training log variable.
    """
    global is_training, training_logs
    if is_training:
        return jsonify({"error": "Training already in progress"}), 400
        
    is_training = True
    training_logs = []
    
    def log_callback(msg):
        training_logs.append(msg)
        print(f"[Training Info] {msg}")
        
    def run_training():
        global is_training
        try:
            # Generate fresh data
            loaded = dl.load_real_dataset(base_dir=DATA_DIR)
            if not loaded:
                log_callback("Real dataset not found, generating mock data instead.")
                dg.generate_mock_dataset(base_dir=DATA_DIR, num_samples=30)
            # Train PyTorch
            cl.train_network(data_dir=TRAIN_DATA_DIR, model_save_path=MODEL_PATH, epochs=12, callback=log_callback)
        except Exception as e:
            log_callback(f"ERROR: Training failed: {str(e)}")
        finally:
            is_training = False
            
    # Run training in background or foreground.
    # Because our mock dataset is small (60 samples), training only takes ~3-5 seconds on CPU.
    # So we can run it synchronously to immediately return the log results to the user!
    run_training()
    
    return jsonify({
        "status": "success",
        "logs": training_logs,
        "model_path": MODEL_PATH
    })

@app.route('/api/training_status', methods=['GET'])
def get_training_status():
    global is_training, training_logs
    return jsonify({
        "is_training": is_training,
        "logs": training_logs
    })

if __name__ == '__main__':
    # Initialize baseline files on startup
    auto_setup_model()
    # Run on default port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
