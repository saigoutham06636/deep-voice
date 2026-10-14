import os
import numpy as np
import scipy.io.wavfile as wav

def generate_voice_wave(is_ai, duration=3.0, sr=16000):
    """
    Synthesizes a mock voice signal.
    - If is_ai=False (Human): Includes natural pitch variation (jitter), amplitude envelope modulation (shimmer),
      formant transitions, natural pauses, and breathy noise.
    - If is_ai=True (AI): Has a flat/robotic pitch, no natural jitter, phase distortions, high-frequency artifacts,
      and periodic mechanical clicks or vocoder buzz.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    # 1. Fundamental Frequency (F0) and Harmonics
    if not is_ai:
        # Human: Natural pitch modulation (F0 around 85-260Hz for diverse speakers, with jitter/random walk)
        f0_base = np.random.uniform(85, 260)
        # Slow pitch variation (prosody) + fast micro-fluctuations (jitter)
        prosody = 15 * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz prosody
        jitter = 3 * np.random.normal(0, 1, len(t))
        # Filter jitter to make it smooth
        jitter = np.convolve(jitter, np.ones(50)/50, mode='same')
        f0 = f0_base + prosody + jitter
    else:
        # AI: Flat, rigid pitch (robotic, very minor variance)
        f0_base = np.random.uniform(85, 260)
        # Flat pitch contour or exact step-changes without smooth glide
        f0 = f0_base * np.ones_like(t)
        # Periodic step changes (typical of text-to-speech without prosody modeling)
        steps = int(duration * 2) # change twice per second
        step_len = len(t) // steps
        for s in range(steps):
            f0[s*step_len:(s+1)*step_len] = f0_base + (s % 3 - 1) * 8.0

    # Integrate frequency to get phase
    phase = 2 * np.pi * np.cumsum(f0) / sr
    
    # Generate fundamental + harmonics (vocal cord simulation)
    # Human voices have declining energy in higher harmonics (spectral tilt)
    # AI vocoders sometimes have metallic resonances or unnaturally loud high frequencies
    num_harmonics = 10 if not is_ai else 12
    for h in range(1, num_harmonics + 1):
        amplitude = (1.0 / h) if not is_ai else (1.2 / (h ** 0.8))
        # Add phase shift for human to sound natural; AI can be in-phase (coherent)
        p_shift = np.random.uniform(0, 2*np.pi) if not is_ai else 0.0
        signal += amplitude * np.sin(h * phase + p_shift)
        
    # 2. Vocal Tract Formant Simulation (bandpass filter simulation)
    # We will simulate formants by applying bandpass filters to make it sound "vocalic" (like vowels)
    # Simple bandpass filters (e.g. resonators at 500Hz, 1500Hz, 2500Hz)
    from scipy.signal import butter, lfilter
    
    def bandpass_filter(data, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)
        
    try:
        # Apply vocal formants (randomized cuts to mimic vocal tract shapes)
        low_cut = np.random.uniform(220, 380)
        high_cut = np.random.uniform(2800, 3800)
        signal = bandpass_filter(signal, low_cut, high_cut, sr, order=2)
    except Exception:
        pass # Fallback to raw synthesis if filter fails
        
    # 3. Noise and Breathiness
    # Human: Soft breath noise (colored noise) mixed with voice
    # AI: Flat white noise, high-frequency vocoder buzz, or robotic artifacts
    noise = np.random.normal(0, 0.05, len(t))
    if not is_ai:
        # Filter noise to vocal range (breathy sound)
        try:
            noise = bandpass_filter(noise, 1000, 3000, sr, order=2)
        except Exception:
            pass
        # Modulate noise with amplitude envelope of the voice (natural gating)
        noise = noise * (0.1 + 0.9 * np.abs(np.sin(2 * np.pi * 0.5 * t)))
        signal += 0.1 * noise
    else:
        # AI: Unmodulated high-frequency white noise or metallic hiss (e.g., above 6kHz)
        try:
            high_noise = bandpass_filter(noise, 6000, 7800, sr, order=2)
            signal += 0.15 * high_noise
        except Exception:
            signal += 0.05 * noise
        # AI artifact: 8kHz vocoder frame click (frame rate 50Hz, click every 20ms)
        click_interval = int(sr * 0.02) # 20ms
        clicks = np.zeros_like(t)
        clicks[::click_interval] = 0.2
        signal += clicks
        
    # 4. Amplitude Envelope (Shimmer and Speech Pauses)
    # Human: Phrases with soft envelopes, pauses between words
    if not is_ai:
        # Natural pauses: silence sections of the audio
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t) # smooth envelope
        # Add random dropouts (pauses)
        pause_start = int(sr * 0.8)
        pause_end = int(sr * 1.3)
        envelope[pause_start:pause_end] *= 0.05
        signal *= envelope
        # Shimmer: small random envelope ripples
        shimmer_mod = 1.0 + 0.03 * np.random.normal(0, 1, len(t))
        signal *= shimmer_mod
    else:
        # AI: Steady, continuous speech without natural pauses, or sharp sudden gates
        envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.4 * t)
        signal *= envelope

    # 5. Ambient Room/Microphone Static Hiss (added to BOTH Human and AI for real microphone compatibility)
    ambient_noise = np.random.normal(0, np.random.uniform(0.005, 0.025), len(t))
    signal += ambient_noise

    # Normalize to -1 to 1 and apply random gain scaling
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = (signal / max_val) * np.random.uniform(0.7, 0.95)
        
    # Soft fade-in and fade-out to prevent pops
    fade_len = int(sr * 0.05)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out
    
    return signal.astype(np.float32)

def generate_mock_dataset(base_dir="backend/data", num_samples=30):
    """
    Generates mock dataset for Human and AI voices and saves them to folders.
    """
    train_human_dir = os.path.join(base_dir, "train", "human")
    train_ai_dir = os.path.join(base_dir, "train", "ai")
    
    os.makedirs(train_human_dir, exist_ok=True)
    os.makedirs(train_ai_dir, exist_ok=True)
    
    print(f"Generating synthetic dataset in '{base_dir}'...")
    
    # Generate Human Samples
    for i in range(num_samples):
        y = generate_voice_wave(is_ai=False, duration=3.0, sr=16000)
        file_path = os.path.join(train_human_dir, f"human_{i:03d}.wav")
        wav.write(file_path, 16000, (y * 32767).astype(np.int16))
        
    # Generate AI Samples
    for i in range(num_samples):
        y = generate_voice_wave(is_ai=True, duration=3.0, sr=16000)
        file_path = os.path.join(train_ai_dir, f"ai_{i:03d}.wav")
        wav.write(file_path, 16000, (y * 32767).astype(np.int16))
        
    print(f"Dataset generated successfully! {num_samples} samples per class.")

if __name__ == "__main__":
    generate_mock_dataset()
