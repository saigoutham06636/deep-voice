import os
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

# We will import librosa dynamically in functions to be robust,
# and provide pure numpy/scipy fallbacks if librosa or soundfile fails.

def load_audio(file_path, target_sr=16000):
    """
    Loads an audio file and resamples it to target_sr.
    Tries to use librosa/soundfile first, falls back to scipy.io.wavfile for WAVs.
    """
    try:
        # pyrefly: ignore [missing-import]
        import librosa
        # pyrefly: ignore [missing-import]
        import soundfile as sf
        y, sr = librosa.load(file_path, sr=target_sr, duration=5.0)
        return y, sr
    except Exception as e:
        print(f"[AudioProcessor Warning] Librosa load failed: {e}. Trying scipy fallback...")
        
    # Fallback for WAV files
    try:
        sr, y = wav.read(file_path)
        # Truncate to 5 seconds to speed up processing
        max_samples = 5 * sr
        if len(y) > max_samples:
            y = y[:max_samples]
            
        # Convert to float32 mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
            
        # Resample using scipy if needed
        if sr != target_sr:
            num_samples = int(len(y) * target_sr / sr)
            y = signal.resample(y, num_samples)
            sr = target_sr
            
        return y.astype(np.float32), sr
    except Exception as fallback_e:
        raise ValueError(f"Could not load audio file using Librosa or Scipy fallback. Error: {fallback_e}")

def get_waveform_points(y, max_points=1000):
    """
    Downsamples the raw waveform to a fixed number of points for fast visualization in the UI.
    """
    if len(y) <= max_points:
        return y.tolist()
    
    # Take peak values in bins to preserve visual envelope
    bin_size = len(y) // max_points
    waveform_peaks = []
    for i in range(max_points):
        start = i * bin_size
        end = start + bin_size
        chunk = y[start:end]
        if len(chunk) > 0:
            val = float(np.max(np.abs(chunk)) * (1 if np.max(chunk) >= -np.min(chunk) else -1))
            waveform_peaks.append(val)
        else:
            waveform_peaks.append(0.0)
    return waveform_peaks

def compute_mel_spectrogram(y, sr, n_mels=64, n_fft=1024, hop_length=512):
    """
    Computes the Mel-spectrogram of the audio.
    Returns:
        spectrogram_db: log-scale Mel spectrogram (2D list of shape [n_mels, time_steps])
        time_axis: list of time stamps
        freq_axis: list of frequency stamps
    """
    try:
        # pyrefly: ignore [missing-import]
        import librosa
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize between 0 and 1 for easier UI mapping
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        if mel_max - mel_min > 0:
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel_db)
            
        return mel_norm.tolist(), librosa.frames_to_time(np.arange(mel_norm.shape[1]), sr=sr, hop_length=hop_length).tolist(), librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr/2).tolist()
    except Exception as e:
        print(f"[AudioProcessor Warning] Librosa mel spectrogram failed: {e}. Generating fallback spectrogram...")
        
    # Scipy spectrogram fallback
    frequencies, times, Sxx = signal.spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    # Convert to log-scale
    with np.errstate(divide='ignore'):
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Downsample frequency axis to n_mels (simple binning)
    if len(frequencies) > n_mels:
        bin_size = len(frequencies) // n_mels
        Sxx_db_reduced = []
        for i in range(n_mels):
            start = i * bin_size
            end = start + bin_size
            Sxx_db_reduced.append(np.mean(Sxx_db[start:end, :], axis=0))
        Sxx_db_reduced = np.array(Sxx_db_reduced)
    else:
        Sxx_db_reduced = Sxx_db
        
    # Normalize between 0 and 1
    db_min = Sxx_db_reduced.min()
    db_max = Sxx_db_reduced.max()
    if db_max - db_min > 0:
        norm_spec = (Sxx_db_reduced - db_min) / (db_max - db_min)
    else:
        norm_spec = np.zeros_like(Sxx_db_reduced)
        
    dummy_freqs = np.linspace(0, sr / 2, n_mels).tolist()
    return norm_spec.tolist(), times.tolist(), dummy_freqs

def extract_mfcc(y, sr, n_mfcc=13, hop_length=512):
    """
    Extracts MFCCs from the audio signal.
    """
    try:
        # pyrefly: ignore [missing-import]
        import librosa
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        return mfccs
    except Exception as e:
        print(f"[AudioProcessor Warning] Librosa MFCC failed: {e}. Using scipy fallback...")
        
    # Basic MFCC fallback (simplified DCT of log mel spectrogram)
    # 1. Get fallback spectrogram
    norm_spec, _, _ = compute_mel_spectrogram(y, sr, n_mels=40, hop_length=hop_length)
    norm_spec = np.array(norm_spec)
    
    # 2. Simple Discrete Cosine Transform (DCT) along frequency dimension
    from scipy.fftpack import dct
    mfccs = dct(norm_spec, type=2, axis=0, norm='ortho')[:n_mfcc, :]
    return mfccs

def compute_spectral_features(y, sr):
    """
    Calculates features related to voice authenticity:
    - Spectral Centroid
    - Spectral Bandwidth
    - Zero Crossing Rate
    - Jitter (estimated via local pitch variations)
    - Shimmer (estimated via local peak amplitude variations)
    """
    features = {}
    
    # Zero crossing rate
    zero_crossings = np.nonzero(np.diff(y > 0))[0]
    zcr = len(zero_crossings) / len(y) if len(y) > 0 else 0
    features['zero_crossing_rate'] = float(zcr)
    
    # FFT for spectral analysis
    n = len(y)
    fft_vals = np.abs(np.fft.rfft(y))
    fft_freqs = np.fft.rfftfreq(n, 1/sr)
    
    # Normalize power spectrum
    power_spectrum = fft_vals ** 2
    sum_power = np.sum(power_spectrum)
    
    if sum_power > 0:
        normalized_power = power_spectrum / sum_power
        # Spectral Centroid
        centroid = np.sum(fft_freqs * normalized_power)
        # Spectral Bandwidth
        bandwidth = np.sqrt(np.sum(((fft_freqs - centroid) ** 2) * normalized_power))
    else:
        centroid = 0.0
        bandwidth = 0.0
        
    features['spectral_centroid'] = float(centroid)
    features['spectral_bandwidth'] = float(bandwidth)
    
    # Spectral Roll-off (85% energy)
    cumulative_power = np.cumsum(power_spectrum)
    if sum_power > 0:
        threshold = 0.85 * sum_power
        rolloff_idx = np.where(cumulative_power >= threshold)[0][0]
        rolloff = fft_freqs[rolloff_idx]
    else:
        rolloff = 0.0
    features['spectral_rolloff'] = float(rolloff)
    
    # Jitter & Shimmer estimation (pitch-based)
    # We use autocorrelation to estimate fundamental frequency contours (pitch)
    # Jitter: cycle-to-cycle variability of pitch
    # Shimmer: cycle-to-cycle variability of peak amplitude
    try:
        # Segment audio into 40ms overlapping frames to estimate local pitch
        frame_len = int(sr * 0.04) # 40ms
        hop_len = int(sr * 0.02)   # 20ms
        pitches = []
        amplitudes = []
        
        for start in range(0, len(y) - frame_len, hop_len):
            frame = y[start:start+frame_len]
            # Hanning window to prevent leakage
            frame = frame * np.hanning(len(frame))
            if np.max(np.abs(frame)) < 0.01:
                continue # silence
                
            # Autocorrelation
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            
            # Find pitch range: 80Hz to 400Hz (human voice range)
            min_lag = int(sr / 400)
            max_lag = int(sr / 80)
            
            if len(corr) > max_lag:
                lag_search = corr[min_lag:max_lag]
                if len(lag_search) > 0:
                    peak_lag = np.argmax(lag_search) + min_lag
                    pitch = sr / peak_lag
                    pitches.append(pitch)
                    amplitudes.append(np.max(np.abs(frame)))
                    
        if len(pitches) > 2:
            pitch_diffs = np.abs(np.diff(pitches))
            jitter = np.mean(pitch_diffs) / np.mean(pitches)
            
            amp_diffs = np.abs(np.diff(amplitudes))
            shimmer = np.mean(amp_diffs) / np.mean(amplitudes)
        else:
            jitter = 0.005 # Baseline natural values
            shimmer = 0.03
    except Exception:
        jitter = 0.005
        shimmer = 0.03
        
    features['jitter'] = float(jitter)
    features['shimmer'] = float(shimmer)
    
    return features
