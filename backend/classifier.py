import os
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
import backend.audio_processor as ap

# Target shape for MFCC input: 13 bands by 94 time frames (approx. 3 seconds at 16kHz, hop=512)
TARGET_MFCC_SHAPE = (13, 94)

class DeepFakeVoiceCRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) designed to classify Mel-Frequency 
    Cepstral Coefficients (MFCC) sequences of human vs AI-synthesized speech.
    """
    def __init__(self):
        super(DeepFakeVoiceCRNN, self).__init__()
        # Input shape: (1, 13, 94) -> Channel=1 (grayscale MFCC feature map)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dimensions after Pool 1 (13x94 -> 6x47)
        # Dimensions after Pool 2 (6x47 -> 3x23)
        # We treat the width (23) as time sequence, and height*channels (3*32=96) as features
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 2) # Outputs LogSoftmax of 2 classes: 0 = Human, 1 = AI

    def forward(self, x):
        # Layer 1
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.pool(x)
        
        # Layer 2
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.pool(x)
        
        # Reshape for LSTM: (batch, channels, height, width) -> (batch, width, channels * height)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2) # (batch, width, channels, height)
        x = x.contiguous().view(batch_size, width, channels * height) # (batch, 23, 96)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        x = lstm_out[:, -1, :] # (batch, 64)
        
        # Fully Connected Layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class AudioMFCCDataset(Dataset):
    """
    PyTorch Dataset wrapper for pre-processed audio waves with optional feature augmentation.
    """
    def __init__(self, data_list, labels, augment=False):
        self.data = data_list
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].copy()
        if self.augment:
            # Add subtle random Gaussian noise to the MFCC map to prevent model over-reliance on clean shapes
            noise = np.random.normal(0, 0.02, x.shape)
            x = x + noise
            # Random scale to mimic volume changes
            x = x * np.random.uniform(0.95, 1.05)
            
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0) # shape (1, 13, 94)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def pad_or_truncate_mfcc(mfcc, target_shape=TARGET_MFCC_SHAPE):
    """
    Pads or truncates MFCC matrix to a fixed target shape.
    """
    n_mfcc, n_frames = mfcc.shape
    target_mfcc, target_frames = target_shape
    
    # Pad or truncate MFCC coefficients
    if n_mfcc > target_mfcc:
        mfcc = mfcc[:target_mfcc, :]
    elif n_mfcc < target_mfcc:
        mfcc = np.pad(mfcc, ((0, target_mfcc - n_mfcc), (0, 0)), mode='constant')
        
    # Pad or truncate frames
    if n_frames > target_frames:
        mfcc = mfcc[:, :target_frames]
    elif n_frames < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - n_frames)), mode='constant')
        
    return mfcc

def extract_dataset_features(data_dir):
    """
    Loads all WAV files in human/ and ai/ folders, extracts MFCCs, and aligns shapes.
    """
    data = []
    labels = []
    
    categories = {"human": 0, "ai": 1}
    for category, label in categories.items():
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            continue
            
        print(f"Loading files from: {cat_dir}")
        for file_name in os.listdir(cat_dir):
            if file_name.endswith(".wav"):
                file_path = os.path.join(cat_dir, file_name)
                try:
                    y, sr = ap.load_audio(file_path)
                    mfcc = ap.extract_mfcc(y, sr)
                    mfcc_norm = pad_or_truncate_mfcc(mfcc)
                    
                    # Normalize MFCC values between -1 and 1
                    max_val = np.max(np.abs(mfcc_norm))
                    if max_val > 0:
                        mfcc_norm = mfcc_norm / max_val
                        
                    data.append(mfcc_norm)
                    labels.append(label)
                except Exception as e:
                    print(f"Skipping {file_name}: {e}")
                    
    return np.array(data), np.array(labels)

def train_network(data_dir="backend/data/train", model_save_path="backend/data/voice_cnn.pth", epochs=15, callback=None):
    """
    Loads dataset, trains the CNN, and saves weights.
    Sends progress reports via the callback function.
    """
    if callback:
        callback("INFO: Starting feature extraction on training dataset...")
    else:
        print("Starting feature extraction...")
        
    X, Y = extract_dataset_features(data_dir)
    if len(X) == 0:
        err_msg = "ERROR: No audio samples found. Run mock_data_generator first."
        if callback: callback(err_msg)
        raise ValueError(err_msg)
        
    # Split into train/validation indices
    dataset_size = len(X)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    # 80/20 train/test split
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_dataset = AudioMFCCDataset(X[train_indices], Y[train_indices], augment=True)
    val_dataset = AudioMFCCDataset(X[val_indices], Y[val_indices], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    model = DeepFakeVoiceCRNN()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = (correct / total) * 100
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
        val_loss = val_loss / max(1, len(val_loader.dataset))
        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 100.0
        
        epoch_summary = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%"
        if callback:
            callback(epoch_summary)
        else:
            print(epoch_summary)
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    success_msg = f"SUCCESS: Model saved successfully to '{model_save_path}'"
    if callback:
        callback(success_msg)
    else:
        print(success_msg)
        
    return history

def heuristic_predict(y, sr):
    """
    DSP Rule-Based Fallback Classifier.
    Analyzes specific physical metrics that vary between organic speech and vocoder models:
    - Jitter/Shimmer levels (organic vocal instability vs machine uniformity).
    - Spectral roll-off & centroid (metallic ringing artifacts in high frequencies).
    - ZCR irregularities.
    """
    feats = ap.compute_spectral_features(y, sr)
    
    jitter = feats['jitter']
    shimmer = feats['shimmer']
    centroid = feats['spectral_centroid']
    rolloff = feats['spectral_rolloff']
    
    score = 0.0
    # 1. AI voices are often way too stable (very low jitter/shimmer) OR have mathematical step patterns.
    # If jitter is extremely low (< 0.002), flag it.
    if jitter < 0.0025:
        score += 0.35
        
    # 2. High-frequency synthesis artifacts: AI vocoders leave spectral residues in high bands.
    # High spectral centroids or unusually high energy above 6kHz (rolloff > 5000Hz).
    if rolloff > 5500:
        score += 0.30
    if centroid > 2200:
        score += 0.20
        
    # 3. Low amplitude dynamics
    if shimmer < 0.015:
        score += 0.15
        
    # Normalize score between 0.15 and 0.95
    score = max(0.1, min(0.95, score))
    
    # Let's say if score > 0.45, it is classified as AI
    is_ai = score > 0.45
    ai_prob = score if is_ai else (1 - score)
    
    label = "AI Generated" if is_ai else "Human"
    
    return label, float(ai_prob), "DSP Heuristic Analysis"

def predict_voice(file_path, model_path="backend/data/voice_cnn.pth"):
    """
    Predicts if the audio file is "Human" or "AI Generated" using the CNN model.
    Falls back to Heuristics if model weights are missing.
    """
    try:
        y, sr = ap.load_audio(file_path)
    except Exception as e:
        return "Unknown", 0.0, f"Error loading audio: {str(e)}"
        
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[Classifier Info] Model weights not found. Using Heuristic fallback.")
        return heuristic_predict(y, sr)
        
    try:
        # Load model and weights
        model = DeepFakeVoiceCRNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Preprocess audio
        mfcc = ap.extract_mfcc(y, sr)
        mfcc_norm = pad_or_truncate_mfcc(mfcc)
        
        max_val = np.max(np.abs(mfcc_norm))
        if max_val > 0:
            mfcc_norm = mfcc_norm / max_val
            
        # Convert to tensor shape (1, 1, 13, 94)
        input_tensor = torch.tensor(mfcc_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.exp(output).squeeze().numpy() # [prob_human, prob_ai]
            
        is_ai = np.argmax(prob) == 1
        confidence = float(prob[1]) if is_ai else float(prob[0])
        label = "AI Generated" if is_ai else "Human"
        
        return label, confidence, "PyTorch CNN Model"
        
    except Exception as e:
        print(f"[Classifier Warning] PyTorch inference failed: {e}. Using Heuristic fallback.")
        return heuristic_predict(y, sr)
