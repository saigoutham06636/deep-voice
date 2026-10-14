# Audio Classifier Architecture Upgrade: CNN to CRNN

This document provides a detailed technical explanation of the recent changes made to the neural network architecture in `backend/classifier.py`. The model was upgraded from a standard **2D Convolutional Neural Network (CNN)** to a **Convolutional Recurrent Neural Network (CRNN)**.

## 1. Overview of the Change

Previously, the project used a standard 2D CNN (`DeepFakeVoiceCNN`) to classify audio. Audio was converted into a Mel-Frequency Cepstral Coefficients (MFCC) spectrogram, which is essentially an image, and the CNN scanned it for visual patterns.

While CNNs are excellent at finding local patterns (like the shape of a specific pitch or noise), they do not have a "memory" of time. To a standard CNN, the beginning of a sound and the end of a sound are just different pixels on an image. 

**The Upgrade:** We injected a Long Short-Term Memory (LSTM) layer into the network to create a **CRNN** (`DeepFakeVoiceCRNN`). This allows the network to extract raw features using convolutions, and then track how those features change sequentially over time.

## 2. Why is a CRNN better for Audio?

Audio is inherently time-series data. In voice classification (like detecting deepfakes or phishing):
* The sequence of phonemes matters.
* The natural hesitation or cadence in human speech is spread across time.
* Machine-generated audio often struggles with long-term temporal coherence (micro-jitters over several seconds).

By adding an LSTM, the network now processes the audio frame-by-frame, remembering past context to make a final decision, significantly improving its ability to differentiate complex human speech from AI.

## 3. Detailed Architecture Breakdown

Here is exactly how data flows through the new `DeepFakeVoiceCRNN`:

### Step 1: Input (MFCC Spectrogram)
* **Shape:** `[Batch_Size, 1, 13, 94]`
* The input is a grayscale image with 13 frequency bands and 94 time frames (roughly 3 seconds of audio).

### Step 2: Spatial Feature Extraction (CNN)
The data passes through two Convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool).
* These layers act as feature extractors. They find local patterns like formants, noise bands, and pitch edges.
* After the pooling layers, the spatial resolution is reduced.
* **Output Shape:** `[Batch_Size, 32 channels, 3 height (freqs), 23 width (time)]`

### Step 3: Reshaping for Time (The "Bridge")
In the old CNN, this `32x3x23` block was simply flattened into a 1D array of 2208 numbers, destroying all time information. 
In the new CRNN, we reshape the tensor so the LSTM understands it as a sequence of time steps:
* We treat the `23` width dimension as our **Sequence Length** (Time Steps).
* We combine the channels (32) and the height (3) into `96` **Features per time step**.
* **New Shape:** `[Batch_Size, 23, 96]`

### Step 4: Temporal Memory (LSTM)
* The reshaped tensor is fed into the `nn.LSTM` layer.
* The LSTM steps through the 23 time frames one by one. It looks at the 96 features at `t=0`, remembers the important parts, moves to `t=1`, and so on.
* Once it reaches the final time step (`t=23`), we extract its final "hidden state" which contains the summary of the entire audio clip.
* **Output Shape:** `[Batch_Size, 64]`

### Step 5: Classification (Fully Connected)
* The 64-dimensional summary from the LSTM is passed through standard Dense (`nn.Linear`) layers.
* **Final Output:** LogSoftmax probabilities for 2 classes (`Human` vs `AI Generated`).

## 4. Impact on Codebase

* **`backend/classifier.py`**: 
  * Class renamed to `DeepFakeVoiceCRNN`.
  * `__init__` method updated to include `self.lstm = nn.LSTM(input_size=96, hidden_size=64, num_layers=1, batch_first=True)`.
  * `forward` method updated with `permute` and `reshape` logic to bridge the CNN and LSTM.
* **Model Weights**: Because the architecture changed, the old `voice_cnn.pth` weights were incompatible. The model was automatically retrained from scratch on the existing dataset using the `/api/train` backend function, and the new weights were saved.
* **Usage**: No changes are required in the frontend (`app.js` or `index.html`) or the inference logic in `app.py`. The new model is a drop-in replacement that takes the same MFCC input and returns the same predictions, just with higher accuracy for temporal audio features!
