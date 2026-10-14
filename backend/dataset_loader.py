import os
import shutil

def load_real_dataset(base_dir, archive_dir=None):
    """
    Copies the real and fake audio files from the 'archive (1)' folder
    into the backend/data/train structure for the CNN model to use.
    
    Returns True if the dataset was successfully loaded, False otherwise.
    """
    if archive_dir is None:
        # Assuming archive (1) is in the root directory (parent of backend)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        archive_dir = os.path.join(root_dir, "archive (1)")
        
    if not os.path.exists(archive_dir):
        print(f"[DatasetLoader] Archive directory not found at: {archive_dir}")
        return False
        
    train_dir = os.path.join(base_dir, "train")
    human_dir = os.path.join(train_dir, "human")
    ai_dir = os.path.join(train_dir, "ai")
    
    # Create target directories
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)
    
    archive_real = os.path.join(archive_dir, "Real", "Real")
    archive_fake = os.path.join(archive_dir, "Fake", "Fake")
    
    if not os.path.exists(archive_real) or not os.path.exists(archive_fake):
        print(f"[DatasetLoader] Expected subdirectories not found in {archive_dir}.")
        return False
        
    print(f"[DatasetLoader] Copying dataset from {archive_dir} to {train_dir}...")
    
    # Copy Human / Real files
    copied_human = 0
    for filename in os.listdir(archive_real):
        if filename.endswith(".wav"):
            src = os.path.join(archive_real, filename)
            dst = os.path.join(human_dir, f"real_{copied_human}.wav")
            shutil.copy2(src, dst)
            copied_human += 1
            
    # Copy AI / Fake files
    copied_ai = 0
    for filename in os.listdir(archive_fake):
        if filename.endswith(".wav"):
            src = os.path.join(archive_fake, filename)
            dst = os.path.join(ai_dir, f"fake_{copied_ai}.wav")
            shutil.copy2(src, dst)
            copied_ai += 1
            
    print(f"[DatasetLoader] Successfully loaded {copied_human} real audio files and {copied_ai} fake audio files.")
    return True
