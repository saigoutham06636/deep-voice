import os
import re

# We will import speech_recognition inside function for robustness
# to avoid failing if not fully loaded.

def transcribe_audio(file_path):
    """
    Transcribes the given WAV file using SpeechRecognition Google Web API.
    Has fallback mock transcripts for simulator presets.
    """
    # Check if this matches a simulated filename to return preset transcripts
    filename = os.path.basename(file_path).lower()
    
    # Preset scenarios for the simulator
    presets = {
        "bank_scam": "This is Officer Jackson from the bank fraud division. We have detected unauthorized transactions on your account. You must transfer your remaining balance to our secure escrow vault immediately to prevent loss. Please confirm your account details and card PIN.",
        "grandchild_emergency": "Grandpa? It's me. I'm in trouble. I got into a car accident and the police arrested me. I need ten thousand dollars for bail right now. Please wire the money via western union immediately, don't tell mom and dad, please!",
        "tech_support": "Hello this is technical support. We detected a major malware infection on your Windows computer. Your passwords and banking information are at risk. Please install our remote assistance software immediately so I can secure your machine.",
        "normal_call": "Hey there! Just calling to check in on our dinner plans for tomorrow night. I was thinking we could try that new Italian place downtown around seven. Let me know if that works for you!",
        "doctor_appointment": "Hello, this is a reminder from the medical center. You have an appointment scheduled with Dr. Smith tomorrow at ten AM. Please arrive fifteen minutes early and bring your ID and insurance card. Thank you."
    }
    
    # Check if the filename itself corresponds to a preset simulation
    for key, text in presets.items():
        if key in filename:
            return text
            
    # Try actual speech recognition
    try:
        # pyrefly: ignore [missing-import]
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"[PhishingDetector Warning] Google Speech recognition failed or offline: {e}")
        # Default fallback transcript based on classification if actual audio fails
        return ""

def analyze_transcript(text):
    """
    Scans the transcribed text for cybersecurity threat markers.
    Categories:
    - Financial Urgency (monetary loss, transfer requests)
    - Identity & Credentials (passwords, OTPs, SSN)
    - Urgency & Fear (action required immediately, arrest, frozen account)
    - Authority Pressure (impersonating bank officials, police, support managers)
    """
    if not text:
        return {
            "risk_score": 0,
            "threat_level": "Safe (No speech detected)",
            "categories": {},
            "matched_keywords": [],
            "advice": "No spoken text detected or transcription unavailable. Ensure mic input is clear."
        }
        
    text_lower = text.lower()
    
    # Cyber security keywords and weightings
    threat_rules = {
        "Financial Requests": {
            "keywords": ["wire transfer", "transfer money", "transfer your balance", "send money", "western union", 
                         "gift card", "crypto", "bitcoin", "credit card", "bank account", "routing number", 
                         "escrow vault", "wire the money", "withdrawal"],
            "weight": 30
        },
        "Identity & Credentials": {
            "keywords": ["social security", "ssn", "verify password", "password", "verification code", 
                         "one time password", "otp", "pin", "login credentials", "security question", 
                         "confirm your account", "card pin"],
            "weight": 30
        },
        "Urgency & Fear": {
            "keywords": ["immediately", "urgent", "right now", "compromised", "police", "arrest", 
                         "lawsuit", "unauthorized transaction", "suspended", "frozen", "locked out", 
                         "prevent loss", "in trouble", "jail", "within 24 hours"],
            "weight": 20
        },
        "Authority Pressure": {
            "keywords": ["government", "irs", "fbi", "police department", "support manager", 
                         "security officer", "fraud division", "fraud department", "official support", 
                         "technical support", "officer jackson"],
            "weight": 20
        }
    }
    
    category_scores = {}
    matched_keywords = []
    total_score = 0.0
    
    for category, rule in threat_rules.items():
        matches = []
        for kw in rule["keywords"]:
            # Word boundary regex search to prevent partial word matches (e.g. "pin" in "pinpoint")
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(kw)
                matched_keywords.append(kw)
                
        # Calculate ratio of matched keywords in category (cap at 1.0)
        match_ratio = min(len(matches) / 2.0, 1.0) # 2 matches maxes out category score
        category_score = match_ratio * rule["weight"]
        category_scores[category] = {
            "matches": matches,
            "score": float(category_score),
            "max_score": rule["weight"]
        }
        total_score += category_score
        
    total_score = float(min(100.0, total_score))
    
    # Categorize threat levels
    if total_score <= 20:
        threat_level = "Low Risk (Safe Call)"
        advice = "The system detected normal conversational markers. No classic social engineering indicators found. It is likely safe, but remain vigilant."
    elif total_score <= 55:
        threat_level = "Medium Risk (Suspicious)"
        advice = "Caution! This call contains vocabulary related to security, authority, or financial transactions. Verify the speaker's identity using an independent channel before sharing details."
    else:
        threat_level = "High Risk (Critical Phishing Threat)"
        advice = "DANGER! High probability of a Voice Phishing (Vishing) scam! The caller is using high-pressure tactics, posing as an authority figure, or demanding financial actions/credentials. HANG UP IMMEDIATELY and contact the official institution directly."
        
    return {
        "risk_score": total_score,
        "threat_level": threat_level,
        "categories": category_scores,
        "matched_keywords": list(set(matched_keywords)),
        "advice": advice
    }
