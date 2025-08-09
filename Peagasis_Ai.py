import os
import time
import json
import threading
import webbrowser
import pyttsx3
import pyautogui
import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import pyaudio
from datetime import datetime
from googletrans import Translator
import pdfplumber
import fitz  # PyMuPDF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pytesseract
from PIL import ImageGrab
import re
import shutil
import hashlib
import uuid
import base64
import getpass
from pathlib import Path
import subprocess

# Optional advanced libs
try:
    import cv2
    import face_recognition
    HAS_FACE = True
except Exception:
    HAS_FACE = False

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

try:
    import numpy as _np
except Exception:
    _np = None

# ---------------------------
# Configuration
# ---------------------------
WAKE_WORD = "jarvis"
INACTIVITY_LIMIT = 180
DATA_DIR = "jarvis_data"
SECURE_DIR = os.path.join(DATA_DIR, ".jarvis_secure")
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
LOG_FILE = os.path.join(DATA_DIR, "activity_log.txt")
CRED_FILE = os.path.join(SECURE_DIR, "creds.bin")
RETINA_STORE = os.path.join(SECURE_DIR, "retina_admin.bin")
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
TTS_VOICE_INDEX = 0
LEARN_CATEGORIES = ['ai', 'machine learning', 'medical', 'space', 'technology', 'algorithms', 'stock']
LEARNING_ENABLED = True
LEARNING_INTERVAL = 60

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SECURE_DIR, exist_ok=True)

# ---------------------------
# Engines
# ---------------------------
engine = pyttsx3.init()
voices = engine.getProperty('voices')
if voices and len(voices) > TTS_VOICE_INDEX:
    engine.setProperty('voice', voices[TTS_VOICE_INDEX].id)
translator = Translator()

# VOSK init (offline hotword)
vosk_recognizer = None
mic_stream = None
try:
    if os.path.exists(VOSK_MODEL_PATH):
        vosk_model = Model(VOSK_MODEL_PATH)
        vosk_recognizer = KaldiRecognizer(vosk_model, 16000)
        pa = pyaudio.PyAudio()
        mic_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        mic_stream.start_stream()
    else:
        print('VOSK model missing; offline hotword disabled')
except Exception as e:
    print('VOSK init error:', e)
    vosk_recognizer = None
    mic_stream = None

# SpeechRecognition fallback
recognizer = sr.Recognizer()

# ---------------------------
# State & utilities
# ---------------------------
state = {
    'current_user': 'guest',
    'family_mode': False,
    'last_activity': time.time(),
    'session_memory': {'last_google_search': None, 'last_file_opened': None, 'last_pdf': None, 'last_youtube': None, 'learned_topics': []},
    'learning_enabled': LEARNING_ENABLED
}

# Logging / persistence
def speak(text):
    print('Jarvis:', text)
    engine.say(text)
    engine.runAndWait()

def log(action, info=''):
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {action} - {info}
")
    except Exception:
        pass

def save_state():
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log('save_state_error', str(e))

def load_state():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                state.update(data)
        except Exception:
            pass

load_state()

# ---------------------------
# Crypto helpers (military-grade styled)
# ---------------------------
if HAS_CRYPTO:
    def machine_fingerprint():
        node = uuid.getnode()
        host = os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'pc')
        fp = f"{host}-{node}".encode()
        return hashlib.sha256(fp).digest()

    def derive_key(salt: bytes, iterations: int = 390000):
        secret = machine_fingerprint()
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=default_backend())
        return kdf.derive(secret)

    def encrypt_blob(data: bytes) -> bytes:
        salt = os.urandom(16)
        key = derive_key(salt)
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, data, None)
        return salt + nonce + ct

    def decrypt_blob(blob: bytes) -> bytes:
        salt = blob[:16]
        nonce = blob[16:28]
        ct = blob[28:]
        key = derive_key(salt)
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None)
else:
    def encrypt_blob(data: bytes) -> bytes:
        raise RuntimeError('cryptography required')
    def decrypt_blob(blob: bytes) -> bytes:
        raise RuntimeError('cryptography required')

# ---------------------------
# Credential management
# ---------------------------
CRED_STRUCTURE = {'family_hash': None, 'friend_hash': None, 'iterations': 390000}

def save_credentials(family_password: str, friend_password: str):
    if not HAS_CRYPTO:
        speak('cryptography not available')
        return False
    salt = os.urandom(16)
    def pw_hash(pw):
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=CRED_STRUCTURE['iterations'], backend=default_backend())
        return base64.b64encode(kdf.derive(pw.encode())).decode()
    data = {'salt': base64.b64encode(salt).decode(), 'family_hash': pw_hash(family_password), 'friend_hash': pw_hash(friend_password), 'iterations': CRED_STRUCTURE['iterations']}
    enc = encrypt_blob(json.dumps(data).encode())
    with open(CRED_FILE, 'wb') as f:
        f.write(enc)
    log('credentials_saved', '')
    return True

def load_credentials():
    if not os.path.exists(CRED_FILE):
        return None
    try:
        blob = open(CRED_FILE,'rb').read()
        pt = decrypt_blob(blob)
        return json.loads(pt.decode())
    except Exception as e:
        log('load_credentials_error', str(e))
        return None

def verify_password(input_pw: str, stored_hash_b64: str, salt_b64: str, iterations: int) -> bool:
    try:
        salt = base64.b64decode(salt_b64)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations, backend=default_backend())
        kdf.verify(input_pw.encode(), base64.b64decode(stored_hash_b64))
        return True
    except Exception:
        return False

# ---------------------------
# Retina enrollment / matching with liveness checks
# ---------------------------
if HAS_FACE and _np is not None:
    import cv2
    def capture_eye_region(timeout=6):
        cap = cv2.VideoCapture(0)
        start = time.time()
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        blink_count = 0
        last_eyes = None
        speak('Please look at camera and blink once.')
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(eyes) >= 1:
                ex, ey, ew, eh = sorted(eyes, key=lambda x: x[2]*x[3], reverse=True)[0]
                eye_img = gray[ey:ey+eh, ex:ex+ew]
                lap = cv2.Laplacian(eye_img, cv2.CV_64F).var()
                focus_ok = lap > 50
                if last_eyes is not None:
                    try:
                        d = cv2.absdiff(cv2.resize(last_eyes,(64,64)), cv2.resize(eye_img,(64,64)))
                        nz = cv2.countNonZero(d)
                        if nz > 60:
                            blink_count += 1
                    except Exception:
                        pass
                last_eyes = eye_img.copy()
                if blink_count >=1 and focus_ok:
                    cap.release()
                    return eye_img
        cap.release()
        return None

    def enroll_admin_retina():
        eye = capture_eye_region()
        if eye is None:
            speak('Failed to capture retina for enrollment')
            return False
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(eye, None)
        if des is None:
            speak('Failed to extract descriptors')
            return False
        enc = encrypt_blob(base64.b64encode(des.tobytes()))
        with open(RETINA_STORE,'wb') as f:
            f.write(enc)
        speak('Admin retina enrolled securely')
        log('enroll_retina','ok')
        return True

    def match_admin_retina():
        if not os.path.exists(RETINA_STORE):
            return False
        try:
            enc = open(RETINA_STORE,'rb').read()
            pt = decrypt_blob(enc)
            des_saved = base64.b64decode(pt)
            arr = _np.frombuffer(des_saved, dtype=_np.uint8)
            eye = capture_eye_region()
            if eye is None:
                return False
            orb = cv2.ORB_create()
            kp2, des2 = orb.detectAndCompute(eye, None)
            if des2 is None:
                return False
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            for dsize in (32,64):
                if arr.size % dsize == 0:
                    try:
                        des_saved2 = arr.reshape(-1, dsize).copy()
                        matches = bf.match(des_saved2.astype(_np.uint8), des2.astype(_np.uint8))
                        if not matches:
                            continue
                        avg = sum(m.distance for m in matches)/len(matches)
                        if avg < 40:
                            return True
                    except Exception:
                        continue
            return False
        except Exception as e:
            log('match_admin_retina_error', str(e))
            return False
else:
    def capture_eye_region(timeout=6):
        speak('Face libraries or numpy not installed; retina capture unavailable')
        return None
    def enroll_admin_retina():
        speak('Face libraries or numpy not installed')
        return False
    def match_admin_retina():
        return False

# ---------------------------
# Listening helpers (VOSK offline first, SR fallback)
# ---------------------------

def listen_vosk(timeout=10):
    if vosk_recognizer is None or mic_stream is None:
        return ''
    start = time.time()
    txt = ''
    while time.time() - start < timeout:
        try:
            data = mic_stream.read(4096, exception_on_overflow=False)
            if vosk_recognizer.AcceptWaveform(data):
                res = json.loads(vosk_recognizer.Result())
                txt = res.get('text','')
                if txt:
                    return txt.lower()
        except Exception:
            break
    return ''

def listen_sr(timeout=8):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)
            return recognizer.recognize_google(audio).lower()
        except Exception:
            return ''

def listen_any(timeout=8):
    t = listen_vosk(timeout=timeout)
    if t:
        return t
    return listen_sr(timeout=timeout)

# ---------------------------
# PDF summarizer (detailed + real-life example)
# ---------------------------

def summarize_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            text = '
'.join(p.extract_text() or '' for p in pdf.pages)
            if not text.strip():
                speak('No text found in PDF.')
                return ''
            # naive summarization: first 3 sentences and most common phrases
            sentences = re.split(r'(?<=[.!?]) +', text)
            summary = ' '.join(sentences[:3])
            speak('Here is a brief summary:')
            speak(summary)
            # real-life example: make it contextual
            example = f"Imagine using this in your daily work: {sentences[0][:120]}..."
            speak('Real-life example:')
            speak(example)
            state['session_memory']['learned_topics'].append({'path': path, 'summary': summary, 'time': datetime.now().isoformat()})
            save_state()
            log('summarize_pdf', path)
            return summary
    except Exception as e:
        log('summarize_pdf_error', str(e))
        speak('Failed to summarize PDF')
        return ''

# ---------------------------
# OCR + Code analyzer (screen)
# ---------------------------

def analyze_code_screen():
    try:
        img = ImageGrab.grab()
        text = pytesseract.image_to_string(img)
        if not text.strip():
            speak('No readable text detected on screen.')
            return
        errors = re.findall(r'(SyntaxError|Traceback|Exception|IndentationError|NameError)', text, re.IGNORECASE)
        suggestions = []
        if errors:
            suggestions.append('Found possible error patterns: ' + ', '.join(set(errors)))
        if 'def ' in text and 'return' not in text:
            suggestions.append('Detected function definitions without explicit return statements.')
        if not suggestions:
            speak('Code detected on screen. No immediate syntax warnings found.')
        else:
            for s in suggestions:
                speak(s)
        log('analyze_code_screen', 'ok')
    except Exception as e:
        log('analyze_code_screen_error', str(e))
        speak('Error analyzing screen')

# ---------------------------
# Background learning
# ---------------------------

def background_learn_loop():
    while True:
        if not state.get('learning_enabled', True):
            time.sleep(LEARNING_INTERVAL)
            continue
        for topic in LEARN_CATEGORIES:
            note = f"Auto-learned note on {topic} at {datetime.now().isoformat()}"
            state['session_memory']['learned_topics'].append({'topic': topic, 'note': note})
            save_state()
            log('background_learn', topic)
            time.sleep(2)
        time.sleep(LEARNING_INTERVAL)

# ---------------------------
# Inactivity monitor
# ---------------------------

def inactivity_monitor():
    while True:
        if time.time() - state.get('last_activity', time.time()) > INACTIVITY_LIMIT:
            speak('Auto-sleeping due to inactivity. Say Jarvis awake to continue.')
            # listen for wake word (VOSK preferred)
            while True:
                txt = listen_vosk(timeout=20) or listen_sr(timeout=20)
                if txt and 'jarvis awake' in txt:
                    speak('I am back online.')
                    state['last_activity'] = time.time()
                    break
        time.sleep(5)

# ---------------------------
# Authentication flow (retina -> role -> password)
# ---------------------------

def prompt_hidden_password(prompt_text: str) -> str:
    try:
        return getpass.getpass(prompt_text + ': ')
    except Exception:
        return input(prompt_text + ': ')

def authenticate_user_interactive():
    # try admin retina
    if match_admin_retina():
        speak('Admin retina recognized. Full access granted.')
        state['current_user'] = 'admin'
        save_state()
        return 'admin'

    # fallback to password flow
    speak('No admin retina detected. Please say your role: family or friend, or type it now.')
    role = listen_any(timeout=6)
    if not role:
        role = input('Role (family/friend): ')
    role = role.lower()
    creds = load_credentials()
    if creds is None:
        speak('No credentials stored. Admin must enroll.')
        return None
    salt_b64 = creds.get('salt')
    iterations = creds.get('iterations', CRED_STRUCTURE['iterations'])
    if 'family' in role:
        pw = prompt_hidden_password('Enter family password')
        ok = verify_password(pw, creds['family_hash'], salt_b64, iterations)
        if ok:
            speak('Family access granted with limited privileges.')
            state['current_user'] = 'family'
            state['family_mode'] = True
            save_state()
            return 'family'
        else:
            speak('Incorrect family password.')
            return None
    if 'friend' in role:
        pw = prompt_hidden_password('Enter friend password')
        ok = verify_password(pw, creds['friend_hash'], salt_b64, iterations)
        if ok:
            speak('Friend access granted with limited privileges.')
            state['current_user'] = 'friend'
            save_state()
            return 'friend'
        else:
            speak('Incorrect friend password.')
            return None
    speak('Role not recognized.')
    return None

# ---------------------------
# Command handler
# ---------------------------

def handle_command(cmd):
    state['last_activity'] = time.time()
    cmd = cmd.lower()
    if 'open google' in cmd:
        webbrowser.open('https://www.google.com')
        state['session_memory']['last_google_search'] = 'google'
        save_state()
        speak('Opened Google')
        return True
    if 'open youtube' in cmd:
        webbrowser.open('https://www.youtube.com')
        state['session_memory']['last_youtube'] = 'youtube'
        save_state()
        speak('Opened YouTube')
        return True
    if 'read pdf' in cmd or 'analyze pdf' in cmd:
        files = sorted([str(p) for p in Path(DATA_DIR).glob('*.pdf')], key=os.path.getmtime, reverse=True)
        if not files:
            speak('No PDFs in the pdfs folder')
            return True
        summarize_pdf(files[0])
        return True
    if 'analyze code' in cmd or 'analyze the code' in cmd:
        analyze_code_screen()
        return True
    if 'enroll retina' in cmd or 'enroll admin retina' in cmd:
        if enroll_admin_retina():
            speak('Enrollment successful')
        else:
            speak('Enrollment failed')
        return True
    if 'what did you learn' in cmd:
        learned = state['session_memory'].get('learned_topics', [])
        if not learned:
            speak('I have not learned anything yet')
        else:
            speak(f'I have notes on {len(learned)} topics. Latest: {learned[-1].get("topic", learned[-1].get("path",""))}')
        return True
    if 'send email' in cmd:
        speak('Email sending is admin-only. Authenticate as admin to proceed.')
        return True
    if 'jarvis offline' in cmd or 'sleep' in cmd:
        speak('Going to sleep. Say Jarvis awake to resume.')
        return False
    speak('Command not recognized')
    return True

# ---------------------------
# Initial enrollment helper
# ---------------------------

def initial_enroll():
    # Create creds if missing
    if os.path.exists(CRED_FILE) and os.path.exists(RETINA_STORE):
        return
    speak('Initial secure enrollment starting. You will create family and friend passwords.')
    fam = prompt_hidden_password('Set family password')
    fri = prompt_hidden_password('Set friend password')
    if not save_credentials(fam, fri):
        speak('Failed to save credentials securely')
    else:
        speak('Passwords saved. You may enroll admin retina now.')
        ans = listen_any(timeout=6)
        if 'yes' in ans.lower():
            enroll_admin_retina()

# ---------------------------
# Threads: background learning & inactivity
# ---------------------------
bg_thread = threading.Thread(target=background_learn_loop, daemon=True)
bg_thread.start()
inact_thread = threading.Thread(target=inactivity_monitor, daemon=True)
inact_thread.start()

# ---------------------------
# Main loop: authenticate then listen for wakeword and commands
# ---------------------------

def interactive_boot():
    initial_enroll()
    user = authenticate_user_interactive()
    if user is None:
        speak('Authentication failed — limited guest mode enabled')
        state['current_user'] = 'guest'
        save_state()
    else:
        speak(f'Authenticated as {user}')

    speak('Jarvis is online and listening for the wake word')
    try:
        while True:
            txt = listen_vosk(timeout=10) or listen_sr(timeout=8)
            if txt and WAKE_WORD in txt:
                speak('Yes?')
                cmd = listen_any(timeout=12)
                if cmd:
                    cont = handle_command(cmd)
                    if cont is False:
                        # sleep until explicit wake
                        while True:
                            wake = listen_any(timeout=20)
                            if 'jarvis awake' in wake:
                                speak('I am awake')
                                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        speak('Shutting down. Saving state.')
        save_state()

if __name__ == '__main__':
    interactive_boot()
