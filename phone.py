import speech_recognition as sr  # pip install speech_recognition#
import pyttsx3 # pip install pyttsx3 #
import openai 
import requests # pip install requests #
from dotenv import load_dotenv  # pip install dotenv #
import os #pip install os#

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") # Here enter Your OpenAI key(API) #

# Initialize Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        # Use OpenAI Whisper for high-accuracy transcription
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())
        transcript = openai.Audio.transcribe("whisper-1", open("temp_audio.wav", "rb"))
        return transcript["text"]
    except Exception as e:
        print("Error:", e)
        return ""

def handle_voice_command():
    while True:
        command = listen().lower()
        if "start tracking" in command:
            response = requests.post("http://localhost:5000/start_tracking", json={"user_id": "123"})
            speak("Tracking started. Share this link: " + response.json()["tracking_link"])
        elif "stop tracking" in command:
            requests.post("http://localhost:5000/stop_tracking", json={"link_id": "123"})
            speak("Tracking stopped.")
        elif "exit" in command:
            speak("Goodbye!")
            break

if __name__ == "__main__":
    speak("Voice control activated. Say 'start tracking' or 'stop tracking'.")
    handle_voice_command()
