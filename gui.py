import tkinter as tk
from voice_ai import speak, handle_voice_command
from threading import Thread

def start_voice_control():
    Thread(target=handle_voice_command).start()

root = tk.Tk()
root.title("AI Voice Tracker")

tk.Button(root, text="🎤 Start Voice Control", command=start_voice_control).pack(pady=20)
root.mainloop()
