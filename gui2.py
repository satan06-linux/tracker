import webbrowser
from flask import Flask

app = Flask(__name__)

@app.route('/login_success')
def login_success():
    return "Login successful! Return to the app."

def open_browser_for_login():
    webbrowser.open("http://localhost:5000/login")

# After login, store JWT in GUI for API calls
