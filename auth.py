from flask import Flask, redirect, session
from authlib.integrations.flask_client import OAuth
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# OAuth2 Config (Google Example)
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='YOUR_GOOGLE_CLIENT_ID',
    client_secret='YOUR_GOOGLE_CLIENT_SECRET',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'email profile'},
)

@app.route('/login')
def login():
    redirect_uri = 'https://yourserver.com/authorize'
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    token = google.authorize_access_token()
    user_info = google.get('userinfo').json()
    session['user'] = user_info  # Store user in encrypted session
    return redirect('/dashboard')
