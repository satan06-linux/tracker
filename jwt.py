from jose import jwt, JWTError
from datetime import datetime, timedelta

SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"

def create_jwt(user_id: str) -> str:
    expires = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expires}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except JWTError:
        return None
