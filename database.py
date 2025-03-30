from pysqlcipher3 import dbapi2 as sqlite
import os

# Generate a secure key (store this safely!)
DB_KEY = os.getenv("DB_KEY", "default-secure-key-32chars-123456789012")

def init_db():
    conn = sqlite.connect('encrypted.db')
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA key='{DB_KEY}'")
    cursor.execute("PRAGMA cipher_use_hmac = ON")  # Enable HMAC
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            encrypted_data TEXT NOT NULL,
            iv TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn
