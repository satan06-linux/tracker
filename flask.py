@app.route('/start_tracking', methods=['POST'])
@require_oauth  # Custom decorator to check OAuth2 login
def start_tracking():
    user_id = session.get("user")["id"]
    key = get_random_bytes(32)
    link_id = secrets.token_urlsafe(16)
    
    # Store encrypted metadata in DB
    db = init_db()
    cursor = db.cursor()
    cursor.execute(f"PRAGMA key='{DB_KEY}'")
    cursor.execute(
        "INSERT INTO locations (user_id, encrypted_data, iv) VALUES (?, ?, ?)",
        (user_id, encrypt_data(json.dumps({"key": key.hex()}), get_random_bytes(16).hex())
    )
    db.commit()
    
    return jsonify({ 
        "tracking_link": f"https://yourserver.com/track/{link_id}",
        "jwt": create_jwt(user_id)  # For future authenticated requests
    })
