import json
import os
import pyrebase

def reset_and_expand_cursor():
    config_path = "../ramanujan_client/firebase_config.json"
    
    if not os.path.exists(config_path):
        print(f"[!] Please run this from the scripts folder. Could not find {config_path}")
        return
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()
    db = firebase.database()
    
    print("[*] Authenticating...")
    user = auth.sign_in_anonymous()
    id_token = user['idToken']
    
    # ── Expanding the Mathematical Universe ──
    # We set a massive new domain for the distributed workers
    new_cursor_state = {
        "degree": 2,
        "chunk_width": 10,     
        "a_min": -100000,
        "a_max": 100000,
        "b_min": -100000,
        "b_max": 100000,
        "current_a_pos": -100000, 
        "current_b_pos": -100000  
    }
    
    print(f"[*] Overwriting V2 Dynamic Task Cursor on Firebase Realtime DB...")
    db.child("v2_dynamic_tasks").child("cursor").set(new_cursor_state, id_token)
    
    print("[+] Cursor successfully expanded to [-100, 100] grid! Your clients will now resume computing.")

if __name__ == "__main__":
    reset_and_expand_cursor()
