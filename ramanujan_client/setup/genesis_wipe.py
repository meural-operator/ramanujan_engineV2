import firebase_admin
from firebase_admin import credentials, db
import os

print("==================================================")
print("     Ramanujan@Home - Database Genesis Wipe       ")
print("==================================================")

try:
    # Authorize with your master cloud key
    cred_opt1 = "C:\\Users\\DIAT\\ashish\\ramanujan_machine\\RamanujanMachine\\ramanujan-engine-firebase-adminsdk-h18yo-6dc62d8544.json"
    cred_opt2 = "ramanujan-engine-firebase-adminsdk.json"
    
    cred_path = cred_opt1 if os.path.exists(cred_opt1) else cred_opt2
    
    print(f"[*] Authenticating with Master Service SDK Key...")
    cred = credentials.Certificate(cred_path)
    # Ensure correct Region Database URL
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://ramanujan-engine-default-rtdb.firebaseio.com/"
    })

    print("[-] EXECUTING ROOT DELETION PROTOCOL...")
    
    # Target absolute database root and destruct
    root_ref = db.reference("/")
    root_ref.delete()
    
    print("[+] CLOUD DATABASE HAS BEEN MATHEMATICALLY ANNIHILATED.")
    print("[+] Users, Results, and Queues are permanently purged.")
    print("[+] You are cleared to launch `global_seeder.py` to establish the initial Genesis Queues!")

except Exception as e:
    print(f"[!] ERRROR Executing Genesis Wipe: {e}")
