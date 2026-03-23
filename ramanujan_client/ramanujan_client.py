import time
import sys
import os
import sqlite3

from network.coordinator import ServerCoordinator
from engine_bridge.executor import RamanujanExecutor

def setup_sqlite():
    conn = sqlite3.connect("pending_discoveries.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pending_hits (
                    v2_bound_id TEXT, 
                    lhs_key TEXT, 
                    rhs_an TEXT, 
                    rhs_bn TEXT, 
                    client_id TEXT, 
                    ts REAL)''')
    conn.commit()
    return conn

def main():
    print("==================================================")
    print("           Ramanujan@Home - Compute Node          ")
    print("==================================================")
    
    config_path = "firebase_config.json"
    if not os.path.exists(config_path):
        print(f"[*] Missing {config_path}. Auto-generating default public access credentials...")
        import json
        default_config = {
            "apiKey": "AIzaSyCp1GpLSTGfjayQ-Yk0tW3dot1cOVpwxSE",
            "authDomain": "ramanujan-engine.firebaseapp.com",
            "databaseURL": "https://ramanujan-engine-default-rtdb.firebaseio.com",
            "projectId": "ramanujan-engine",
            "storageBucket": "ramanujan-engine.firebasestorage.app",
            "messagingSenderId": "662438808921",
            "appId": "1:662438808921:web:7d286acd925a08f1efd048",
            "measurementId": "G-LT414LS49S"
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=4)
        print("[+] Credentials generated! The client will now anonymously attach to the community node.")
    
    # Initialize the Firebase REST coordinator
    coordinator = ServerCoordinator(config_path=config_path)
    
    # 1. Authentication
    id_token = coordinator.authenticate_user()
    if not id_token:
        print("[!] Client shutting down due to authentication failure.")
        sys.exit(1)
        
    print(f"[*] Authenticated as UID: {coordinator.user.get('localId', coordinator.user.get('userId'))}")
        
    # Initialize the PyTorch Execution Bridge
    executor = RamanujanExecutor()
    
    print("[*] Engine V2 Initialized. Entering Community compute loop...\n")
    
    # Initialize local durable cache
    db_conn = setup_sqlite()
    db_cursor = db_conn.cursor()

    try:
        while True:
            # --- START CLOUD SYNC SWEEP ---
            # Automatically try to upload any stranded un-synced discoveries from previous loops/crashes
            db_cursor.execute("SELECT rowid, * FROM pending_hits")
            pending_rows = db_cursor.fetchall()
            
            if pending_rows:
                success_ids = coordinator.submit_results_bulk(pending_rows)
                if success_ids:
                    # Natively parameterized bulk deletion for SQLite
                    db_cursor.execute(f"DELETE FROM pending_hits WHERE rowid IN ({','.join('?'*len(success_ids))})", success_ids)
                    db_conn.commit()
                    print(f"[*] SQLite Cache Cleared. {len(pending_rows) - len(success_ids)} remaining.")
            # --- END CLOUD SYNC SWEEP ---

            # Step 1: Request a mutually-exclusive Tensor bounds matrix from the server
            work_unit = coordinator.request_work_unit()
            
            if work_unit is None:
                print("[-] Sleeping for 60 seconds before checking for new Work Units...")
                time.sleep(60)
                continue
                
            # Step 2: Spin up the Async Dual-Thread GPU engine over the bounded space
            hits = executor.execute_work_unit(work_unit)
            
            # Step 2.5: FATAL DATA LOSS PREVENTION - Save verified hits locally via SQLite
            if len(hits) > 0:
                print(f"\n[!!!] CRITICAL: {len(hits)} VERIFIED MATHEMATICAL HITS DISCOVERED!")
                print(f"[*] Writing to permanent local SQLite backup...")
                for ht in hits:
                    db_cursor.execute("INSERT INTO pending_hits VALUES (?, ?, ?, ?, ?, ?)", 
                                      (work_unit.get('v2_bound_id', 'unknown'), 
                                       str(ht.lhs_key), 
                                       str(ht.rhs_an_poly), 
                                       str(ht.rhs_bn_poly), 
                                       coordinator.client_id, 
                                       time.time()))
                db_conn.commit()
                print(f"[+] Discoveries durably secured in pending_discoveries.db\n")
            
            print("\n[*] Ready for next computation partition...\n")
            
    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt detected!")
        if db_conn:
            db_conn.close()
        print("[*] Gracefully shutting down Ramanujan@Home client.")
        sys.exit(0)

if __name__ == "__main__":
    main()
