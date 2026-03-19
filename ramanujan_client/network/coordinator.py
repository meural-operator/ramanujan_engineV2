import json
import os
import time
import requests
import pyrebase

class ServerCoordinator:
    """
    Handles the Cloud-Native Firebase Realtime Database syncing 
    along with Google Sign-In Authentication via Pyrebase4.
    """
    def __init__(self, config_path="firebase_config.json", token_path="user_token.json"):
        self.config_path = config_path
        self.token_path = token_path
        self.firebase = None
        self.auth = None
        self.db = None
        self.user = None
        self.id_token = None
        
        self._initialize_firebase()

    def _initialize_firebase(self):
        if not os.path.exists(self.config_path):
            print(f"[!] ERROR: {self.config_path} not found. Please create it first.")
            return
            
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
            
        self.firebase = pyrebase.initialize_app(self.config)
        self.auth = self.firebase.auth()
        self.db = self.firebase.database()

    def authenticate_user(self):
        """
        Implements the 'Initial Web-Based Login with Refresh Tokens' flow.
        Returns the currently valid idToken.
        """
        if not self.auth:
            return None

        # 1. Check for Existing Refresh Token
        if os.path.exists(self.token_path):
            print("[*] Found existing user_token.json. Attempting to refresh session...")
            try:
                with open(self.token_path, "r") as f:
                    token_data = json.load(f)
                    refresh_token = token_data.get("refreshToken")
                    
                if refresh_token:
                    self.user = self.auth.refresh(refresh_token)
                    self.id_token = self.user['idToken']
                    
                    self._save_token({
                        "refreshToken": self.user['refreshToken'],
                        "localId": self.user.get('userId', self.user.get('localId')) 
                    })
                    print("[+] Firebase Session refreshed successfully.")
                    return self.id_token
                    
            except Exception as e:
                print(f"[-] Session refresh failed: {e}. Falling back to initial login.")
                try:
                    os.remove(self.token_path)
                except OSError:
                    pass

        # 2. Initial Google Sign-In (User-Guided)
        return self.perform_initial_google_login()

    def perform_initial_google_login(self):
        print("\n" + "="*50)
        print("          Ramanujan@Home - Authentication         ")
        print("="*50)
        print("1. Your web browser has automatically opened our secure portal:")
        print("   https://ramanujan-engine.web.app")
        print("   (If it didn't open, please visit that link manually).")
        print("2. Sign in with your Google Account.")
        print("3. Click 'Copy ID Token'.")
        print("4. Right-click here and paste that Token to proceed.\n")
        
        try:
            import webbrowser
            webbrowser.open("https://ramanujan-engine.web.app")
        except Exception:
            pass
            
        # We need an id_token to exchange
        pasted_id_token = input("Paste your Secure Node Token here: ").strip()
        
        if not pasted_id_token:
            print("[!] Authentication aborted.")
            return None
            
        try:
            print("[*] Verifying Secure Node Token...")
            import base64
            
            # 1. Attempt to decode the modernized offline token bundle 
            try:
                decoded_str = base64.b64decode(pasted_id_token).decode('utf-8')
                token_data = json.loads(decoded_str)
                
                self.id_token = token_data.get('idToken')
                refresh_token = token_data.get('refreshToken')
                local_id = token_data.get('localId')
                
                self.user = token_data
                
                if refresh_token:
                    self._save_token({
                        "refreshToken": refresh_token,
                        "localId": local_id
                    })
                    print("[+] Token bundled and saved securely. Persistent session established!")
                else:
                    print("[+] Temporary Firebase Token accepted (No refresh capability).")
                    
                return self.id_token
                
            except Exception:
                # 2. Fallback: Parse the raw unbundled Firebase JWT specifically
                print("[*] Detected raw Identity Token. Applying direct node access...")
                
                # Standard raw Firebase Tokens do not possess infinite refresh paths
                self.id_token = pasted_id_token
                
                # Natively decrypt the JWT payload offline to securely extract the Firebase LocalID (UID)
                local_id = "unknown_client"
                try:
                    parts = pasted_id_token.split('.')
                    if len(parts) >= 2:
                        import base64
                        payload_b64 = parts[1]
                        # Fix Base64Url padding constraints automatically
                        payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
                        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode('utf-8'))
                        local_id = payload.get('user_id', payload.get('sub', "unknown_client"))
                except Exception as e:
                    print(f"[-] JWT UID extraction failed computationally: {e}")
                    
                self.user = {"idToken": pasted_id_token, "localId": local_id}
                
                print("[+] Fallback Authentication complete!")
                return self.id_token
                
        except Exception as e:
            print(f"[!] Authentication failed: {e}")
            return None

    def _save_token(self, token_data):
        with open(self.token_path, "w") as f:
            json.dump(token_data, f, indent=4)

    def request_work_unit(self):
        if not self.id_token:
            print("[!] ERROR: Not authenticated.")
            return None

        print("[*] Querying Firebase for pending work units...")
        try:
            # Query the Realtime Database for work_units where status is "pending"
            work_units_query = self.db.child("work_units").order_by_child("status").equal_to("pending").limit_to_first(5).get(self.id_token)
            
            if not work_units_query.val():
                print("[-] No pending work units available.")
                return None
                
            # Iterate through pending units to find one we can claim
            for unit in work_units_query.each():
                unit_id = unit.key()
                unit_data = unit.val()
                
                print(f"[*] Attempting to claim Work Unit: {unit_id}")
                
                user_uid = self.user.get('localId', self.user.get('userId'))
                
                # Update its status to "assigned", set assigned_to to Firebase UID
                update_data = {
                    "status": "assigned",
                    "assigned_to": user_uid,
                    "time_assigned": int(time.time())
                }
                
                # Attempt the atomic-like update using the idToken and precise Security Rules
                try:
                    self.db.child("work_units").child(unit_id).update(update_data, self.id_token)
                    print(f"[+] Successfully claimed Work Unit {unit_id}!")
                    
                    if "id" not in unit_data:
                        unit_data["id"] = unit_id
                        
                    return unit_data
                except Exception as update_err:
                    print(f"[-] Someone else claimed {unit_id} first or permission denied. Trying next...")
                    continue
                    
            print("[-] Flushed pending queue, but could not claim a unit.")
            return None
            
        except Exception as e:
            print(f"[!] Failed to fetch work units: {e}")
            return None

    def submit_results(self, work_unit_id, hits):
        if not self.id_token:
            return False
            
        print(f"[*] Submitting {len(hits)} verified results to Firebase...")
        user_uid = self.user.get('localId', self.user.get('userId'))
        
        try:
            for hit in hits:
                result_data = {
                    "lhs_key": str(hit.lhs_key),
                    "rhs_an_poly": str(hit.rhs_an_poly),
                    "rhs_bn_poly": str(hit.rhs_bn_poly),
                    "client_id": user_uid,
                    "timestamp": int(time.time())
                }
                
                # Push the atomic hit object, relying on security rules to lock it eternally 
                self.db.child("results").push(result_data, self.id_token)
            
            # Finalize the workload block as completed
            self.db.child("work_units").child(work_unit_id).update({
                "status": "completed"
            }, self.id_token)
            
            print(f"[+] Successfully submitted {len(hits)} results and marked Work Unit {work_unit_id} as completed.")
            return True
            
        except Exception as e:
            print(f"[!] Error submitting results: {e}")
            return False
