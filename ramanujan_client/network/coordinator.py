import json
import os
import time
import uuid
import requests

import pyrebase


class ServerCoordinator:
    """
    V2-Only Firebase Coordinator.
    Handles Cloud-Native Firebase Realtime Database syncing.
    Uses infinite-scale V2 Dynamic Tasks (Cursor) and abandons V1 static chunks.
    No Flutter UI telemetry or User Account requirements.
    """
    def __init__(self, config_path="firebase_config.json"):
        self.config_path = config_path
        self.firebase = None
        self.db = None
        self.auth = None
        self.user = None
        self.id_token = None
        
        self.client_id = str(uuid.uuid4())
        
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
        
        print(f"[*] Firebase Initialized successfully.")

    def authenticate_user(self):
        """
        Authenticates securely using Anonymously provisioned identities.
        (Since Flutter integration was removed, we use Anonymous Auth for scale).
        """
        print(f"[*] Attempting Anonymous Node Authentication...")
        try:
            self.user = self.auth.sign_in_anonymous()
            self.id_token = self.user['idToken']
            self.client_id = self.user.get('localId', self.client_id)
            print(f"[+] Client authenticated anonymously! UID: {self.client_id}")
            return self.id_token
        except Exception as e:
            # If Firebase proj doesn't have Anonymous auth enabled, fallback to None (open rules)
            print(f"[-] Anonymous Auth failed or disabled: {e}. Falling back to Open Access mode.")
            return "open_access"

    def request_work_unit(self):
        """
        V2 Cursor Logic: Claims a mathematical bounded chunk on-the-fly directly 
        from the infinite `v2_dynamic_tasks/cursor` node, rather than querying 250k rows.
        """
        print(f"[*] Querying V2 Dynamic Task Cursor...")
        try:
            # 1. Fetch current cursor parameters with ETag for atomic locking
            url = f"{self.firebase.database_url}/v2_dynamic_tasks/cursor.json?auth={self.id_token}"
            headers = {'X-Firebase-ETag': 'true'}
            
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200 or not resp.json():
                print(f"[-] V2 Cursor not found or unauthorized. Sleeping.")
                return None
                
            cursor_data = resp.json()
            etag = resp.headers.get('ETag')
            
            degree = cursor_data.get("degree", 2)
            chunk_width = cursor_data.get("chunk_width", 5)
            
            # The current central boundary position
            curr_pos_a = cursor_data.get("current_a_pos", -10)
            curr_pos_b = cursor_data.get("current_b_pos", -10)
            
            b_max = cursor_data.get("b_max", 30)
            a_max = cursor_data.get("a_max", 30)

            # Check if bounds hit completion
            if curr_pos_a >= a_max:
                print(f"[+] V2 Phase space exhausted! All bounds generated.")
                return None

            # 2. Build explicit cartesian bounds for this node
            # Create [min, max] list for each polynomial coefficient (e.g. 3 coefficients for deg 2)
            a_coef_range = [[curr_pos_a, curr_pos_a + chunk_width]] * (degree + 1)
            b_coef_range = [[curr_pos_b, curr_pos_b + chunk_width]] * (degree + 1)

            work_unit = {
                "id": str(uuid.uuid4()),
                "v2_bound_id": f"bound_a{curr_pos_a}_b{curr_pos_b}",
                "constant_name": "euler-mascheroni",
                "a_deg": degree,
                "b_deg": degree,
                "a_coef_range": a_coef_range,
                "b_coef_range": b_coef_range
            }
            
            # 3. Patch the cursor forward so the next nodes receive new bounds
            next_b = curr_pos_b + chunk_width
            next_a = curr_pos_a
            
            if next_b >= b_max:
                next_b = cursor_data.get("b_min", -30)
                next_a += chunk_width

            # Apply patch to local copy
            cursor_data["current_a_pos"] = next_a
            cursor_data["current_b_pos"] = next_b
            
            # 4. Attempt an Atomic Write using Firebase REST ETag
            put_headers = {'if-match': etag}
            put_resp = requests.put(url, json=cursor_data, headers=put_headers)
            
            if put_resp.status_code == 412: # HTTP 412 Precondition Failed
                print(f"[-] Race collision detected! Another node locked these bounds just milliseconds ago. Re-entering queue...")
                return None # The execution loop will natively retry and grab the next available spot
            elif put_resp.status_code != 200:
                print(f"[!] Atomic lock failed with status {put_resp.status_code}. Sleeping.")
                return None

            print(f"[+] Mutually Exclusive Lock Acquired: V2 Bounds starting at a={curr_pos_a}, b={curr_pos_b}")

            return work_unit

        except Exception as e:
            print(f"[!] Failed to fetch V2 work units: {e}")
            return None

    def submit_results(self, work_unit, hits):
        if not hits:
            return True # Nothing to sync
            
        print(f"[*] Submitting {len(hits)} verified results to V2 results pool...")
        v2_bound_id = work_unit.get('v2_bound_id')
        
        # Instantiate pristine Database bypass
        db = self.firebase.database()
        
        try:
            for hit in hits:
                result_data = {
                    "v2_bound_id": v2_bound_id,
                    "lhs_key": str(hit.lhs_key),
                    "rhs_an_poly": str(hit.rhs_an_poly),
                    "rhs_bn_poly": str(hit.rhs_bn_poly),
                    "client_id": self.client_id,
                    "timestamp": int(time.time())
                }
                
                # Push atomic hit to V2 array
                db.child("v2_dynamic_tasks").child("results").push(result_data, self.id_token)
            
            print(f"[+] Successfully synced {len(hits)} hits to Cloud.")
            return True
            
        except Exception as e:
            print(f"[!] Error submitting hits: {e}")
            return False

    def submit_results_bulk(self, pending_rows):
        if not pending_rows:
            return []
            
        print(f"[*] Attempting Cloud Sync for {len(pending_rows)} pending discoveries...")
        successful_ids = []
        db = self.firebase.database()
        
        for row in pending_rows:
            rowid, v2_bound_id, lhs_key, rhs_an, rhs_bn, client_id, timestamp = row
            result_data = {
                "v2_bound_id": v2_bound_id,
                "lhs_key": lhs_key,
                "rhs_an_poly": rhs_an,
                "rhs_bn_poly": rhs_bn,
                "client_id": client_id,
                "timestamp": timestamp
            }
            try:
                db.child("v2_dynamic_tasks").child("results").push(result_data, self.id_token)
                successful_ids.append(rowid)
            except Exception as e:
                print(f"[!] Sync failed for hit {rowid}: {e}")
                
        if successful_ids:
            print(f"[+] Successfully synced {len(successful_ids)}/{len(pending_rows)} discoveries to Cloud.")
            
        return successful_ids
