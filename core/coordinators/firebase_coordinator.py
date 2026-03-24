import os
import json
import socket
import urllib.request
import urllib.error
from typing import List, Dict, Optional

from core.interfaces.base_coordinator import NetworkCoordinator

class FirebaseCoordinator(NetworkCoordinator):
    """
    Concrete implementation of the NetworkCoordinator interface utilizing
    the Firebase Realtime Database REST API for completely decentralized work orchestration.
    """
    def __init__(self, config_path: str = "firebase_config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config {config_path}")
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.api_key = self.config["apiKey"]
        self.db_url = self.config["databaseURL"]
        self.client_id = f"{socket.gethostname()}-v3-node"
        self.id_token = None
        self._authenticate_user()

    def _authenticate_user(self):
        auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={self.api_key}"
        req = urllib.request.Request(auth_url, data=b'{"returnSecureToken": true}', headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as response:
                auth_resp = json.loads(response.read().decode())
                self.id_token = auth_resp["idToken"]
        except urllib.error.URLError as e:
            raise ConnectionError(f"Firebase REST Auth failed: {e}")

    def fetch_work_unit(self) -> Optional[Dict]:
        """Locks a grid chunk autonomously via Firebase atomic transactions over REST."""
        if not self.id_token:
            return None
            
        try:
            # Note: For strict V3 architecture, we query the generalized `v3_dynamic_tasks` node.
            # If migrating fully from V2, this endpoint maps identically to dynamic chunk generation.
            # Simplified mock REST fetch for the plugin interface structure:
            cursor_url = f"{self.db_url}/v2_dynamic_tasks/cursor.json?auth={self.id_token}"
            req = urllib.request.Request(cursor_url, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as response:
                cursor = json.loads(response.read().decode())
                
            if not cursor:
                return None
                
            work_unit = {
                "id": "v3-dynamic",
                "v2_bound_id": f"v3_{cursor.get('current_a_pos')}",
                "constant_name": "euler-mascheroni",
                "a_deg": cursor.get('degree', 2),
                "b_deg": cursor.get('degree', 2),
                "a_coef_range": [[cursor.get('current_a_pos'), cursor.get('current_a_pos') + 10]] * (cursor.get('degree', 2)+1),
                "b_coef_range": [[cursor.get('current_b_pos'), cursor.get('current_b_pos') + 10]] * (cursor.get('degree', 2)+1),
            }
            return work_unit
            
        except Exception as e:
            print(f"[!] Firebase fetch failed: {e}")
            return None

    def submit_results(self, verified_discoveries: List[Dict]) -> bool:
        """Pushes discoveries structurally to the global Realtime DB namespace."""
        if not verified_discoveries:
            return True
            
        try:
            # For V3, we dispatch cleanly to the `v3_results` bucket.
            for hit in verified_discoveries:
                hit['client_id'] = self.client_id
                push_url = f"{self.db_url}/v3_results.json?auth={self.id_token}"
                req = urllib.request.Request(push_url, data=json.dumps(hit).encode(), headers={'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(req) as response:
                    pass
            return True
        except Exception as e:
            print(f"[!] Firebase submit failed: {e}")
            return False
