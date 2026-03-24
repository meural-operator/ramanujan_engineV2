import firebase_admin
from firebase_admin import credentials, db
import os
import uuid
import sys
import copy

# Dynamic local fallback for Cartesian imports during headless seeding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RamanujanMachine')))
from modules.continued_fractions.domains.CartesianProductPolyDomain import CartesianProductPolyDomain

SERVICE_ACCOUNT_KEY_PATH = r"C:\Users\DIAT\ashish\ramanujan_machine\ramanujan-engine-firebase-adminsdk-fbsvc-4d453f21c2.json"
DATABASE_URL = "https://ramanujan-engine-default-rtdb.firebaseio.com"

def generate_tiered_space():
    print("==================================================")
    print("   Ramanujan@Home - Global Hypercube Seeder       ")
    print("==================================================")

    key_path = os.path.abspath(os.path.join(os.path.dirname(__file__), SERVICE_ACCOUNT_KEY_PATH))
    if not os.path.exists(key_path):
        print(f"[!] ERROR: Admin SDK key not found at '{key_path}'. Please move the JSON file here.")
        return

    print("[*] Authenticating with Firebase Admin SDK...")
    cred = credentials.Certificate(key_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})

    print("[*] Establishing the explicit Phase Space: euler-mascheroni (Deg 2, Bounds [-50, 50])")
    
    # 1. Initialize the monolithic abstract space
    master_domain = CartesianProductPolyDomain(
        a_deg=2, a_coef_range=[-50, 50],
        b_deg=2, b_coef_range=[-50, 50]
    )
    
    # The domain possesses 6 variables. We manually restrict `a_coef_range[0]` to establish 3 non-overlapping root Hypercubes!
    # Master Cube size = 101^6 = 1.06 Trillion.
    
    print("[*] Performing Mathematical Domain Partitioning...")
    # --- LARGE TIER: 200 Sub-domains, ~2.1 Billion evaluations each. (RTX ADA Focus) ---
    root_large = copy.deepcopy(master_domain)
    root_large.a_coef_range[0] = [-50, -10] 
    root_large._setup_metadata()
    chunks_large = root_large.split_domains_to_processes(200)
    
    # --- MEDIUM TIER: 4000 Sub-domains, ~105 Million evaluations each. (Std GPU Focus) ---
    root_medium = copy.deepcopy(master_domain)
    root_medium.a_coef_range[0] = [-9, 30] 
    root_medium._setup_metadata()
    chunks_medium = root_medium.split_domains_to_processes(4000)
    
    # --- SMALL TIER: 21000 Sub-domains, ~10 Million evaluations each. (CPU Focus) ---
    root_small = copy.deepcopy(master_domain)
    root_small.a_coef_range[0] = [31, 50] 
    root_small._setup_metadata()
    chunks_small = root_small.split_domains_to_processes(21000)

    # 2. Package the payloads
    def push_chunks(tier_name, chunk_list):
        print(f"[*] Packaging {len(chunk_list)} [{tier_name}] elements...")
        tier_payloads = {}
        for idx, sub_domain in enumerate(chunk_list):
            uid = str(uuid.uuid4())[:12] # Ensure long entropy
            tier_payloads[uid] = {
                "id": uid,
                "tier": tier_name,
                "constant_name": "euler-mascheroni",
                "a_deg": sub_domain.a_deg,
                "b_deg": sub_domain.b_deg,
                "a_coef_range": sub_domain.a_coef_range,
                "b_coef_range": sub_domain.b_coef_range,
                "evaluations": sub_domain.num_iterations,
                "status": "pending",
                "assigned_to": "",
                "time_assigned": 0
            }
        
        # Batch upload to explicit RTDB branches via Admin SDK
        ref = db.reference(f'work_units_{tier_name}')
        # Clean existing test data from this tier
        ref.delete()
        ref.update(tier_payloads)
        print(f"[+] Successfully flashed `{tier_name}` tier to Firebase.")

    push_chunks("large", chunks_large)
    push_chunks("medium", chunks_medium)
    push_chunks("small", chunks_small)
    
    print("\n[+] Global Generation Complete! 1+ Trillion Mathematical Nodes correctly queued.")

if __name__ == "__main__":
    generate_tiered_space()
