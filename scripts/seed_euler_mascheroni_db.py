#!/usr/bin/env python3
"""
Seed Script: Local Euler-Mascheroni LHS DB Generator
=====================================================

To perform the GPU-accelerated enumeration, the core framework needs a pre-calculated 
Hash Table of Mobius Transformations of the Target Constant.

This script mathematically generates `euler_mascheroni.db` into the client folder.
Run this ONCE before distributing the node package.
"""

import os
import sys
import time
from ramanujan.LHSHashTable import LHSHashTable
from ramanujan.constants import g_const_dict

def main():
    print("==================================================")
    print("   Seeding Euler-Mascheroni DB (LHS Hash Table)   ")
    print("==================================================\n")
    
    # Place it directly in the client folder or repo root depending on execution path
    # ramanujan_client code expects it in its CWD, so let's put it in the root for now
    dest_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ramanujan_client', 'euler_mascheroni.db'))
    
    if os.path.exists(dest_path):
        print(f"[*] Database already exists at: {dest_path}")
        print("[*] No action required.")
        return

    print(f"[*] Generating depth-30 Hash Table. This will take ~5-15 seconds and max out CPU...")
    start = time.time()
    
    # Instantiating it automatically builds and saves the .db file
    lhs = LHSHashTable(
        name=dest_path,
        search_range=30,  # Strict depth of 30 for Mobius tree
        const_vals=[g_const_dict['euler-mascheroni']]
    )
    
    elapsed = time.time() - start
    print(f"[+] Successfully generated DB in {elapsed:.1f}s.")
    print(f"[+] File saved to: {dest_path}")
    print(f"[+] Client is now ready to run `ramanujan_client.py`")


if __name__ == '__main__':
    main()
