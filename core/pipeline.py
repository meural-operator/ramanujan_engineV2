import time
import sqlite3
from typing import List

from core.interfaces.base_problem import TargetProblem
from core.interfaces.base_strategy import BoundingStrategy
from core.interfaces.base_engine import ExecutionEngine
from core.interfaces.base_coordinator import NetworkCoordinator

class UniversalPipelineRouter:
    """
    The ultimate decoupled V3 routing pipeline.
    Arbitrarily shuttles workload dimensions between any combination of mathematical
    constants, AI heuristics, GPU tensor arrays, and network states.
    """
    def __init__(self, 
                 target: TargetProblem, 
                 strategies: List[BoundingStrategy], 
                 engine: ExecutionEngine, 
                 network: NetworkCoordinator):
        self.target = target
        self.strategies = strategies
        self.engine = engine
        self.network = network

    def execute_work_unit(self, work_unit) -> List[dict]:
        print(f"\n==================================================")
        print(f"[*] Engine V3: Universal Pipeline Mode")
        print(f"[*] Target Constant: {self.target.name} [Precision: {self.target.precision}]")
        print(f"==================================================\n")
        
        a_bounds = work_unit.get('a_coef_range')
        b_bounds = work_unit.get('b_coef_range')
        
        # 1. Pipe boundaries through sequential AI/Mathematics reduction heuristics
        for strat in self.strategies:
            print(f"[*] Dispatching Phase-Space to strategy plugin: {strat.strategy_name}")
            a_bounds, b_bounds = strat.prune_bounds(a_bounds, b_bounds)
            print(f"    -> Dynamically refined a: {a_bounds}")
            print(f"    -> Dynamically refined b: {b_bounds}")
            
        # 2. Dispatch surviving bound constraints iteratively into Bare-Metal compute
        print(f"[*] Sinking tightly constrained subspace into Engine plugin: {self.engine.engine_id}")
        verified_discoveries = self.engine.batch_evaluate(a_bounds, b_bounds, self.target)
        
        print(f"\n[+] Pipeline execution finished. Ultra-verified formulas identified: {len(verified_discoveries)}")
        return verified_discoveries
        
    def run_compute_loop(self, sqlite_path="v3_pending_discoveries.db"):
        print(f"[*] Mathematical Discovery Framework Initialized. Entering Community compute loop...\n")
        
        conn = sqlite3.connect(sqlite_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS pending_hits (
                        v2_bound_id TEXT, lhs_key TEXT, rhs_an TEXT, 
                        rhs_bn TEXT, client_id TEXT, ts REAL)''')
        conn.commit()

        try:
            while True:
                # 1. Guaranteed Sync Architecture
                c.execute("SELECT rowid, * FROM pending_hits")
                pending_rows = c.fetchall()
                if pending_rows:
                    if self.network.submit_results(pending_rows):
                        success_ids = [str(r[0]) for r in pending_rows]
                        c.execute(f"DELETE FROM pending_hits WHERE rowid IN ({','.join('?'*len(success_ids))})", success_ids)
                        conn.commit()
                        print(f"[*] Edge Storage Synced. Flushed {len(success_ids)} verified targets to global state.")
                
                # 2. Fetch Abstract Payload
                unit = self.network.fetch_work_unit()
                if not unit:
                    print("[-] Decentralized Work Queue exhausted. Delaying constraint polling 60s...")
                    time.sleep(60)
                    continue
                    
                # 3. Pipeline Execution Hook
                hits = self.execute_work_unit(unit)
                
                # 4. Zero-Loss Local Cache Intercept
                if hits:
                    print(f"\n[!!!] CRITICAL: {len(hits)} VERIFIED MATHEMATICAL HITS DISCOVERED!")
                    
                    # 4a. LLL/PSLQ Algebraic Identity Resolution
                    try:
                        from modules.continued_fractions.utils.lll_identity_resolver import (
                            resolve_identity, format_identity_report
                        )
                        print(f"[*] Running LLL/PSLQ integer relation detection on {len(hits)} hits...")
                        for ht in hits:
                            identity = resolve_identity(
                                float(str(ht.get('lhs_key', '0')).replace('_', '.')),
                                basis_constants={'gamma', 'pi', 'log2', 'zeta2', 'zeta3', '1'}
                            )
                            ht['identity'] = identity
                            report = format_identity_report(ht, identity)
                            print(report)
                    except Exception as e:
                        print(f"[!] LLL resolver unavailable ({e}), skipping identity resolution.")
                    
                    # 4b. Permanent local SQLite backup
                    print(f"[*] Writing to permanent local SQLite backup...")
                    for ht in hits:
                        c.execute("INSERT INTO pending_hits VALUES (?, ?, ?, ?, ?, ?)", 
                                  (unit.get('v2_bound_id', 'unknown'), 
                                   str(ht['lhs_key']), str(ht['a_coef']), str(ht['b_coef']), 
                                   "v3-edge-node", time.time()))
                    conn.commit()
                print("\n[*] Ready for next computational block...\n")
                
        except KeyboardInterrupt:
            print("\n[!] Gracefully shutting down Ramanujan V3 Universal Framework.")
            conn.close()
