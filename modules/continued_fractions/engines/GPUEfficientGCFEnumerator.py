import torch
import queue
import threading
import mpmath
from time import time
from typing import List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from .EfficientGCFEnumerator import EfficientGCFEnumerator, Match, RefinedMatch
from modules.continued_fractions.utils.mobius import EfficientGCF
from modules.continued_fractions.targets import g_N_initial_search_terms, g_N_verify_terms, g_N_verify_compare_length

# Module-level worker function required for ProcessPool serialization
def _cpu_verify_worker(match_obj, lhs_possibilities, s_name_path):
    """
    Hyper-precision mpmath verification task executed completely independently in a secondary Process.
    """
    with mpmath.workdps(g_N_verify_terms * 2): # Run CPU verify at hyper-precision
        # Re-initialize isolated GCF structure
        an = EfficientGCFEnumerator.create_an_series(match_obj.rhs_an_poly, g_N_verify_terms)
        bn = EfficientGCFEnumerator.create_bn_series(match_obj.rhs_bn_poly, g_N_verify_terms)
        gcf = EfficientGCF(an, bn)
        rhs_str = mpmath.nstr(gcf.evaluate(), g_N_verify_compare_length)
        
        # Determine how to fetch LHS matches without passing huge tables over Pickle
        all_matches = []
        if lhs_possibilities is not None:
            all_matches = lhs_possibilities.get(match_obj.lhs_key, [])
        elif s_name_path:
            # Fallback to local disk loading if memory wasn't shared
            import pickle
            with open(s_name_path, 'rb') as f:
                temp_dict = pickle.load(f)
                all_matches = temp_dict.get(match_obj.lhs_key, [])
                
        # Refine Results
        for i, lhs_match in enumerate(all_matches):
            val = lhs_match[0]
            if mpmath.isinf(val) or mpmath.isnan(val):
                continue
            val_str = mpmath.nstr(val, g_N_verify_compare_length)
            if val_str == rhs_str:
                return RefinedMatch(*match_obj, i, lhs_match[1], lhs_match[2])
    return None

class GPUEfficientGCFEnumerator(EfficientGCFEnumerator):
    """
    GPU-accelerated version of EfficientGCFEnumerator using PyTorch.
    Evaluates Generalized Continued Fractions in massive parallel batches.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Scale CPU workers based on threads available (reserve 1 for main GPU loop)
        self.num_workers = max(1, os.cpu_count() - 1)
        
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[GPU] Using {gpu_name} ({gpu_mem:.1f} GB VRAM) | ProcessPool: {self.num_workers} isolated engines")
        else:
            print(f"[GPU] CUDA not available, falling back to CPU | ProcessPool: {self.num_workers} isolated engines")

    def full_execution(self, verbose=True):
        """
        Overrides AbstractGCFEnumerator's sequential pipeline.
        We handle tracking and high-precision verification asynchronously.
        """
        return self._first_enumeration(verbose)
        
    def _improve_results_precision(self, *args, **kwargs):
        raise NotImplementedError("Handled asynchronously in GPUEfficientGCFEnumerator")
        
    def _refine_results(self, *args, **kwargs):
        raise NotImplementedError("Handled asynchronously in GPUEfficientGCFEnumerator")

    def _first_enumeration(self, verbose: bool) -> List[RefinedMatch]:
        start = time()
        a_coef_iter = list(self.get_an_iterator())
        b_coef_iter = list(self.get_bn_iterator())
        
        # 1. Build batched tensors for all a_n and b_n
        a_series_list = []
        a_coef_list = []
        for coef in a_coef_iter:
            an = self.create_an_series(coef, g_N_initial_search_terms)
            if 0 not in an[1:]:
                a_series_list.append(an)
                a_coef_list.append(coef)

        b_series_list = []
        b_coef_list = []
        for coef in b_coef_iter:
            bn = self.create_bn_series(coef, g_N_initial_search_terms)
            if 0 not in bn[1:]:
                b_series_list.append(bn)
                b_coef_list.append(coef)

        if not a_series_list or not b_series_list:
            if verbose:
                print("No valid polynomial sequences found.")
            return []

        # Move sequences to Tensor
        a_tensor = torch.tensor(a_series_list, dtype=torch.float64, device=self.device) # (N_a, N_terms)
        b_tensor = torch.tensor(b_series_list, dtype=torch.float64, device=self.device) # (N_b, N_terms)
        
        N_a = a_tensor.shape[0]
        N_b = b_tensor.shape[0]
        N_terms = a_tensor.shape[1]
        
        num_iterations = N_a * N_b
        if verbose:
            print(f"Created final enumerations filters after {time() - start:.2f}s")
            print(f"Batch evaluating {num_iterations} combinations on {self.device}...")
        
        raw_hits = []
        refined_results = []
        
        pbar_worker = tqdm(total=0, desc="CPU Verify Multiproc", position=1, leave=False)
        # Using ProcessPoolExecutor to shatter GIL bottlenecks during Deep Mpmath Precision
        executor = ProcessPoolExecutor(max_workers=self.num_workers)
        futures = []
        
        # Prepare lightweight dict pointer for passing LHS references securely
        tbl_dict = self.hash_table.lhs_possibilities if self.hash_table.lhs_possibilities is not None else None
        tbl_file = getattr(self.hash_table, 's_name', None)
        
        key_factor = round(1 / self.threshold)
        processed_combinations = 0
        log_start_time = time()
        
        # ─────────────────────────────────────────────────────────
        # ZERO-LATENCY GPU OPTIMIZATION:
        # Move LHS hash keys directly to CUDA for vectorized matching
        # ─────────────────────────────────────────────────────────
        if self.hash_table.lhs_possibilities is not None:
            valid_keys = list(self.hash_table.lhs_possibilities.keys())
        else:
            # If using only the bloom filter (from .db cache), we can't easily vectorize.
            # But the user loaded from dept30 so we trigger a quick load if needed.
            # For pure GPU speed, we force load the keys.
            valid_keys = []
            if getattr(self.hash_table, 's_name', None):
                import pickle
                with open(self.hash_table.s_name, 'rb') as f:
                    temp_dict = pickle.load(f)
                    valid_keys = list(temp_dict.keys())
                    
        # Convert string keys to long integers and move to GPU
        valid_keys_int = [int(k) for k in valid_keys]
        lhs_keys_tensor = torch.tensor(valid_keys_int, dtype=torch.long, device=self.device)
        
        # Batch chunks tuned for 20GB VRAM
        # Each cross-product element uses ~560 bytes (tensors + buffers)
        # CHUNK_A * CHUNK_B should stay under ~10M for safety
        # Reduce these if you still get OOM errors
        CHUNK_A = 500
        CHUNK_B = 10000
        
        # Determine total outer loops for tqdm
        total_a_chunks = (N_a + CHUNK_A - 1) // CHUNK_A
        total_b_chunks = (N_b + CHUNK_B - 1) // CHUNK_B
        total_pbar_steps = total_a_chunks * total_b_chunks
        
        with tqdm(total=num_iterations, desc="GPU GPU Batch Eval") as pbar:
            for i in range(0, N_a, CHUNK_A):
                a_chunk = a_tensor[i : i + CHUNK_A] 
                chunk_a_size = a_chunk.shape[0]
                
                for j in range(0, N_b, CHUNK_B):
                    b_chunk = b_tensor[j : j + CHUNK_B] 
                    chunk_b_size = b_chunk.shape[0]
                
                    # Cross product using broadcasting and reshape
                    a_expanded = a_chunk.unsqueeze(1).expand(chunk_a_size, chunk_b_size, N_terms).reshape(-1, N_terms)
                    b_expanded = b_chunk.unsqueeze(0).expand(chunk_a_size, chunk_b_size, N_terms).reshape(-1, N_terms)
                    
                    bsz = a_expanded.shape[0]
                    
                    prev_q = torch.zeros(bsz, dtype=torch.float64, device=self.device)
                    q = torch.ones(bsz, dtype=torch.float64, device=self.device)
                    prev_p = torch.ones(bsz, dtype=torch.float64, device=self.device)
                    p = a_expanded[:, 0].clone()
                    
                    # Batched convergent evaluation
                    for k in range(1, N_terms):
                        tmp_q = q.clone()
                        tmp_p = p.clone()
                        q = a_expanded[:, k] * q + b_expanded[:, k] * prev_q
                        p = a_expanded[:, k] * p + b_expanded[:, k] * prev_p
                        prev_q = tmp_q
                        prev_p = tmp_p
                        
                        # Periodic scaling to prevent float64 overflow, taking max magnitude
                        scale = q.abs().clamp_min(1.0)
                        q /= scale
                        p /= scale
                        prev_q /= scale
                        prev_p /= scale

                    dist = key_factor * p / q
                    dist = torch.nan_to_num(dist, nan=0.0)
                    hash_keys = dist.trunc().long()
                    
                    # ─────────────────────────────────────────────────────────
                    # VECTORIZED MATCHING (Replaces 5M Python loop)
                    # ─────────────────────────────────────────────────────────
                    # Check intersections entirely on CUDA
                    matches_mask = torch.isin(hash_keys, lhs_keys_tensor)
                    
                    # Extract indices of matches
                    match_indices = torch.nonzero(matches_mask).squeeze(1)
                    
                    if match_indices.numel() > 0:
                        # Move only the specific hit indices back to CPU
                        match_indices_cpu = match_indices.cpu().numpy()
                        keys_cpu = hash_keys[match_indices].cpu().numpy()
                        
                        for idx_gpu, key in zip(match_indices_cpu, keys_cpu):
                            a_idx = i + int(idx_gpu) // chunk_b_size
                            b_idx = j + int(idx_gpu) % chunk_b_size
                            match = Match(key, a_coef_list[a_idx], b_coef_list[b_idx])
                            raw_hits.append(match)
                            
                            # Submit task to isolated CPU process instead of Threadpool to shatter GIL
                            future = executor.submit(_cpu_verify_worker, match, tbl_dict, tbl_file)
                            futures.append(future)
                            pbar_worker.total += 1
                            pbar_worker.refresh()
                    
                    # ─────────────────────────────────────────────────────────
                    # LIVE PROGRESS LOGGING & ETA via tqdm
                    # ─────────────────────────────────────────────────────────
                    processed_combinations += bsz
                    pbar.update(bsz)
                    pbar.set_postfix({'Raw Hits': len(raw_hits)})
                    
                    if verbose and (processed_combinations % (bsz * 50) == 0 or processed_combinations == num_iterations):
                        # Write persistently to log file every ~50 batches to avoid IO bottleneck
                        with open("search_progress.log", "a") as logf:
                            log_str = f"Processed: {processed_combinations:,}/{num_iterations:,} | Raw Hits: {len(raw_hits)} | Verified: {len(refined_results)}"
                            logf.write(f"[{time()}] {log_str}\n")
                        
        pbar_worker.set_description("CPU Draining Process Queue")
        
        # Yield out completed futures seamlessly
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                refined_results.append(res)
                if verbose:
                    tqdm.write(f"\n[!] VERIFIED CONJECTURE FOUND! {res}")
            
            pbar_worker.update(1)
            pbar_worker.set_postfix({'Verified Hits': len(refined_results)})

        executor.shutdown(wait=True)
        pbar_worker.close()

        if verbose:
            print(f"\nCreated results after {time() - start:.2f}s. Found {len(refined_results)} final verified matches.")
            
        return refined_results
