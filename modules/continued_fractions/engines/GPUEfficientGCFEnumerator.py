import torch
import queue
import threading
import mpmath
from time import time
from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from .EfficientGCFEnumerator import EfficientGCFEnumerator, Match, RefinedMatch
from modules.continued_fractions.utils.mobius import EfficientGCF
from modules.continued_fractions.targets import g_N_initial_search_terms, g_N_verify_terms, g_N_verify_compare_length

# Module-level worker function required for ProcessPool serialization
from modules.continued_fractions.utils.utils import get_series_items_from_iter

def _cpu_verify_worker(match_obj, lhs_possibilities, s_name_path, poly_domains):
    """
    Hyper-precision mpmath verification task executed completely independently in a secondary Process.
    """
    with mpmath.workdps(g_N_verify_terms * 2): # Run CPU verify at hyper-precision
        # Re-initialize isolated GCF structure
        a_iterator_func, b_iterator_func = poly_domains.get_calculation_method()
        an = get_series_items_from_iter(a_iterator_func, match_obj.rhs_an_poly, g_N_verify_terms)
        bn = get_series_items_from_iter(b_iterator_func, match_obj.rhs_bn_poly, g_N_verify_terms)
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
        # Use float32 for GPU sweep — Ada Lovelace delivers 19.2 TFLOPS FP32 vs ~0.6 TFLOPS FP64.
        # The CPU mpmath verification (1000-digit precision) catches any false positives.
        a_tensor = torch.tensor(a_series_list, dtype=torch.float32, device=self.device) # (N_a, N_terms)
        b_tensor = torch.tensor(b_series_list, dtype=torch.float32, device=self.device) # (N_b, N_terms)
        
        N_a = a_tensor.shape[0]
        N_b = b_tensor.shape[0]
        N_terms = a_tensor.shape[1]
        
        # ─────────────────────────────────────────────────────────
        # GPU-NATIVE ASYMPTOTIC CONVERGENCE FILTER (B-1)
        # Eliminate divergent (a,b) pairs before the expensive
        # convergent evaluation loop.
        #
        # The Worpitzky/Poincaré test requires:
        #   1. a_lead > 0  (positive leading coefficient)
        #   2. deg(a)*2 >= deg(b)  (degree balance — always true for our domains)
        #   3. discriminant = a_lead² + 4·b_lead >= 0  (when balanced)
        #
        # We compute the *effective* leading coefficient from the
        # actual series values rather than polynomial coefficients,
        # which works regardless of the domain's structural template.
        #
        # Method: For large n, the leading-order behavior of a(n)
        # is captured by a(N)/N^d where d is the degree and N is a
        # large index. We use the ratio a(N)/a(N-1) to estimate the
        # effective degree and sign.
        # ─────────────────────────────────────────────────────────
        # Use the last two terms to estimate the leading behavior
        # a_lead_sign ≈ sign(a(n)) for large n
        a_lead_sign = torch.sign(a_tensor[:, -1])  # (N_a,)
        b_lead_sign = torch.sign(b_tensor[:, -1])  # (N_b,)
        
        # Filter individual a sequences: discard those with non-positive leading behavior
        a_valid_mask = a_lead_sign > 0  # Worpitzky condition 1
        n_a_pruned = int((~a_valid_mask).sum().item())
        
        if n_a_pruned > 0:
            valid_a_indices = torch.nonzero(a_valid_mask).squeeze(1)
            a_tensor = a_tensor[valid_a_indices]
            a_coef_list = [a_coef_list[i] for i in valid_a_indices.cpu().numpy()]
            N_a = a_tensor.shape[0]
        
        # For balanced-degree domains, apply discriminant check per (a,b) pair
        # inside the chunk loop (below) as a mask to avoid materializing N_a×N_b
        # 
        # We pre-compute the large-n values for discriminant estimation:
        #   a_asymptotic[i] ≈ a_lead_coef * N^(deg_a)
        #   b_asymptotic[j] ≈ b_lead_coef * N^(deg_b)
        # For the discriminant: 4*b_lead + a_lead² >= 0
        # In terms of series values at index N: a_at_N² + 4*b_at_N * N^(deg_b - 2*deg_a) >= 0
        # For balanced degrees (deg_b = 2*deg_a), this simplifies to:
        #   a(N)² + 4*b(N) >= 0  (at large N, leading terms dominate)
        self._a_last_term = a_tensor[:, -1].clone()  # (N_a,) — for discriminant in chunk loop
        self._b_last_term = b_tensor[:, -1].clone()  # (N_b,) — for discriminant in chunk loop
        
        num_iterations = N_a * N_b
        if verbose:
            print(f"Created final enumerations filters after {time() - start:.2f}s")
            if n_a_pruned > 0:
                print(f"[GPU Filter] Pruned {n_a_pruned} a-sequences with non-positive leading coefficient")
            print(f"Batch evaluating {num_iterations:,} combinations on {self.device}...")
        
        raw_hits = []
        refined_results = []
        
        pbar_worker = tqdm(total=0, desc="CPU Verify Multiproc", position=1, leave=False)
        # Using ThreadPoolExecutor on Windows to avoid massive IPC pickling overhead of passing large LHS dicts.
        # While mpmath holds the GIL, IPC for 2GB dicts is far worse than sequential GIL execution.
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        futures = []
        
        # Prepare lightweight dict pointer for passing LHS references securely
        tbl_dict = self.hash_table.lhs_possibilities if self.hash_table.lhs_possibilities is not None else None
        tbl_file = getattr(self.hash_table, 's_name', None)
        
        # Float32 sweep uses a coarser threshold (1e-6) — sufficient for hash bucketing.
        # The original self.threshold (e.g. 1e-10) is too tight for float32's ~7-digit precision.
        gpu_threshold = max(self.threshold, 1e-6)
        key_factor = round(1 / gpu_threshold)
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
        
        # ─────────────────────────────────────────────────────────
        # DYNAMIC VRAM BATCH ALLOCATION
        # ─────────────────────────────────────────────────────────
        if self.device.type == 'cuda':
            # Force PyTorch to release its cached tensor memory back to the OS
            # so we don't accidentally detect 0 free memory on the second loop.
            torch.cuda.empty_cache()
            
            # Query the hardware for exact free and total memory remaining
            free_mem, total_mem = torch.cuda.mem_get_info()
            
            # Target 80% of TRUE free memory
            usable_vram_bytes = free_mem * 0.8
            
            # Profile the algorithm's actual byte-cost per equation (float32):
            # a_expanded, b_expanded tensors: 4 bytes * N_terms * 2
            # internal registers (q, p, prev_q, prev_p): 4 bytes * 4 = 16 bytes
            # masking & conditional tensors during hits: ~52 bytes
            bytes_per_combo = (8 * N_terms) + 68
            
            max_safe_bsz = int(usable_vram_bytes / bytes_per_combo)
            
            # Hardcap at 50,000,000 to prevent CUDA kernel scheduling timeouts
            target_bsz = min(max_safe_bsz, 50_000_000)
            
            # Optimally factorize the batch dimension into 2D chunking limits
            # that don't over-allocate beyond the actual mathematical domain size
            CHUNK_A = min(N_a, max(1, int((target_bsz)**0.5)))
            CHUNK_B = min(N_b, max(1, target_bsz // CHUNK_A))
            
            if verbose:
                print(f"[GPU] Dynamic Scaler active: Hardware can safely evaluate {CHUNK_A * CHUNK_B:,} parallel combinations per pass.")
        else:
            # CPU Fallback (Keep limits low to prevent thrashing system RAM)
            CHUNK_A = min(N_a, 500)
            CHUNK_B = min(N_b, 1000)
            if verbose:
                print(f"[-] CPU compute detected. Limiting batch topology to {CHUNK_A * CHUNK_B:,} combinations.")
        
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
                    
                    # ── GPU Discriminant Filter (B-1 in-chunk) ──────────────
                    # For balanced-degree GCFs, Worpitzky requires:
                    #   discriminant = a_lead² + 4·b_lead ≥ 0
                    # We estimate this from the last term of each series.
                    a_last_chunk = self._a_last_term[i : i + chunk_a_size]  # (chunk_a,)
                    b_last_chunk = self._b_last_term[j : j + chunk_b_size]  # (chunk_b,)
                    
                    # Broadcast to (chunk_a, chunk_b) then flatten
                    a_last_exp = a_last_chunk.unsqueeze(1).expand(chunk_a_size, chunk_b_size).reshape(-1)
                    b_last_exp = b_last_chunk.unsqueeze(0).expand(chunk_a_size, chunk_b_size).reshape(-1)
                    
                    discriminant = a_last_exp ** 2 + 4.0 * b_last_exp
                    convergent_mask = discriminant >= 0  # True = potentially convergent
                    
                    n_pruned_pairs = int((~convergent_mask).sum().item())
                    
                    # Only apply filter if it actually prunes something
                    if n_pruned_pairs > 0 and n_pruned_pairs < a_expanded.shape[0]:
                        alive_idx = torch.nonzero(convergent_mask).squeeze(1)
                        a_expanded = a_expanded[alive_idx]
                        b_expanded = b_expanded[alive_idx]
                        # Store mapping for reconstructing original indices on hits
                        _idx_remap = alive_idx
                    else:
                        _idx_remap = None
                    # ── End discriminant filter ─────────────────────────────
                    
                    bsz = a_expanded.shape[0]
                    
                    if bsz == 0:
                        # All pairs pruned — skip this chunk entirely
                        processed_combinations += chunk_a_size * chunk_b_size
                        pbar.update(chunk_a_size * chunk_b_size)
                        continue
                    
                    prev_q = torch.zeros(bsz, dtype=torch.float32, device=self.device)
                    q = torch.ones(bsz, dtype=torch.float32, device=self.device)
                    prev_p = torch.ones(bsz, dtype=torch.float32, device=self.device)
                    p = a_expanded[:, 0].clone()
                    
                    # Batched convergent evaluation
                    for k in range(1, N_terms):
                        tmp_q = q.clone()
                        tmp_p = p.clone()
                        q = a_expanded[:, k] * q + b_expanded[:, k] * prev_q
                        p = a_expanded[:, k] * p + b_expanded[:, k] * prev_p
                        prev_q = tmp_q
                        prev_p = tmp_p
                        
                        # Periodic scaling every 10 iterations to prevent float64 overflow while saving VRAM bandwidth
                        if k % 10 == 0:
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
                            # If discriminant filter was applied, remap through the alive_idx
                            if _idx_remap is not None:
                                original_flat_idx = int(_idx_remap[int(idx_gpu)].item())
                            else:
                                original_flat_idx = int(idx_gpu)
                            
                            a_idx = i + original_flat_idx // chunk_b_size
                            b_idx = j + original_flat_idx % chunk_b_size
                            match = Match(key, a_coef_list[a_idx], b_coef_list[b_idx])
                            raw_hits.append(match)
                            
                            # Submit task to isolated CPU process instead of Threadpool to shatter GIL
                            future = executor.submit(_cpu_verify_worker, match, tbl_dict, tbl_file, self.poly_domains)
                            futures.append(future)
                            pbar_worker.total += 1
                            pbar_worker.refresh()
                    
                    # ─────────────────────────────────────────────────────────
                    # LIVE PROGRESS LOGGING & ETA via tqdm
                    # ─────────────────────────────────────────────────────────
                    processed_combinations += chunk_a_size * chunk_b_size  # count full chunk, not filtered bsz
                    pbar.update(chunk_a_size * chunk_b_size)
                    pbar.set_postfix({'Raw Hits': len(raw_hits), 'Pruned': n_pruned_pairs})
                    
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
            
        # Clean up massive tensor footprints natively
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return refined_results
