import math
import random
from .CartesianProductPolyDomain import CartesianProductPolyDomain


class MCTSNode:
    __slots__ = ['assigned_coefs', 'children', 'visits', 'wins']
    def __init__(self, assigned_coefs):
        self.assigned_coefs = assigned_coefs
        self.children = {}
        self.visits = 0
        self.wins = 0.0

    def ucb(self, total_visits, c_param=2.0):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + c_param * math.sqrt(math.log(total_visits) / self.visits)


class MCTSPolyDomain(CartesianProductPolyDomain):
    """
    Monte Carlo Tree Search (MCTS) PolyDomain
    Maps polynomial coefficients as states in a proper MCTS decision tree.
    Each level of the tree assigns a value to a specific coefficient (a_lead down to b_0).
    
    Uses UCB-1 selection, Progressive Widening for large ranges, random rollouts,
    and backpropagation to concentrate exploration on bounds that satisfy rigorous 
    asymptotic convergence constraints.
    
    Reward Signal:
      Instead of a binary pass/fail from filter_gcfs(), the reward is a GRADED
      convergence quality score. For each rolled-out leaf, we compute a short GCF
      recurrence (50 terms) and measure how stable the p/q ratio looks. This gives
      the MCTS tree a continuous signal to differentiate "barely convergent" from
      "strongly convergent" coefficient combinations.
    """
    def __init__(self, a_deg, a_coef_range, b_deg, b_coef_range, mcts_iterations=2000,
                 mcts_top_k=50, c_param=2.0, gcf_eval_depth=50, *args, **kwargs):
        self.mcts_iterations = mcts_iterations
        self.mcts_top_k = mcts_top_k
        self.c_param = c_param
        self.gcf_eval_depth = gcf_eval_depth
        super().__init__(a_deg, a_coef_range, b_deg, b_coef_range, *args, **kwargs)
        
    def _setup_metadata(self):
        self._run_mcts_tree()
        super()._setup_metadata()

    def _evaluate_gcf_convergence(self, a_coefs, b_coefs):
        """
        Compute a graded convergence quality score [0.0, 1.0] for a candidate GCF.
        
        Evaluates the GCF recurrence for gcf_eval_depth terms using the polynomial
        a(n), b(n) defined by the coefficient vectors, and measures convergence quality
        as the ratio of settled-to-oscillating final convergents.
        
        Returns:
            float: 0.0 if divergent/invalid, up to 1.0 for strongly convergent.
        """
        # Build polynomial evaluation: a(n) = sum(a_coefs[i] * n^(deg-i))
        a_deg = len(a_coefs) - 1
        b_deg = len(b_coefs) - 1
        
        def poly_a(n):
            val = 0
            for i, c in enumerate(a_coefs):
                val += c * (n ** (a_deg - i))
            return val
        
        def poly_b(n):
            val = 0
            for i, c in enumerate(b_coefs):
                val += c * (n ** (b_deg - i))
            return val
        
        # Run the standard GCF recurrence: p_n = a_n*p_{n-1} + b_n*p_{n-2}
        prev_p, p = 1.0, float(poly_a(0))
        prev_q, q = 0.0, 1.0
        
        convergents = []
        
        try:
            for n in range(1, self.gcf_eval_depth):
                a_n = float(poly_a(n))
                b_n = float(poly_b(n))
                
                new_p = a_n * p + b_n * prev_p
                new_q = a_n * q + b_n * prev_q
                prev_p, prev_q = p, q
                p, q = new_p, new_q
                
                # Periodic scaling to prevent overflow
                if abs(q) > 1e10:
                    scale = abs(q)
                    p /= scale
                    q /= scale
                    prev_p /= scale
                    prev_q /= scale
                
                if abs(q) > 1e-15:
                    ratio = p / q
                    if math.isfinite(ratio):
                        convergents.append(ratio)
                    else:
                        return 0.0  # Divergent
        except (OverflowError, ZeroDivisionError, ValueError):
            return 0.0
        
        if len(convergents) < 10:
            return 0.0  # Too few terms to judge
        
        # Graded reward: measure how settled the last quarter of convergents is
        # compared to the first quarter. A strongly convergent GCF has the ratio
        # stabilizing; a divergent one keeps oscillating or growing.
        quarter = len(convergents) // 4
        early_range = max(convergents[:quarter]) - min(convergents[:quarter]) if quarter > 0 else 1e10
        late_range = max(convergents[-quarter:]) - min(convergents[-quarter:]) if quarter > 0 else 1e10
        
        if early_range < 1e-15:
            return 0.5  # Constant from the start; mildly interesting
        
        # Convergence ratio: how much did the range shrink?
        # log-scale to get graded [0, 1] signal
        shrink_ratio = late_range / (early_range + 1e-300)
        
        if shrink_ratio >= 1.0:
            return 0.0  # Not converging (range didn't shrink)
        
        # Map shrink_ratio ∈ (0, 1) to reward ∈ (0, 1) via -log10
        try:
            score = min(1.0, -math.log10(shrink_ratio + 1e-300) / 10.0)
        except ValueError:
            score = 0.0
        
        return max(0.0, score)

    def _run_mcts_tree(self):
        """
        Executes the UCB-1 MCTS loop with Progressive Widening.
        Constructs tight bounding boxes from the most successful rolled-out leaves.
        """
        all_ranges = self.a_coef_range + self.b_coef_range
        total_depth = len(all_ranges)
        
        root = MCTSNode(assigned_coefs=tuple())
        successful_leaves = []
        
        for _ in range(self.mcts_iterations):
            # 1. SELECT & PROGRESSIVE WIDENING
            node = root
            path = [node]
            depth = 0
            
            while depth < total_depth:
                coef_range = all_ranges[depth]
                range_size = coef_range[1] - coef_range[0] + 1
                
                target_children = math.ceil(2.0 * math.sqrt(node.visits + 1))
                target_children = min(target_children, range_size)
                
                if len(node.children) < target_children:
                    for _ in range(50):
                        a = random.randint(coef_range[0], coef_range[1])
                        if a not in node.children:
                            new_coefs = node.assigned_coefs + (a,)
                            child = MCTSNode(assigned_coefs=new_coefs)
                            node.children[a] = child
                            node = child
                            path.append(node)
                            depth += 1
                            break
                    break
                else:
                    best_child = None
                    best_score = -float('inf')
                    for child in node.children.values():
                        score = child.ucb(node.visits, self.c_param)
                        if score > best_score:
                            best_score = score
                            best_child = child
                    
                    if best_child is None:
                        break
                        
                    node = best_child
                    path.append(node)
                    depth += 1
            
            # 2. ROLLOUT (Random Simulation to leaf)
            rollout_coefs = list(node.assigned_coefs)
            for d in range(depth, total_depth):
                r = all_ranges[d]
                rollout_coefs.append(random.randint(r[0], r[1]))
                
            # 3. EVALUATE LEAF — Graded GCF convergence reward (not binary filter)
            a_len = len(self.a_coef_range)
            a_c = rollout_coefs[:a_len]
            b_c = rollout_coefs[a_len:]
            
            reward = 0.0
            # First gate: pass the algebraic convergence filter (Poincaré)
            if self.filter_gcfs(a_c, b_c):
                # Second gate: graded numerical convergence quality
                reward = self._evaluate_gcf_convergence(a_c, b_c)
                if reward > 0.1:
                    successful_leaves.append((a_c, b_c, reward))
                    
            # 4. BACKPROPAGATE
            for n in path:
                n.visits += 1
                n.wins += reward
                
        # 5. AGGREGATE BOUNDS
        if len(successful_leaves) > 0:
            # Sort by convergence quality (best first) and take top K
            successful_leaves.sort(key=lambda x: x[2], reverse=True)
            top_leaves = successful_leaves[:self.mcts_top_k]
            
            a_len = len(self.a_coef_range)
            success_a = [leaf[0] for leaf in top_leaves]
            success_b = [leaf[1] for leaf in top_leaves]
            
            for idx in range(len(self.a_coef_range)):
                min_a = min(a[idx] for a in success_a)
                max_a = max(a[idx] for a in success_a)
                self.a_coef_range[idx] = [min_a, max_a]
                
            for idx in range(len(self.b_coef_range)):
                min_b = min(b[idx] for b in success_b)
                max_b = max(b[idx] for b in success_b)
                self.b_coef_range[idx] = [min_b, max_b]
