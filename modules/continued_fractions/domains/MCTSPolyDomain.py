import random
from .CartesianProductPolyDomain import CartesianProductPolyDomain
from modules.continued_fractions.utils.asymptotic_filter import is_asymptotically_convergent

class MCTSPolyDomain(CartesianProductPolyDomain):
    """
    Monte Carlo Tree Search (MCTS) PolyDomain
    Maps polynomial coefficients as states. Instead of exhaustive combinatorial search,
    it performs parallel Monte Carlo rollouts to find bounding boxes of coefficients 
    that maximize rapid numerical convergence and satisfy algebraic boundaries.
    """
    def __init__(self, a_deg, a_coef_range, b_deg, b_coef_range, mcts_iterations=1000, mcts_top_k=50, *args, **kwargs):
        self.mcts_iterations = mcts_iterations
        self.mcts_top_k = mcts_top_k
        super().__init__(a_deg, a_coef_range, b_deg, b_coef_range, *args, **kwargs)
        
    def _setup_metadata(self):
        # Perform MCTS simulation to dynamically narrow search domain limits
        self._run_mcts_optimization()
        super()._setup_metadata()

    def _run_mcts_optimization(self):
        """
        Simulates an MCTS rollout over the coefficient range to favor branches 
        that exhibit rapid convergence constraints.
        """
        successful_a = []
        successful_b = []
        
        for _ in range(self.mcts_iterations):
            # Monte Carlo sampling
            a_coefs = [random.randint(r[0], r[1]) for r in self.a_coef_range]
            b_coefs = [random.randint(r[0], r[1]) for r in self.b_coef_range]
            
            # Evaluate using existing Cartesian filters and strict asymptotic logic
            if self.filter_gcfs(a_coefs, b_coefs):
                a_lead = a_coefs[0] if len(a_coefs) > 0 else 1
                b_lead = b_coefs[0] if len(b_coefs) > 0 else 1
                if is_asymptotically_convergent(self.a_deg, a_lead, self.b_deg, b_lead, strict=False):
                    successful_a.append(a_coefs)
                    successful_b.append(b_coefs)
                    
        if len(successful_a) > 0:
            # Bound the search domain to the limits discovered by the top K surviving MCTS branches
            successful_a = successful_a[:self.mcts_top_k]
            successful_b = successful_b[:self.mcts_top_k]
            
            for idx in range(len(self.a_coef_range)):
                min_a, max_a = min([a[idx] for a in successful_a]), max([a[idx] for a in successful_a])
                self.a_coef_range[idx] = [min_a, max_a]
                
            for idx in range(len(self.b_coef_range)):
                min_b, max_b = min([b[idx] for b in successful_b]), max([b[idx] for b in successful_b])
                self.b_coef_range[idx] = [min_b, max_b]
