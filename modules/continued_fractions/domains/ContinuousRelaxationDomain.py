import torch
import torch.optim as optim
from .CartesianProductPolyDomain import CartesianProductPolyDomain

class ContinuousRelaxationDomain(CartesianProductPolyDomain):
    """
    Experimental ContinuousRelaxationDomain.
    Relaxes the discrete integer search space for coefficients to the continuous real domain.
    Employs gradient descent optimization (via PyTorch) to minimize the evaluation loss to 
    satisfy necessary algebraic convergence conditions, then applies lattice rounding 
    to snap backward to nearest integers.
    
    Usage:
        ContinuousRelaxationDomain(a_deg=2, a_coef_range=[-10, 10], 
                                   b_deg=2, b_coef_range=[-10, 10],
                                   lr=0.1, epochs=100)
    
    a_coef_range and b_coef_range should be flat [min, max] lists, 
    exactly like CartesianProductPolyDomain expects.
    """
    def __init__(self, a_deg, a_coef_range, b_deg, b_coef_range, lr=0.1, epochs=100, *args, **kwargs):
        self.lr = lr
        self.epochs = epochs
        # The parent __init__ expands flat a_coef_range=[-10,10] into
        # self.a_coef_range = [[-10,10], [-10,10], [-10,10]] (one per coefficient).
        # Then it calls _setup_metadata(), which we override to run gradient descent first.
        super().__init__(a_deg, a_coef_range, b_deg, b_coef_range, *args, **kwargs)
        
    def _setup_metadata(self):
        # Run gradient descent optimization to shrink the per-coefficient bounds
        # BEFORE the parent computes domain sizes and iterators
        self._run_gradient_descent()
        super()._setup_metadata()

    def _run_gradient_descent(self):
        """
        Differentiable relaxation over the polynomial coefficient constraints.
        After the parent __init__ expands self.a_coef_range into per-coefficient ranges,
        this method optimizes those ranges using PyTorch gradient descent to shrink the 
        bounding box toward the region satisfying the Worpitzky convergence condition:
            4*b_leading + a_leading^2 > 0
        """
        # At this point self.a_coef_range is already expanded by the parent, e.g.:
        # [[-10, 10], [-10, 10], [-10, 10]] for degree 2
        # Run optimization on GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        a_bounds = torch.tensor(self.a_coef_range, dtype=torch.float32, device=device, requires_grad=True)
        b_bounds = torch.tensor(self.b_coef_range, dtype=torch.float32, device=device, requires_grad=True)
        
        optimizer = optim.Adam([a_bounds, b_bounds], lr=self.lr)
        
        for _ in range(self.epochs):
            optimizer.zero_grad()
            
            # Mean of each coefficient's [min, max] range as proxy for likely value
            a_mean = a_bounds.mean(dim=1)
            b_mean = b_bounds.mean(dim=1)
            
            # Loss: penalty for violating convergence condition 4*b[0] + a[0]^2 > 0
            # Using the leading coefficient index (index 0 = highest degree)
            a_lead = a_mean[0] if len(a_mean) > 0 else torch.tensor(1.0)
            b_lead = b_mean[0] if len(b_mean) > 0 else torch.tensor(1.0)
            
            margin = 4 * b_lead + a_lead**2
            loss = torch.relu(1.0 - margin)  # Push margin to be at least 1.0
            
            # Add small regularization to shrink the bounding width slightly 
            loss = loss + 0.05 * (a_bounds[:, 1] - a_bounds[:, 0]).sum()
            loss = loss + 0.05 * (b_bounds[:, 1] - b_bounds[:, 0]).sum()
            
            loss.backward()
            optimizer.step()
            
            # Enforce min <= max during descent (using .data to avoid autograd issues)
            with torch.no_grad():
                a_bounds.data[:, 0] = torch.min(a_bounds.data[:, 0], a_bounds.data[:, 1] - 0.1)
                b_bounds.data[:, 0] = torch.min(b_bounds.data[:, 0], b_bounds.data[:, 1] - 0.1)
                
        # Snap back to integers
        self.a_coef_range = [[int(torch.floor(r[0]).item()), int(torch.ceil(r[1]).item())] for r in a_bounds]
        self.b_coef_range = [[int(torch.floor(r[0]).item()), int(torch.ceil(r[1]).item())] for r in b_bounds]
