"""
cfuse_torch.py - PyTorch integration for cFUSE with proper gradient flow

UPDATED: Now supports SPATIALLY VARYING parameters efficiently!
- Forward: C++ batch natively handles both shared [n_params] and per-HRU [n_hrus, n_params]
- Backward: Enzyme AD for shared, loop for spatial (until C++ gradient fn extended)

Usage:
    from cfuse.torch import DifferentiableFUSEBatch
    
    # Shared params [n_params]
    runoff = DifferentiableFUSEBatch.apply(
        params, initial_states, forcing, config_dict, dt
    )
    
    # Spatial params [n_hrus, n_params] - SAME CALL, forward is efficient!
    runoff = DifferentiableFUSEBatch.apply(
        spatial_params, initial_states, forcing, config_dict, dt
    )
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Import cFUSE core
try:
    import cfuse_core
    HAS_CFUSE = True
    HAS_ENZYME = getattr(cfuse_core, 'HAS_ENZYME', False)
except ImportError:
    HAS_CFUSE = False
    HAS_ENZYME = False


class DifferentiableFUSEBatch(torch.autograd.Function):
    """
    PyTorch autograd Function for batch cFUSE with proper gradient flow.
    
    Supports BOTH shared and spatially-varying parameters:
    - params [n_params]: Shared across all HRUs
    - params [n_hrus, n_params]: Per-HRU parameters
    
    Forward: Uses C++ batch (handles BOTH cases natively - no Python loop!)
    Backward: Enzyme AD for shared, loop over HRUs for spatial
    """
    
    @staticmethod
    def forward(ctx, 
                params: torch.Tensor,
                initial_states: torch.Tensor, 
                forcing: torch.Tensor,
                config_dict: Dict[str, Any],
                dt: float) -> torch.Tensor:
        """
        Forward pass through cFUSE for multiple HRUs.
        
        Args:
            params: [n_params] shared OR [n_hrus, n_params] per-HRU
            initial_states: [n_hrus, n_states] Initial state per HRU
            forcing: [n_timesteps, n_hrus, 3] Forcing (precip, pet, temp)
            config_dict: FUSE model configuration
            dt: Timestep in days
            
        Returns:
            runoff: [n_timesteps, n_hrus] Runoff in mm/timestep
        """
        ctx.config_dict = config_dict
        ctx.dt = dt
        ctx.spatial_params = (params.dim() == 2)  # Track if spatial
        
        # Convert to numpy - C-contiguous for C++
        params_np = np.ascontiguousarray(params.detach().cpu().numpy().astype(np.float32))
        states_np = np.ascontiguousarray(initial_states.detach().cpu().numpy().astype(np.float32))
        forcing_np = np.ascontiguousarray(forcing.detach().cpu().numpy().astype(np.float32))
        
        # Run C++ forward pass - handles both shared and per-HRU params natively!
        # No Python loop needed - C++ iterates over HRUs with OpenMP
        final_states, runoff = cfuse_core.run_fuse_batch(
            states_np, forcing_np, params_np, config_dict, float(dt)
        )
        
        ctx.save_for_backward(params, initial_states, forcing)
        ctx.device = params.device
        ctx.dtype = params.dtype
        
        return torch.from_numpy(runoff.astype(np.float32)).to(params.device)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using Enzyme AD.
        
        For shared params: Single call to run_fuse_batch_gradient
        For spatial params: Loop over HRUs (gradient fn doesn't support spatial yet)
        """
        params, initial_states, forcing = ctx.saved_tensors
        
        params_np = np.ascontiguousarray(params.detach().cpu().numpy().astype(np.float32))
        states_np = np.ascontiguousarray(initial_states.detach().cpu().numpy().astype(np.float32))
        forcing_np = np.ascontiguousarray(forcing.detach().cpu().numpy().astype(np.float32))
        grad_np = np.ascontiguousarray(grad_output.detach().cpu().numpy().astype(np.float32))
        
        if not ctx.spatial_params:
            # Shared params - single efficient call
            if HAS_ENZYME and hasattr(cfuse_core, 'run_fuse_batch_gradient'):
                grad_params_np = cfuse_core.run_fuse_batch_gradient(
                    states_np, forcing_np, params_np, grad_np,
                    ctx.config_dict, float(ctx.dt)
                )
            else:
                grad_params_np = cfuse_core.run_fuse_batch_gradient_numerical(
                    states_np, forcing_np, params_np, grad_np,
                    ctx.config_dict, float(ctx.dt)
                )
            grad_params = torch.from_numpy(grad_params_np.astype(np.float32))
        else:
            # Spatial params - compute gradient per HRU
            # Note: backward is only called once per epoch, so loop is acceptable
            n_hrus, n_params = params.shape
            grad_params_np = np.zeros((n_hrus, n_params), dtype=np.float32)
            
            for h in range(n_hrus):
                # Extract single HRU data
                hru_states = np.ascontiguousarray(states_np[h:h+1, :])   # [1, n_states]
                hru_forcing = np.ascontiguousarray(forcing_np[:, h:h+1, :])  # [n_timesteps, 1, 3]
                hru_params = np.ascontiguousarray(params_np[h, :])       # [n_params]
                hru_grad = np.ascontiguousarray(grad_np[:, h:h+1])       # [n_timesteps, 1]
                
                # Compute gradient for this HRU
                if HAS_ENZYME and hasattr(cfuse_core, 'run_fuse_batch_gradient'):
                    hru_grad_params = cfuse_core.run_fuse_batch_gradient(
                        hru_states, hru_forcing, hru_params, hru_grad,
                        ctx.config_dict, float(ctx.dt)
                    )
                else:
                    hru_grad_params = cfuse_core.run_fuse_batch_gradient_numerical(
                        hru_states, hru_forcing, hru_params, hru_grad,
                        ctx.config_dict, float(ctx.dt)
                    )
                grad_params_np[h, :] = hru_grad_params
            
            grad_params = torch.from_numpy(grad_params_np)
        
        grad_params = grad_params.to(ctx.device, dtype=ctx.dtype)
        return grad_params, None, None, None, None


class FUSEModule(nn.Module):
    """
    PyTorch Module wrapper for FUSE with learnable parameters.
    
    Supports both shared and spatially-varying parameters.
    """
    
    def __init__(self, config, n_hrus: int, dt: float, 
                 param_bounds: Dict[str, Tuple[float, float]],
                 spatial_params: bool = False):
        """
        Initialize FUSE module.
        
        Args:
            config: FUSE ModelConfig object or dict
            n_hrus: Number of HRUs
            dt: Timestep in days
            param_bounds: Dict of {param_name: (lower, upper)}
            spatial_params: If True, each HRU has its own parameters
        """
        super().__init__()
        
        self.config = config
        self.config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
        self.n_hrus = n_hrus
        self.dt = dt
        self.spatial_params = spatial_params
        
        # Setup parameter bounds
        self.param_names = list(param_bounds.keys())
        n_params = len(self.param_names)
        
        lowers = torch.tensor([param_bounds[n][0] for n in self.param_names], dtype=torch.float32)
        uppers = torch.tensor([param_bounds[n][1] for n in self.param_names], dtype=torch.float32)
        self.register_buffer('param_lower', lowers)
        self.register_buffer('param_upper', uppers)
        
        # Raw (unconstrained) parameters
        if spatial_params:
            # [n_hrus, n_params] - small random init for diversity
            init = torch.randn(n_hrus, n_params) * 0.2
            self.raw_params = nn.Parameter(init)
        else:
            # [n_params]
            self.raw_params = nn.Parameter(torch.zeros(n_params))
        
        # Number of states
        if HAS_CFUSE:
            self.n_states = cfuse_core.get_num_active_states(self.config_dict)
        else:
            self.n_states = 10  # Default
    
    def get_physical_params(self, hru_idx: Optional[int] = None) -> torch.Tensor:
        """
        Transform raw parameters to physical space using sigmoid.
        
        Args:
            hru_idx: If spatial and specified, return params for single HRU
            
        Returns:
            Physical parameters
        """
        if self.spatial_params:
            # Broadcasting: [n_hrus, n_params] with [n_params] bounds
            phys = self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(self.raw_params)
            if hru_idx is not None:
                return phys[hru_idx]
            return phys
        else:
            return self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(self.raw_params)
    
    def get_initial_state(self) -> torch.Tensor:
        """Get default initial states."""
        state = torch.zeros(self.n_hrus, self.n_states)
        state[:, 0] = 50.0  # Upper tension storage
        if self.n_states > 1:
            state[:, 1] = 20.0  # Upper free storage
        if self.n_states > 2:
            state[:, 2] = 200.0  # Lower storage
        return state
    
    def forward(self, forcing: torch.Tensor, 
                initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run FUSE for all HRUs.
        
        Args:
            forcing: [n_timesteps, n_hrus, 3] Forcing data
            initial_state: [n_hrus, n_states] Initial states (optional)
            
        Returns:
            runoff: [n_timesteps, n_hrus] Runoff in mm/timestep
        """
        if initial_state is None:
            initial_state = self.get_initial_state().to(forcing.device)
        
        phys_params = self.get_physical_params()
        
        return DifferentiableFUSEBatch.apply(
            phys_params, initial_state, forcing,
            self.config_dict, self.dt
        )
    
    def get_param_dict(self) -> Dict[str, float]:
        """Get current physical parameters as dict (mean if spatial)."""
        phys = self.get_physical_params()
        if self.spatial_params:
            return {name: phys[:, i].mean().item() for i, name in enumerate(self.param_names)}
        else:
            return {name: phys[i].item() for i, name in enumerate(self.param_names)}
    
    def get_param_stats(self) -> Dict[str, Dict[str, float]]:
        """Get parameter statistics (for spatial params)."""
        phys = self.get_physical_params()
        if not self.spatial_params:
            return {name: {'value': phys[i].item()} for i, name in enumerate(self.param_names)}
        
        stats = {}
        for i, name in enumerate(self.param_names):
            vals = phys[:, i]
            stats[name] = {
                'mean': vals.mean().item(),
                'std': vals.std().item(),
                'min': vals.min().item(),
                'max': vals.max().item()
            }
        return stats


def gradient_check(config_dict: Dict, n_hrus: int = 5, n_timesteps: int = 100, 
                   eps: float = 1e-4, verbose: bool = True,
                   spatial_params: bool = False) -> bool:
    """
    Verify gradient computation by comparing AD with finite differences.
    
    Args:
        config_dict: FUSE model configuration
        n_hrus: Number of HRUs for test
        n_timesteps: Number of timesteps
        eps: Finite difference epsilon
        verbose: Print results
        spatial_params: Test with spatial params
        
    Returns:
        True if gradients match within tolerance
    """
    n_params = cfuse_core.NUM_PARAMETERS
    n_states = cfuse_core.get_num_active_states(config_dict)
    
    # Create test data
    np.random.seed(42)
    states = np.zeros((n_hrus, n_states), dtype=np.float32)
    states[:, 0] = 50.0
    
    forcing = np.random.rand(n_timesteps, n_hrus, 3).astype(np.float32)
    forcing[:, :, 0] *= 20  # Precip 0-20 mm
    forcing[:, :, 1] *= 5   # PET 0-5 mm
    forcing[:, :, 2] = forcing[:, :, 2] * 20 - 5  # Temp -5 to 15 C
    
    if spatial_params:
        params = np.random.rand(n_hrus, n_params).astype(np.float32) * 0.5 + 0.25
    else:
        params = np.random.rand(n_params).astype(np.float32) * 0.5 + 0.25
    
    grad_runoff = np.random.rand(n_timesteps, n_hrus).astype(np.float32)
    
    # Convert to torch
    params_t = torch.tensor(params, requires_grad=True)
    states_t = torch.tensor(states)
    forcing_t = torch.tensor(forcing)
    grad_t = torch.tensor(grad_runoff)
    
    # Forward
    runoff = DifferentiableFUSEBatch.apply(
        params_t, states_t, forcing_t, config_dict, 1.0
    )
    
    # Backward
    runoff.backward(grad_t)
    ad_grad = params_t.grad.numpy()
    
    # Numerical gradient (finite differences)
    if spatial_params:
        num_grad = np.zeros_like(params)
        for h in range(n_hrus):
            for i in range(n_params):
                params_plus = params.copy()
                params_plus[h, i] += eps
                _, runoff_plus = cfuse_core.run_fuse_batch(
                    states, forcing, params_plus, config_dict, 1.0
                )
                
                params_minus = params.copy()
                params_minus[h, i] -= eps
                _, runoff_minus = cfuse_core.run_fuse_batch(
                    states, forcing, params_minus, config_dict, 1.0
                )
                
                num_grad[h, i] = np.sum((runoff_plus - runoff_minus) * grad_runoff) / (2 * eps)
    else:
        num_grad = cfuse_core.run_fuse_batch_gradient_numerical(
            states, forcing, params, grad_runoff, config_dict, 1.0, eps
        )
    
    # Compare
    abs_diff = np.abs(ad_grad - num_grad)
    rel_diff = abs_diff / (np.abs(num_grad) + 1e-8)
    
    max_rel = rel_diff.max()
    mean_rel = rel_diff.mean()
    
    passed = max_rel < 0.1  # 10% tolerance
    
    if verbose:
        print(f"Gradient Check Results ({'spatial' if spatial_params else 'shared'}):")
        print(f"  Max relative error: {max_rel:.4f}")
        print(f"  Mean relative error: {mean_rel:.4f}")
        print(f"  Status: {'PASSED' if passed else 'FAILED'}")
    
    return passed
