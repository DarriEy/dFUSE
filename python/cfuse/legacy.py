"""
cFUSE: Differentiable Framework for Understanding Structural Errors

A GPU-native, differentiable implementation of the FUSE hydrological model
framework (Clark et al., 2008 WRR).

This module provides:
- PyTorch-compatible differentiable model layer
- Multiple physics configurations (PRMS, Sacramento, TOPMODEL, VIC)
- GPU acceleration via CUDA
- Automatic differentiation for parameter optimization
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import IntEnum

# Try to import compiled C++ module
try:
    import cfuse_core
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    print("Warning: cfuse_core not found, using pure Python fallback")


# ============================================================================
# CONFIGURATION ENUMS
# ============================================================================

class UpperLayerArch(IntEnum):
    """Upper soil layer architecture options"""
    SINGLE_STATE = 0     # Single state S1 (TOPMODEL, VIC style)
    TENSION_FREE = 1     # Separate tension and free storage (Sacramento)
    TENSION2_FREE = 2    # Two tension stores + free (PRMS)


class LowerLayerArch(IntEnum):
    """Lower soil layer architecture options"""
    SINGLE_NOEVAP = 0    # Single reservoir, no evaporation (TOPMODEL/PRMS)
    SINGLE_EVAP = 1      # Single reservoir with evaporation (VIC)
    TENSION_2RESERV = 2  # Tension + two parallel reservoirs (Sacramento)


class BaseflowType(IntEnum):
    """Baseflow parameterization options"""
    LINEAR = 0           # Linear reservoir (PRMS)
    PARALLEL_LINEAR = 1  # Two parallel linear reservoirs (Sacramento)
    NONLINEAR = 2        # Nonlinear power function (VIC)
    TOPMODEL = 3         # TOPMODEL power law transmissivity


class PercolationType(IntEnum):
    """Percolation parameterization options"""
    TOTAL_STORAGE = 0    # Based on total upper zone storage (VIC)
    FREE_STORAGE = 1     # Based on free storage (PRMS)
    LOWER_DEMAND = 2     # Driven by lower zone demand (Sacramento)


class SurfaceRunoffType(IntEnum):
    """Surface runoff / saturated area options"""
    UZ_LINEAR = 0        # Linear function of tension storage (PRMS)
    UZ_PARETO = 1        # Pareto distribution / VIC 'b' curve
    LZ_GAMMA = 2         # TOPMODEL topographic index distribution


class EvaporationType(IntEnum):
    """Evaporation parameterization options"""
    SEQUENTIAL = 0       # Sequential: upper layer first, then lower
    ROOT_WEIGHT = 1      # Root weighting between layers


class InterflowType(IntEnum):
    """Interflow parameterization options"""
    NONE = 0             # No interflow
    LINEAR = 1           # Linear function of free storage


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class FUSEConfig:
    """Complete FUSE model configuration"""
    upper_arch: UpperLayerArch = UpperLayerArch.SINGLE_STATE
    lower_arch: LowerLayerArch = LowerLayerArch.SINGLE_NOEVAP
    baseflow: BaseflowType = BaseflowType.LINEAR
    percolation: PercolationType = PercolationType.TOTAL_STORAGE
    surface_runoff: SurfaceRunoffType = SurfaceRunoffType.UZ_LINEAR
    evaporation: EvaporationType = EvaporationType.SEQUENTIAL
    interflow: InterflowType = InterflowType.NONE
    enable_snow: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for C++ interface"""
        return {
            'upper_arch': int(self.upper_arch),
            'lower_arch': int(self.lower_arch),
            'baseflow': int(self.baseflow),
            'percolation': int(self.percolation),
            'surface_runoff': int(self.surface_runoff),
            'evaporation': int(self.evaporation),
            'interflow': int(self.interflow),
            'enable_snow': self.enable_snow
        }
    
    @property
    def num_upper_states(self) -> int:
        """Number of state variables in upper layer"""
        if self.upper_arch == UpperLayerArch.SINGLE_STATE:
            return 1
        elif self.upper_arch == UpperLayerArch.TENSION_FREE:
            return 2
        else:
            return 3
    
    @property
    def num_lower_states(self) -> int:
        """Number of state variables in lower layer"""
        if self.lower_arch == LowerLayerArch.TENSION_2RESERV:
            return 3
        else:
            return 1
    
    @property
    def num_states(self) -> int:
        """Total number of state variables"""
        n = self.num_upper_states + self.num_lower_states
        if self.enable_snow:
            n += 1
        return n


# Predefined configurations for parent models
PRMS_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.TENSION2_FREE,
    lower_arch=LowerLayerArch.SINGLE_NOEVAP,
    baseflow=BaseflowType.LINEAR,
    percolation=PercolationType.FREE_STORAGE,
    surface_runoff=SurfaceRunoffType.UZ_LINEAR,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.LINEAR
)

SACRAMENTO_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.TENSION_FREE,
    lower_arch=LowerLayerArch.TENSION_2RESERV,
    baseflow=BaseflowType.PARALLEL_LINEAR,
    percolation=PercolationType.LOWER_DEMAND,
    surface_runoff=SurfaceRunoffType.UZ_LINEAR,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.NONE
)

TOPMODEL_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.SINGLE_STATE,
    lower_arch=LowerLayerArch.SINGLE_NOEVAP,
    baseflow=BaseflowType.TOPMODEL,
    percolation=PercolationType.TOTAL_STORAGE,
    surface_runoff=SurfaceRunoffType.LZ_GAMMA,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.NONE
)

VIC_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.SINGLE_STATE,
    lower_arch=LowerLayerArch.SINGLE_EVAP,
    baseflow=BaseflowType.NONLINEAR,
    percolation=PercolationType.TOTAL_STORAGE,
    surface_runoff=SurfaceRunoffType.UZ_PARETO,
    evaporation=EvaporationType.ROOT_WEIGHT,
    interflow=InterflowType.NONE
)


# ============================================================================
# PARAMETER BOUNDS
# ============================================================================

# Parameter indices
PARAM_NAMES = [
    'S1_max', 'S2_max', 'f_tens', 'f_rchr', 'f_base', 'r1',
    'ku', 'c', 'alpha', 'psi', 'kappa', 'ki',
    'ks', 'n', 'v', 'v_A', 'v_B',
    'Ac_max', 'b', 'lambda', 'chi', 'mu_t',
    'T_rain', 'T_melt', 'melt_rate', 'lapse_rate', 'opg'
]

# Default parameter bounds from Clark et al. (2008) Table 3
PARAM_BOUNDS = {
    'S1_max': (50.0, 5000.0),
    'S2_max': (100.0, 10000.0),
    'f_tens': (0.05, 0.95),
    'f_rchr': (0.05, 0.95),
    'f_base': (0.05, 0.95),
    'r1': (0.05, 0.95),
    'ku': (0.01, 1000.0),
    'c': (1.0, 20.0),
    'alpha': (1.0, 250.0),
    'psi': (1.0, 5.0),
    'kappa': (0.05, 0.95),
    'ki': (0.01, 1000.0),
    'ks': (0.001, 10000.0),
    'n': (1.0, 10.0),
    'v': (0.001, 0.25),
    'v_A': (0.001, 0.25),
    'v_B': (0.001, 0.25),
    'Ac_max': (0.05, 0.95),
    'b': (0.001, 3.0),
    'lambda': (5.0, 10.0),
    'chi': (2.0, 5.0),
    'mu_t': (0.01, 5.0),
    'T_rain': (-2.0, 4.0),
    'T_melt': (-2.0, 4.0),
    'melt_rate': (1.0, 10.0),
    'lapse_rate': (-7.0, -4.0),
    'opg': (0.0, 1.0)
}


def get_default_params() -> torch.Tensor:
    """Get default parameter values (raw values that give midpoint when constrained)
    
    Since we use sigmoid constraint: constrained = lo + (hi - lo) * sigmoid(raw)
    To get midpoint: sigmoid(raw) = 0.5, so raw = 0
    """
    # All zeros gives sigmoid(0) = 0.5, which maps to midpoint of bounds
    return torch.zeros(len(PARAM_NAMES), dtype=torch.float32)


def get_default_constrained_params() -> torch.Tensor:
    """Get default constrained parameter values (midpoint of bounds)"""
    params = []
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        params.append((lo + hi) / 2)
    return torch.tensor(params, dtype=torch.float32)


def constrain_params(params: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid constraints to keep parameters in valid bounds
    
    Maps raw_params ∈ (-∞, +∞) to constrained ∈ (lo, hi) via:
        constrained = lo + (hi - lo) * sigmoid(raw)
    
    This is differentiable with gradient:
        d(constrained)/d(raw) = (hi - lo) * sigmoid(raw) * (1 - sigmoid(raw))
    """
    bounds = torch.tensor(
        [(PARAM_BOUNDS[name][0], PARAM_BOUNDS[name][1]) for name in PARAM_NAMES],
        dtype=params.dtype, device=params.device
    )
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + (hi - lo) * torch.sigmoid(params)


# ============================================================================
# PYTORCH AUTOGRAD FUNCTION
# ============================================================================

class FUSEFunction(Function):
    """
    PyTorch autograd function for differentiable FUSE model.
    
    Supports both forward and backward passes through the hydrological model.
    Uses adjoint method for efficient O(1) gradient computation.
    """
    
    @staticmethod
    def forward(ctx, initial_state, forcing, params, config_dict, dt, use_adjoint=True):
        """
        Forward pass through FUSE model.
        
        Args:
            initial_state: Initial model state [num_states]
            forcing: Forcing data [n_timesteps, 3] (precip, pet, temp)
            params: Model parameters [num_params]
            config_dict: Model configuration dictionary
            dt: Timestep in days
            use_adjoint: If True, store state trajectory for adjoint backward
            
        Returns:
            runoff: Total runoff [n_timesteps]
            
        Note:
            Currently forces GPU->CPU synchronization for C++ interface.
            Future versions will support direct GPU tensor passing via DLPack
            or torch::extension for zero-copy data transfer.
        """
        # Convert to numpy for C++ interface
        # TODO: Use DLPack or torch.utils.cpp_extension to avoid GPU->CPU copy
        # when initial_state.is_cuda == True
        state_np = initial_state.detach().cpu().numpy().astype(np.float32)
        forcing_np = forcing.detach().cpu().numpy().astype(np.float32)
        params_np = params.detach().cpu().numpy().astype(np.float32)
        
        if HAS_NATIVE and use_adjoint:
            # Use forward with trajectory for adjoint backward
            final_state, runoff, state_trajectory = cfuse_core.run_fuse_forward_with_trajectory(
                state_np, forcing_np, params_np, config_dict, dt
            )
            ctx.state_trajectory = state_trajectory
        elif HAS_NATIVE:
            # Regular forward
            final_state, runoff = cfuse_core.run_fuse(
                state_np, forcing_np, params_np, config_dict, dt
            )
            ctx.state_trajectory = None
        else:
            # Pure Python fallback
            final_state, runoff = _run_fuse_python(
                state_np, forcing_np, params_np, config_dict, dt
            )
            ctx.state_trajectory = None
        
        # Save for backward
        ctx.save_for_backward(initial_state, forcing, params)
        ctx.config_dict = config_dict
        ctx.dt = dt
        ctx.use_adjoint = use_adjoint and HAS_NATIVE
        ctx.final_state = torch.from_numpy(final_state)
        
        return torch.from_numpy(runoff).to(initial_state.device)
    
    @staticmethod
    def backward(ctx, grad_runoff):
        """
        Backward pass using adjoint method (fast) or numerical differentiation (fallback).
        """
        initial_state, forcing, params = ctx.saved_tensors
        config_dict = ctx.config_dict
        dt = ctx.dt
        
        # Convert to numpy
        state_np = initial_state.detach().cpu().numpy().astype(np.float32)
        forcing_np = forcing.detach().cpu().numpy().astype(np.float32)
        params_np = params.detach().cpu().numpy().astype(np.float32)
        grad_runoff_np = grad_runoff.detach().cpu().numpy().astype(np.float32)
        
        if ctx.use_adjoint and ctx.state_trajectory is not None:
            # Fast adjoint method - O(1) backward pass
            grad_params_np = cfuse_core.compute_gradient_adjoint(
                state_np, forcing_np, params_np, grad_runoff_np,
                ctx.state_trajectory, config_dict, dt
            )
            # Validate gradients - fall back to numerical if adjoint fails
            if not np.all(np.isfinite(grad_params_np)):
                # Adjoint returned inf/nan, fall back to numerical
                grad_params_np = cfuse_core.compute_gradient_numerical(
                    state_np, forcing_np, params_np, grad_runoff_np, 
                    config_dict, dt, 1e-4
                )
            grad_params = torch.from_numpy(grad_params_np).to(params.device)
        elif HAS_NATIVE:
            # Numerical differentiation in C++ - O(n_params) but still fast
            grad_params_np = cfuse_core.compute_gradient_numerical(
                state_np, forcing_np, params_np, grad_runoff_np, 
                config_dict, dt, 1e-4
            )
            grad_params = torch.from_numpy(grad_params_np).to(params.device)
        else:
            # Slow Python fallback
            eps = 1e-4
            grad_params = torch.zeros_like(params)
            
            for i in range(len(params)):
                params_plus = params.clone()
                params_plus[i] += eps
                params_plus_np = params_plus.detach().cpu().numpy().astype(np.float32)
                _, runoff_plus = _run_fuse_python(
                    state_np, forcing_np, params_plus_np, config_dict, dt
                )
                
                params_minus = params.clone()
                params_minus[i] -= eps
                params_minus_np = params_minus.detach().cpu().numpy().astype(np.float32)
                _, runoff_minus = _run_fuse_python(
                    state_np, forcing_np, params_minus_np, config_dict, dt
                )
                
                d_runoff = (runoff_plus - runoff_minus) / (2 * eps)
                grad_params[i] = (d_runoff * grad_runoff_np).sum()
        
        # Gradient w.r.t. initial state (not implemented yet)
        grad_state = torch.zeros_like(initial_state)
        
        return grad_state, None, grad_params, None, None, None


# ============================================================================
# PYTORCH MODULE
# ============================================================================

class FUSE(nn.Module):
    """
    Differentiable FUSE hydrological model as PyTorch module.
    
    This module wraps the FUSE model for use in PyTorch training loops.
    Parameters can be optimized via gradient descent.
    
    Features:
    - Adjoint method for fast O(1) gradient computation
    - Batch execution for multi-basin simulations
    - GPU acceleration via CUDA (when available)
    
    Example:
        >>> model = FUSE(config=TOPMODEL_CONFIG)
        >>> runoff = model(initial_state, forcing)
        >>> loss = mse_loss(runoff, observed)
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        config: FUSEConfig = None,
        dt: float = 1.0,
        learnable_params: bool = True,
        initial_params: torch.Tensor = None,
        use_adjoint: bool = False  # Numerical gradients by default (adjoint is experimental)
    ):
        """
        Initialize FUSE model.
        
        Args:
            config: Model configuration (default: VIC-style)
            dt: Timestep in days (default: 1.0)
            learnable_params: Whether parameters are learnable
            initial_params: Initial parameter values (default: midpoint of bounds)
            use_adjoint: Use adjoint method for gradients (faster, default: True)
        """
        super().__init__()
        
        self.config = config or VIC_CONFIG
        self.config_dict = self.config.to_dict()
        self.dt = dt
        self.use_adjoint = use_adjoint
        
        # Initialize parameters
        if initial_params is None:
            initial_params = get_default_params()
        
        if learnable_params:
            # Store as unconstrained parameters, apply constraint in forward
            self.raw_params = nn.Parameter(initial_params.clone())
        else:
            self.register_buffer('raw_params', initial_params)
    
    @property
    def params(self) -> torch.Tensor:
        """Get constrained parameter values"""
        return constrain_params(self.raw_params)
    
    def get_initial_state(
        self,
        S1_init: float = 100.0,
        S2_init: float = 500.0,
        SWE_init: float = 0.0
    ) -> torch.Tensor:
        """
        Get initial state tensor.
        
        Args:
            S1_init: Initial upper layer storage (mm)
            S2_init: Initial lower layer storage (mm)
            SWE_init: Initial snow water equivalent (mm)
            
        Returns:
            state: Initial state tensor
        """
        state = torch.zeros(self.config.num_states)
        
        # Set based on architecture
        if self.config.upper_arch == UpperLayerArch.SINGLE_STATE:
            state[0] = S1_init
        elif self.config.upper_arch == UpperLayerArch.TENSION_FREE:
            state[0] = S1_init * 0.7  # Tension
            state[1] = S1_init * 0.3  # Free
        else:  # TENSION2_FREE
            state[0] = S1_init * 0.5  # Primary tension
            state[1] = S1_init * 0.2  # Secondary tension
            state[2] = S1_init * 0.3  # Free
        
        # Lower layer
        idx = self.config.num_upper_states
        if self.config.lower_arch == LowerLayerArch.TENSION_2RESERV:
            state[idx] = S2_init * 0.3     # Tension
            state[idx + 1] = S2_init * 0.5 # Primary reservoir
            state[idx + 2] = S2_init * 0.2 # Secondary reservoir
        else:
            state[idx] = S2_init
        
        # Snow
        if self.config.enable_snow:
            state[-1] = SWE_init
        
        return state
    
    def forward(
        self,
        initial_state: torch.Tensor,
        forcing: torch.Tensor
    ) -> torch.Tensor:
        """
        Run model forward.
        
        Args:
            initial_state: Initial state [num_states] or [batch, num_states]
            forcing: Forcing data [n_timesteps, 3] or [batch, n_timesteps, 3]
                     Columns: precipitation, potential ET, temperature (for snow)
                     
        Returns:
            runoff: Total runoff [n_timesteps] or [batch, n_timesteps]
        """
        # Handle batched input
        if initial_state.dim() == 1:
            return FUSEFunction.apply(
                initial_state, forcing, self.params, self.config_dict, 
                self.dt, self.use_adjoint
            )
        else:
            # Batch processing - use C++ batch if available
            return self.forward_batch(initial_state, forcing)
    
    def forward_batch(
        self,
        initial_states: torch.Tensor,
        forcing: torch.Tensor,
        use_cuda: bool = False
    ) -> torch.Tensor:
        """
        Run model for batch of HRUs.
        
        Args:
            initial_states: Initial states [batch, num_states]
            forcing: Forcing data [n_timesteps, 3] (shared) or [n_timesteps, batch, 3]
            use_cuda: Use CUDA if available
            
        Returns:
            runoff: Total runoff [n_timesteps, batch]
        """
        batch_size = initial_states.shape[0]
        
        if HAS_NATIVE:
            states_np = initial_states.detach().cpu().numpy().astype(np.float32)
            forcing_np = forcing.detach().cpu().numpy().astype(np.float32)
            params_np = self.params.detach().cpu().numpy().astype(np.float32)
            
            # Check for CUDA
            if use_cuda and hasattr(cfuse_core, 'HAS_CUDA') and cfuse_core.HAS_CUDA:
                final_states, runoff = cfuse_core.run_fuse_cuda(
                    states_np, forcing_np, params_np, self.config_dict, self.dt
                )
            else:
                final_states, runoff = cfuse_core.run_fuse_batch(
                    states_np, forcing_np, params_np, self.config_dict, self.dt
                )
            
            return torch.from_numpy(runoff).to(initial_states.device)
        else:
            # Python fallback - sequential processing
            outputs = []
            for i in range(batch_size):
                out = FUSEFunction.apply(
                    initial_states[i], 
                    forcing if forcing.dim() == 2 else forcing[:, i, :], 
                    self.params, self.config_dict, self.dt, False
                )
                outputs.append(out)
            return torch.stack(outputs, dim=1)
    
    def forward_batch_optimized(
        self,
        initial_states: torch.Tensor,
        forcing: torch.Tensor,
        workspace: 'CUDAWorkspace'
    ) -> torch.Tensor:
        """
        Run model for batch of HRUs using pre-allocated CUDA workspace (FAST).
        
        This method avoids cudaMalloc/cudaFree overhead by reusing workspace.
        Use this in training loops for maximum performance.
        
        Args:
            initial_states: Initial states [batch, num_states]
            forcing: Forcing data [n_timesteps, 3] (shared) or [n_timesteps, batch, 3]
            workspace: Pre-allocated CUDAWorkspace object
            
        Returns:
            runoff: Total runoff [n_timesteps, batch]
        """
        if not HAS_NATIVE or not hasattr(cfuse_core, 'run_fuse_cuda_workspace'):
            raise RuntimeError("forward_batch_optimized requires cfuse_core with CUDA support")
        
        batch_size = initial_states.shape[0]
        n_timesteps = forcing.shape[0]
        n_states = initial_states.shape[1]
        shared_forcing = (forcing.dim() == 2)
        
        # Resize workspace if needed
        workspace.resize_if_needed(
            batch_size, n_states, n_timesteps, shared_forcing, shared_params=True
        )
        
        # Convert to numpy (TODO: add direct GPU tensor support)
        states_np = initial_states.detach().cpu().numpy().astype(np.float32)
        forcing_np = forcing.detach().cpu().numpy().astype(np.float32)
        params_np = self.params.detach().cpu().numpy().astype(np.float32)
        
        # Call optimized CUDA batch with workspace
        final_states, runoff = cfuse_core.run_fuse_cuda_workspace(
            states_np, forcing_np, params_np, self.config_dict, self.dt,
            workspace.data_ptr, workspace.size
        )
        
        return torch.from_numpy(runoff).to(initial_states.device)
    
    def extra_repr(self) -> str:
        return f'config={self.config}, dt={self.dt}, use_adjoint={self.use_adjoint}'


# ============================================================================
# CUDA WORKSPACE MANAGEMENT
# ============================================================================

class CUDAWorkspace:
    """
    Pre-allocated GPU memory workspace for efficient CUDA batch execution.
    
    Avoids expensive cudaMalloc/cudaFree calls in training loops by reusing
    a single allocation. PyTorch's caching allocator manages the memory.
    
    Example:
        >>> workspace = CUDAWorkspace(n_hru=100, n_states=10, n_timesteps=365)
        >>> for epoch in range(100):
        ...     runoff = model.forward_batch_optimized(states, forcing, workspace)
        ...     loss.backward()
    """
    
    def __init__(
        self,
        n_hru: int,
        n_states: int,
        n_timesteps: int,
        shared_forcing: bool = True,
        shared_params: bool = True,
        device: str = 'cuda'
    ):
        """
        Allocate workspace for CUDA batch execution.
        
        Args:
            n_hru: Number of HRUs
            n_states: Number of state variables per HRU
            n_timesteps: Number of timesteps
            shared_forcing: If True, forcing is shared across HRUs
            shared_params: If True, parameters are shared across HRUs
            device: CUDA device ('cuda' or 'cuda:0', etc.)
        """
        if not HAS_NATIVE or not hasattr(cfuse_core, 'HAS_CUDA') or not cfuse_core.HAS_CUDA:
            raise RuntimeError("CUDAWorkspace requires cfuse_core with CUDA support")
        
        self.n_hru = n_hru
        self.n_states = n_states
        self.n_timesteps = n_timesteps
        self.shared_forcing = shared_forcing
        self.shared_params = shared_params
        self.device = device
        
        # Compute workspace size
        self.size = cfuse_core.compute_cuda_workspace_size(
            n_hru, n_states, n_timesteps, shared_forcing, shared_params
        )
        
        # Allocate via PyTorch - this uses the caching allocator!
        self._buffer = torch.empty(self.size, dtype=torch.uint8, device=device)
    
    @property
    def data_ptr(self) -> int:
        """Get raw pointer to GPU memory."""
        return self._buffer.data_ptr()
    
    def resize_if_needed(
        self,
        n_hru: int,
        n_states: int,
        n_timesteps: int,
        shared_forcing: bool = True,
        shared_params: bool = True
    ):
        """Resize workspace if needed (reallocates only if larger)."""
        new_size = cfuse_core.compute_cuda_workspace_size(
            n_hru, n_states, n_timesteps, shared_forcing, shared_params
        )
        if new_size > self.size:
            self._buffer = torch.empty(new_size, dtype=torch.uint8, device=self.device)
            self.size = new_size
            self.n_hru = n_hru
            self.n_states = n_states
            self.n_timesteps = n_timesteps
            self.shared_forcing = shared_forcing
            self.shared_params = shared_params
    
    def __repr__(self):
        return (f"CUDAWorkspace(n_hru={self.n_hru}, n_states={self.n_states}, "
                f"n_timesteps={self.n_timesteps}, size={self.size/1024:.1f}KB)")


def create_workspace(
    n_hru: int,
    n_states: int = 10,
    n_timesteps: int = 365,
    shared_forcing: bool = True,
    shared_params: bool = True,
    device: str = 'cuda'
) -> CUDAWorkspace:
    """
    Factory function to create a CUDA workspace.
    
    Example:
        >>> workspace = create_workspace(n_hru=100, n_timesteps=365)
        >>> runoff = model.forward_batch_optimized(states, forcing, workspace)
    """
    return CUDAWorkspace(n_hru, n_states, n_timesteps, shared_forcing, shared_params, device)


# ============================================================================
# BATCH SIMULATION HELPERS  
# ============================================================================

def run_batch(
    model: FUSE,
    initial_states: torch.Tensor,
    forcing: torch.Tensor,
    use_cuda: bool = False
) -> torch.Tensor:
    """
    Run FUSE model for batch of basins.
    
    This is the recommended way to run multi-basin simulations.
    Uses C++ parallelization (OpenMP on CPU, CUDA on GPU).
    
    Args:
        model: FUSE model instance
        initial_states: Initial states [n_basins, num_states]
        forcing: Forcing [n_timesteps, 3] (shared) or [n_timesteps, n_basins, 3]
        use_cuda: Use GPU if available
        
    Returns:
        runoff: Runoff time series [n_timesteps, n_basins]
    """
    return model.forward_batch(initial_states, forcing, use_cuda)


# ============================================================================
# PURE PYTHON FALLBACK
# ============================================================================

def _smooth_max(a, b, k=0.01):
    """Smooth maximum function"""
    return k * np.log(np.exp(a/k) + np.exp(b/k))


def _smooth_min(a, b, k=0.01):
    """Smooth minimum function"""
    return -_smooth_max(-a, -b, k)


def _logistic_overflow(S, S_max, w):
    """Logistic overflow function"""
    e_mult = 5.0
    x = (S - S_max - w * e_mult) / w
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def _run_fuse_python(initial_state, forcing, params, config_dict, dt):
    """
    Pure Python implementation of FUSE model.
    
    This is a simplified fallback for when the C++ module is not available.
    Only implements VIC-style configuration for now.
    """
    n_timesteps = forcing.shape[0]
    
    # Extract parameters
    S1_max = params[0]
    S2_max = params[1]
    ku = params[6]
    c = params[7]
    ks = params[12]
    n = params[13]
    b = params[18]
    T_rain = params[22]
    melt_rate = params[23]
    
    # Initialize state
    S1 = initial_state[0]
    S2 = initial_state[1] if len(initial_state) > 1 else 500.0
    SWE = initial_state[-1] if config_dict.get('enable_snow', True) else 0.0
    
    runoff = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        precip = forcing[t, 0]
        pet = forcing[t, 1]
        temp = forcing[t, 2] if forcing.shape[1] > 2 else 10.0
        
        # Snow module
        snow_frac = 1.0 / (1.0 + np.exp(temp - T_rain))
        snow = precip * snow_frac
        rain = precip * (1.0 - snow_frac)
        pot_melt = max(0, melt_rate * (temp - T_rain))
        SWE_after = SWE + snow
        melt = min(pot_melt, SWE_after)
        SWE = max(0, SWE_after - melt)
        throughfall = rain + melt
        
        # Saturated area (VIC b-curve)
        S1_frac = min(S1, S1_max) / S1_max
        Ac = 1.0 - (1.0 - S1_frac) ** b
        
        # Surface runoff
        qsx = Ac * throughfall
        infiltration = throughfall - qsx
        
        # Evaporation
        e1 = pet * S1_frac
        
        # Percolation
        q12 = ku * (S1_frac ** c)
        
        # Baseflow (nonlinear)
        S2_frac = S2 / S2_max
        qb = ks * (S2_frac ** n)
        
        # State updates
        dS1 = infiltration - e1 - q12
        dS2 = q12 - qb
        
        S1 = max(0, S1 + dS1 * dt)
        S2 = max(0, S2 + dS2 * dt)
        
        # Total runoff
        runoff[t] = qsx + qb
    
    final_state = np.array([S1, S2, SWE])
    return final_state, runoff


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_model(
    model_type: str = 'vic',
    dt: float = 1.0,
    **kwargs
) -> FUSE:
    """
    Create FUSE model with predefined configuration.
    
    Args:
        model_type: One of 'prms', 'sacramento', 'topmodel', 'vic'
        dt: Timestep in days
        **kwargs: Additional arguments passed to FUSE constructor
        
    Returns:
        FUSE model instance
    """
    configs = {
        'prms': PRMS_CONFIG,
        'sacramento': SACRAMENTO_CONFIG,
        'topmodel': TOPMODEL_CONFIG,
        'vic': VIC_CONFIG
    }
    
    config = configs.get(model_type.lower())
    if config is None:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: {list(configs.keys())}")
    
    return FUSE(config=config, dt=dt, **kwargs)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Create model
    model = create_model('vic', dt=1.0, learnable_params=True)
    
    # Generate synthetic forcing
    n_days = 365
    forcing = torch.zeros(n_days, 3)
    forcing[:, 0] = 5.0 + 3.0 * torch.sin(torch.linspace(0, 2*np.pi, n_days))  # Precip
    forcing[:, 1] = 3.0 + 2.0 * torch.sin(torch.linspace(0, 2*np.pi, n_days))  # PET
    forcing[:, 2] = 15.0 + 10.0 * torch.sin(torch.linspace(0, 2*np.pi, n_days))  # Temp
    
    # Initial state
    initial_state = model.get_initial_state()
    
    # Forward pass
    runoff = model(initial_state, forcing)
    print(f"Runoff shape: {runoff.shape}")
    print(f"Mean runoff: {runoff.mean():.2f} mm/day")
    
    # Backward pass (gradient computation)
    loss = runoff.mean()
    loss.backward()
    print(f"Parameter gradients computed: {model.raw_params.grad is not None}")
