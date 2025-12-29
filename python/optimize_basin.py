"""
Enhanced cFUSE Basin Optimization with Enzyme AD

Features:
- Learning rate scheduling (cosine annealing, step decay, warmup)
- Spinup period exclusion from loss
- Early stopping with patience
- Gradient clipping
- Parameter monitoring and logging
- Multiple optimizer options (Adam, AdamW, SGD with momentum, L-BFGS)
- Checkpoint saving/loading
- Multi-objective loss options (NSE, KGE, RMSE)
"""

import torch
import cfuse_core
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json
import argparse

from cfuse import FUSEConfig, VIC_CONFIG
from cfuse.netcdf import read_fuse_forcing, parse_file_manager, read_elevation_bands

# =============================================================================
# PATH RESOLUTION
# =============================================================================

# Get the directory where this script resides (cFUSE/python)
SCRIPT_DIR = Path(__file__).resolve().parent

# Get the repository root (cFUSE/)
REPO_ROOT = SCRIPT_DIR.parent

# Define the default data path relative to the repo root
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "domain_Bow_at_Banff_lumped_era5"

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for optimization run"""
    # Data paths
    basin_id: str = "Bow_at_Banff_lumped_era5"
    base_path: str = str(DEFAULT_DATA_PATH)
    
    # Optimization settings
    n_iterations: int = 200
    optimizer: str = "adam"  # adam, adamw, sgd, lbfgs
    lr_initial: float = 0.1
    lr_min: float = 0.001
    lr_schedule: str = "cosine"  # cosine, step, exponential, warmup_cosine
    lr_warmup_steps: int = 10
    lr_step_size: int = 50  # for step schedule
    lr_gamma: float = 0.5  # decay factor for step/exponential
    
    # Gradient backend
    gradient_backend: str = "enzyme"  # enzyme, cvodes
    
    # Regularization
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    
    # Loss function
    loss_type: str = "nse"  # nse, kge, rmse, nse_log
    spinup_days: int = 365  # exclude first N days from loss
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 30
    min_delta: float = 0.001
    
    # Logging and checkpoints
    log_interval: int = 10
    checkpoint_interval: int = 50
    output_dir: str = "optimization_results"
    
    # Reference elevation
    ref_elev: float = 1018.0
    
    # Routing
    route_shape: float = 2.5
    
    # L-BFGS specific settings
    lbfgs_max_iter: int = 20  # max iterations per step (line search evals)
    lbfgs_history_size: int = 10  # number of past updates to store
    lbfgs_line_search: str = "strong_wolfe"  # strong_wolfe or None


# =============================================================================
# PARAMETER DEFINITIONS
# =============================================================================

PARAM_NAMES = [
    'S1_max', 'S2_max', 'f_tens', 'f_rchr', 'f_base', 'r1',
    'ku', 'c', 'alpha', 'psi', 'kappa', 'ki',
    'ks', 'n', 'v', 'v_A', 'v_B',
    'Ac_max', 'b', 'lambda', 'chi', 'mu_t',
    'T_rain', 'T_melt', 'melt_rate', 'lapse_rate', 'opg',
    'MFMAX', 'MFMIN'
]

PARAM_BOUNDS = {
    'S1_max': (50.0, 5000.0), 'S2_max': (100.0, 10000.0),
    'f_tens': (0.05, 0.95), 'f_rchr': (0.05, 0.95), 'f_base': (0.05, 0.95),
    'r1': (0.05, 0.95), 'ku': (0.01, 1000.0), 'c': (1.0, 20.0),
    'alpha': (1.0, 250.0), 'psi': (1.0, 5.0), 'kappa': (0.05, 0.95),
    'ki': (0.01, 1000.0), 'ks': (0.001, 10000.0), 'n': (1.0, 10.0),
    'v': (0.001, 0.25), 'v_A': (0.001, 0.25), 'v_B': (0.001, 0.25),
    'Ac_max': (0.05, 0.95), 'b': (0.001, 3.0), 'lambda': (5.0, 10.0),
    'chi': (2.0, 5.0), 'mu_t': (0.01, 5.0), 'T_rain': (-2.0, 4.0),
    'T_melt': (-2.0, 4.0), 'melt_rate': (1.0, 10.0),
    'lapse_rate': (-9.8, 0.0), 'opg': (0.0, 1.0),
    'MFMAX': (1.0, 10.0), 'MFMIN': (0.0, 10.0)
}

# Default initial parameters (physically reasonable starting point)
DEFAULT_INIT_PARAMS = {
    'S1_max': 200.0, 'S2_max': 2000.0,  # Was 100, 1000
    'ku': 10.0,      # Was 500 - very high!
    'ki': 10.0,      # Was 500 - very high!
    'c': 4.0,        # Was 10.5
    'alpha': 100.0,  # Was 125.5
    'v': 0.125, 'v_A': 0.125, 'v_B': 0.125,
    'Ac_max': 0.5, 'b': 1.5, 'lambda': 7.5,
    'chi': 3.5, 'mu_t': 0.9,
    'T_rain': 1.0, 'T_melt': 1.0, 'melt_rate': 5.5,
    'lapse_rate': -5.0, 'opg': 0.5,
    'MFMAX': 4.2, 'MFMIN': 2.4
}


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_nse(sim: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Nash-Sutcliffe Efficiency (returns loss = 1 - NSE)"""
    sse = torch.sum((sim[mask] - obs[mask])**2)
    sst = torch.sum((obs[mask] - obs[mask].mean())**2)
    return sse / sst


def compute_nse_log(sim: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, 
                    eps: float = 0.01) -> torch.Tensor:
    """NSE on log-transformed flows (better for low flows)"""
    sim_log = torch.log(sim[mask] + eps)
    obs_log = torch.log(obs[mask] + eps)
    sse = torch.sum((sim_log - obs_log)**2)
    sst = torch.sum((obs_log - obs_log.mean())**2)
    return sse / sst


def compute_kge(sim: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Kling-Gupta Efficiency (returns loss = 1 - KGE)"""
    sim_m = sim[mask]
    obs_m = obs[mask]
    
    # Correlation
    sim_mean = sim_m.mean()
    obs_mean = obs_m.mean()
    sim_std = sim_m.std()
    obs_std = obs_m.std()
    
    cov = torch.mean((sim_m - sim_mean) * (obs_m - obs_mean))
    r = cov / (sim_std * obs_std + 1e-10)
    
    # Bias ratio
    beta = sim_mean / (obs_mean + 1e-10)
    
    # Variability ratio
    gamma = sim_std / (obs_std + 1e-10)
    
    # KGE
    kge = 1 - torch.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    return 1 - kge  # Return as loss


def compute_rmse(sim: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Root Mean Square Error"""
    mse = torch.mean((sim[mask] - obs[mask])**2)
    return torch.sqrt(mse)


LOSS_FUNCTIONS = {
    'nse': compute_nse,
    'nse_log': compute_nse_log,
    'kge': compute_kge,
    'rmse': compute_rmse
}


# =============================================================================
# HELPERS
# =============================================================================

def inverse_sigmoid(val: float, low: float, high: float) -> float:
    """Convert physical parameter to unconstrained space"""
    val = np.clip(val, low + 1e-5, high - 1e-5)
    norm = (val - low) / (high - low)
    return np.log(norm / (1 - norm))


def get_lr_scheduler(optimizer, config: OptimizationConfig):
    """Create learning rate scheduler"""
    # L-BFGS typically doesn't use LR schedulers in the same way
    if config.optimizer == "lbfgs":
        return None
        
    if config.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_iterations, eta_min=config.lr_min
        )
    elif config.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
        )
    elif config.lr_schedule == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.lr_gamma
        )
    elif config.lr_schedule == "warmup_cosine":
        def lr_lambda(step):
            if step < config.lr_warmup_steps:
                return step / config.lr_warmup_steps
            else:
                progress = (step - config.lr_warmup_steps) / (config.n_iterations - config.lr_warmup_steps)
                return config.lr_min/config.lr_initial + (1 - config.lr_min/config.lr_initial) * 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return None


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================================================================
# C++ AUTOGRAD FUNCTION
# =============================================================================

class CppFUSEFunction(torch.autograd.Function):
    """
    Custom autograd function that wraps C++ forward/backward passes.
    
    Supports two gradient backends:
    - enzyme: Enzyme AD through explicit Euler (default)
    - cvodes: CVODES adjoint sensitivity with implicit BDF solver
    """
    
    @staticmethod
    def forward(ctx, params_physical, forcing, state_init, elev_bands, config_dict, 
                ref_elev, route_shape, gradient_backend="enzyme"):
        params_np = params_physical.detach().numpy().astype(np.float32)
        forcing_np = forcing.detach().numpy().astype(np.float32)
        state_np = state_init.detach().numpy().astype(np.float32)
        
        # Run Forward (single HRU via batch API)
        initial_states = state_np[None, :]
        _, runoff = cfuse_core.run_fuse_batch(
            initial_states, forcing_np, params_np, config_dict, 1.0
        )
        runoff = runoff[:, 0]
        
        # Routing
        p_map = {name: i for i, name in enumerate(PARAM_NAMES)}
        delay = float(params_np[p_map['mu_t']])
        
        runoff_routed = cfuse_core.route_runoff(runoff, route_shape, delay, 1.0)
        
        ctx.save_for_backward(params_physical, forcing, state_init)
        ctx.config_dict = config_dict
        ctx.shape = route_shape
        ctx.delay = delay
        ctx.gradient_backend = gradient_backend
        
        return torch.from_numpy(runoff_routed)

    @staticmethod
    def backward(ctx, grad_output):
        params_physical, forcing, state_init = ctx.saved_tensors

        grad_np = grad_output.detach().numpy().astype(np.float32)
        params_np = params_physical.detach().numpy().astype(np.float32)
        forcing_np = forcing.detach().numpy().astype(np.float32)
        state_np = state_init.detach().numpy().astype(np.float32)

        # Route gradient back to instantaneous runoff
        grad_rev = cfuse_core.route_runoff(grad_np[::-1], float(ctx.shape), float(ctx.delay), 1.0)
        grad_instant = grad_rev[::-1]

        initial_states = state_np[None, :]
        forcing_batch = forcing_np[:, None, :]
        grad_batch = grad_instant[:, None]

        # Select gradient backend
        if ctx.gradient_backend == "enzyme" and hasattr(cfuse_core, 'run_fuse_batch_gradient'):
            grad_params = cfuse_core.run_fuse_batch_gradient(
                initial_states, forcing_batch, params_np, grad_batch,
                ctx.config_dict, 1.0
            )
        else:
            grad_params = cfuse_core.run_fuse_batch_gradient_numerical(
                initial_states, forcing_batch, params_np, grad_batch,
                ctx.config_dict, 1.0
            )
        
        return torch.from_numpy(grad_params), None, None, None, None, None, None, None


# =============================================================================
# MODEL WRAPPER
# =============================================================================

class CppFUSEModel(torch.nn.Module):
    def __init__(self, init_params: dict, elev_bands, config, ref_elev: float, 
                 route_shape: float, gradient_backend: str = "enzyme"):
        super().__init__()
        
        self.param_names = PARAM_NAMES
        self.param_bounds = [PARAM_BOUNDS[n] for n in PARAM_NAMES]
        
        # Initialize parameters in unconstrained space
        raw_params = []
        for i, name in enumerate(PARAM_NAMES):
            val = init_params.get(name, sum(PARAM_BOUNDS[name]) / 2)
            low, high = self.param_bounds[i]
            raw = inverse_sigmoid(val, low, high)
            raw_params.append(raw)
            
        self.raw_params = torch.nn.Parameter(torch.tensor(raw_params, dtype=torch.float32))
        
        self.elev_bands = elev_bands
        self.config_dict = config.to_dict()
        self.state_init = torch.tensor([50.0, 250.0], dtype=torch.float32)
        self.ref_elev = ref_elev
        self.route_shape = route_shape
        self.gradient_backend = gradient_backend
        
    def get_physical_params(self) -> torch.Tensor:
        """Convert raw parameters to physical space"""
        phys = []
        for i, raw in enumerate(self.raw_params):
            low, high = self.param_bounds[i]
            val = low + (high - low) * torch.sigmoid(raw)
            phys.append(val)
        return torch.stack(phys)
    
    def get_params_dict(self) -> dict:
        """Get parameters as dictionary"""
        phys = self.get_physical_params().detach().numpy()
        return {name: float(phys[i]) for i, name in enumerate(PARAM_NAMES)}

    def forward(self, forcing: torch.Tensor) -> torch.Tensor:
        params_physical = self.get_physical_params()
        return CppFUSEFunction.apply(
            params_physical, forcing, self.state_init,
            self.elev_bands, self.config_dict, self.ref_elev, self.route_shape,
            self.gradient_backend
        )


# =============================================================================
# OPTIMIZATION LOOP
# =============================================================================

def run_optimization(config: OptimizationConfig):
    """Main optimization function"""
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    fm_path = Path(config.base_path) / "settings/FUSE/fm_catch.txt"
    fm = parse_file_manager(fm_path)
    forcing = read_fuse_forcing(Path(fm['input_path']) / f"{config.basin_id}{fm['suffix_forcing']}")
    elev_bands = read_elevation_bands(Path(fm['input_path']) / f"{config.basin_id}{fm['suffix_elev_bands']}")
    
    f_tensor = forcing.to_tensor()
    obs_tensor = torch.tensor(forcing.q_obs, dtype=torch.float32)
    obs_tensor[obs_tensor < 0] = float('nan')
    
    # Create observation mask (excluding spinup)
    mask = ~torch.isnan(obs_tensor)
    if config.spinup_days > 0:
        mask[:config.spinup_days] = False
    
    n_valid = mask.sum().item()
    print(f"Data: {len(obs_tensor)} timesteps, {n_valid} valid observations (after {config.spinup_days} day spinup)")
    
    # Check CVODES availability
    has_cvodes = getattr(cfuse_core, 'HAS_CVODES', False)
    if config.gradient_backend == "cvodes" and not has_cvodes:
        print("WARNING: CVODES backend requested but not available. Falling back to Enzyme.")
        config.gradient_backend = "enzyme"
    
    # Initialize model
    model = CppFUSEModel(
        init_params=DEFAULT_INIT_PARAMS,
        elev_bands=elev_bands,
        config=VIC_CONFIG,
        ref_elev=config.ref_elev,
        route_shape=config.route_shape,
        gradient_backend=config.gradient_backend
    )
    
    # Create optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_initial, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_initial, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr_initial, momentum=0.9, weight_decay=config.weight_decay)
    elif config.optimizer == "lbfgs":
        # Without line search, L-BFGS needs a much smaller learning rate
        lbfgs_lr = config.lr_initial
        if config.lbfgs_line_search == "none" and config.lr_initial > 0.2:
            lbfgs_lr = 0.1
            print(f"  Note: Reducing L-BFGS LR from {config.lr_initial} to {lbfgs_lr} (no line search)")
        
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn=config.lbfgs_line_search if config.lbfgs_line_search != "none" else None
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Create LR scheduler (not used for L-BFGS)
    scheduler = get_lr_scheduler(optimizer, config)
    
    # Early stopping
    early_stopper = EarlyStopping(patience=config.patience, min_delta=config.min_delta) if config.early_stopping else None
    
    # Get loss function
    loss_fn = LOSS_FUNCTIONS[config.loss_type]
    
    # Tracking
    history = {
        'loss': [], 'nse': [], 'kge': [], 'lr': [],
        'grad_norm': [], 'best_nse': -np.inf, 'best_iter': 0
    }
    best_params = None
    
    print(f"\nStarting optimization:")
    print(f"  Optimizer: {config.optimizer}, LR: {config.lr_initial}, Schedule: {config.lr_schedule}")
    print(f"  Loss: {config.loss_type}, Iterations: {config.n_iterations}")
    print(f"  Gradient backend: {config.gradient_backend.upper()}")
    if config.optimizer == "lbfgs":
        print(f"  L-BFGS settings: max_iter={config.lbfgs_max_iter}, history={config.lbfgs_history_size}, line_search={config.lbfgs_line_search}")
    print(f"  Early stopping: {config.early_stopping} (patience={config.patience})")
    print()
    
    pbar = tqdm(range(config.n_iterations), desc="Optimizing")
    
    # For L-BFGS, we need a closure
    is_lbfgs = config.optimizer == "lbfgs"
    lbfgs_consecutive_failures = 0
    max_lbfgs_failures = 5  # Reset optimizer after this many consecutive failures
    
    for i in pbar:
        # Check for NaN parameters (indicates corrupted optimizer state)
        if torch.isnan(model.raw_params).any() or torch.isinf(model.raw_params).any():
            tqdm.write(f"Iter {i}: NaN/Inf in parameters detected, resetting to best known values")
            if best_params is not None:
                # Reset to best known parameters
                with torch.no_grad():
                    for j, name in enumerate(PARAM_NAMES):
                        val = best_params[name]
                        low, high = model.param_bounds[j]
                        model.raw_params[j] = inverse_sigmoid(val, low, high)
            else:
                # Reset to initial parameters
                with torch.no_grad():
                    for j, name in enumerate(PARAM_NAMES):
                        val = DEFAULT_INIT_PARAMS.get(name, sum(PARAM_BOUNDS[name]) / 2)
                        low, high = model.param_bounds[j]
                        model.raw_params[j] = inverse_sigmoid(val, low, high)
            
            # Reset optimizer
            if is_lbfgs:
                reset_lr = 0.1 if config.lbfgs_line_search == "none" else config.lr_initial
                optimizer = torch.optim.LBFGS(
                    model.parameters(), lr=reset_lr,
                    max_iter=config.lbfgs_max_iter, history_size=config.lbfgs_history_size,
                    line_search_fn=config.lbfgs_line_search if config.lbfgs_line_search != "none" else None
                )
            continue
        
        # Variables to track within closure for L-BFGS
        current_loss = None
        current_q_sim = None
        grad_norm_value = 0.0
        
        if is_lbfgs:
            # L-BFGS requires a closure that computes the loss and gradients
            # Errors can occur inside the closure during line search, so we handle them there
            closure_error = [False]  # Use list to allow modification in nested function
            
            def closure():
                nonlocal current_loss, current_q_sim, grad_norm_value
                try:
                    optimizer.zero_grad()
                    q_sim = model(f_tensor)
                    
                    # Check for NaN/Inf in simulation output
                    if torch.isnan(q_sim).any() or torch.isinf(q_sim).any():
                        closure_error[0] = True
                        # Return a high loss to signal bad parameters
                        return torch.tensor(1e6, requires_grad=True)
                    
                    loss = loss_fn(q_sim, obs_tensor, mask)
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        closure_error[0] = True
                        return torch.tensor(1e6, requires_grad=True)
                    
                    loss.backward()
                    
                    # Check for NaN gradients
                    if model.raw_params.grad is None or torch.isnan(model.raw_params.grad).any():
                        closure_error[0] = True
                        if model.raw_params.grad is not None:
                            model.raw_params.grad = torch.zeros_like(model.raw_params.grad)
                        return loss
                    
                    # Gradient clipping
                    grad_norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm).item()
                    
                    current_loss = loss
                    current_q_sim = q_sim.detach()
                    return loss
                    
                except Exception as e:
                    closure_error[0] = True
                    tqdm.write(f"  Closure error: {e}")
                    return torch.tensor(1e6, requires_grad=True)
            
            # L-BFGS step (may call closure multiple times)
            try:
                optimizer.step(closure)
            except (RuntimeError, IndexError) as e:
                closure_error[0] = True
                tqdm.write(f"Iter {i}: L-BFGS step failed ({type(e).__name__})")
            
            # Handle any errors that occurred
            if closure_error[0]:
                lbfgs_consecutive_failures += 1
                
                # Fall back to a simple gradient descent step
                # First evaluate with gradients enabled
                optimizer.zero_grad()
                q_sim_fallback = model(f_tensor)
                if not (torch.isnan(q_sim_fallback).any() or torch.isinf(q_sim_fallback).any()):
                    loss_fallback = loss_fn(q_sim_fallback, obs_tensor, mask)
                    if not (torch.isnan(loss_fallback) or torch.isinf(loss_fallback)):
                        loss_fallback.backward()
                        if model.raw_params.grad is not None and not torch.isnan(model.raw_params.grad).any():
                            # Take a small gradient step
                            with torch.no_grad():
                                model.raw_params -= 0.01 * model.raw_params.grad
                
                # Reset optimizer if too many consecutive failures
                if lbfgs_consecutive_failures >= max_lbfgs_failures:
                    tqdm.write(f"  Resetting L-BFGS optimizer after {max_lbfgs_failures} consecutive failures")
                    if config.lbfgs_line_search == "strong_wolfe":
                        tqdm.write(f"  Tip: Try --lbfgs-line-search none --lr 0.1 for more stability")
                    
                    # Use reduced LR for reset if no line search
                    reset_lr = config.lr_initial
                    if config.lbfgs_line_search == "none" and config.lr_initial > 0.2:
                        reset_lr = 0.1
                    
                    optimizer = torch.optim.LBFGS(
                        model.parameters(),
                        lr=reset_lr,
                        max_iter=config.lbfgs_max_iter,
                        history_size=config.lbfgs_history_size,
                        line_search_fn=config.lbfgs_line_search if config.lbfgs_line_search != "none" else None
                    )
                    lbfgs_consecutive_failures = 0
                continue
            else:
                lbfgs_consecutive_failures = 0  # Reset on success
            
            loss = current_loss
            q_sim = current_q_sim
            grad_norm = grad_norm_value
            
        else:
            # Standard optimizer flow (Adam, AdamW, SGD)
            optimizer.zero_grad()
            
            # Forward pass
            q_sim = model(f_tensor)
            
            # Compute loss
            loss = loss_fn(q_sim, obs_tensor, mask)
            
            # Backward pass
            loss.backward()
            
            # Check for NaN gradients
            if torch.isnan(model.raw_params.grad).any():
                tqdm.write(f"Iter {i}: NaN gradient detected, skipping step")
                optimizer.zero_grad()
                continue
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            
            # Optimizer step
            optimizer.step()
            
            # LR scheduler step
            if scheduler is not None:
                scheduler.step()
        
        # Compute metrics for logging
        with torch.no_grad():
            # Re-run forward if needed (for L-BFGS, q_sim might be stale after step)
            if is_lbfgs:
                q_sim = model(f_tensor)
            
            # Check for invalid simulation
            if torch.isnan(q_sim).any() or torch.isinf(q_sim).any():
                nse = float('nan')
                kge = float('nan')
            else:
                nse_loss = compute_nse(q_sim, obs_tensor, mask)
                nse = 1 - nse_loss.item() if not torch.isnan(nse_loss) else float('nan')
                kge_loss = compute_kge(q_sim, obs_tensor, mask)
                kge = 1 - kge_loss.item() if not torch.isnan(kge_loss) else float('nan')
        
        # Track history
        current_lr = optimizer.param_groups[0]['lr']
        history['loss'].append(loss.item() if hasattr(loss, 'item') else float(loss))
        history['nse'].append(nse)
        history['kge'].append(kge)
        history['lr'].append(current_lr)
        history['grad_norm'].append(float(grad_norm) if not isinstance(grad_norm, float) else grad_norm)
        
        # Track best (only for valid scores)
        if not np.isnan(nse) and nse > history['best_nse']:
            history['best_nse'] = nse
            history['best_iter'] = i
            best_params = model.get_params_dict()
        
        # Update progress bar
        pbar.set_postfix({
            'NSE': f"{nse:.4f}",
            'KGE': f"{kge:.4f}",
            'LR': f"{current_lr:.2e}",
            'Best': f"{history['best_nse']:.4f}"
        })
        
        # Logging
        if i > 0 and i % config.log_interval == 0:
            grad_norm_str = f"{grad_norm:.2f}" if isinstance(grad_norm, float) else f"{grad_norm.item():.2f}"
            tqdm.write(f"Iter {i}: NSE={nse:.4f}, KGE={kge:.4f}, LR={current_lr:.2e}, GradNorm={grad_norm_str}")
        
        # Checkpointing
        if i > 0 and i % config.checkpoint_interval == 0:
            checkpoint = {
                'iteration': i,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history,
                'config': config.__dict__
            }
            torch.save(checkpoint, output_dir / f"checkpoint_{i}.pt")
        
        # Early stopping (skip if NaN)
        if early_stopper is not None and not np.isnan(nse):
            if early_stopper(-nse):  # Negative because we want to maximize
                tqdm.write(f"Early stopping triggered at iteration {i}")
                break
    
    # Final results
    print(f"\n{'='*60}")
    print(f"Optimization Complete")
    print(f"{'='*60}")
    print(f"Final NSE: {history['nse'][-1]:.4f}")
    print(f"Final KGE: {history['kge'][-1]:.4f}")
    print(f"Best NSE:  {history['best_nse']:.4f} (iteration {history['best_iter']})")
    
    print(f"\nBest Parameters:")
    for name, val in best_params.items():
        bounds = PARAM_BOUNDS[name]
        print(f"  {name:12s}: {val:10.4f}  [{bounds[0]:8.3f}, {bounds[1]:8.3f}]")
    
    # Save results
    results = {
        'config': config.__dict__,
        'history': {k: [float(x) for x in vs] if isinstance(vs, list) else vs 
                    for k, vs in history.items()},
        'best_params': best_params,
        'final_nse': history['nse'][-1],
        'best_nse': history['best_nse']
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    create_plots(history, q_sim.detach().numpy(), obs_tensor.numpy(), mask.numpy(), 
                 config, output_dir)
    
    return model, history, best_params


def create_plots(history: dict, q_sim: np.ndarray, obs: np.ndarray, mask: np.ndarray,
                 config: OptimizationConfig, output_dir: Path):
    """Create diagnostic plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training curve
    ax = axes[0, 0]
    ax.plot(history['nse'], label='NSE', color='blue')
    ax.plot(history['kge'], label='KGE', color='green', alpha=0.7)
    ax.axhline(history['best_nse'], color='red', linestyle='--', label=f'Best NSE: {history["best_nse"]:.4f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Metric')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Learning rate
    ax = axes[0, 1]
    ax.plot(history['lr'], color='orange')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Hydrograph (last 2 years)
    ax = axes[1, 0]
    n_plot = min(730, len(q_sim))
    t = np.arange(len(q_sim) - n_plot, len(q_sim))
    ax.plot(t, obs[-n_plot:], 'k-', label='Observed', alpha=0.7)
    ax.plot(t, q_sim[-n_plot:], 'b-', label='Simulated', alpha=0.7)
    ax.set_xlabel('Day')
    ax.set_ylabel('Discharge (mm/day)')
    ax.set_title('Hydrograph (Last 2 Years)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Scatter plot
    ax = axes[1, 1]
    valid_obs = obs[mask]
    valid_sim = q_sim[mask]
    ax.scatter(valid_obs, valid_sim, alpha=0.3, s=10)
    max_val = max(valid_obs.max(), valid_sim.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
    ax.set_xlabel('Observed (mm/day)')
    ax.set_ylabel('Simulated (mm/day)')
    ax.set_title(f'Scatter Plot (NSE={history["best_nse"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimization_results.png", dpi=150)
    plt.close()
    
    # Flow duration curve
    fig, ax = plt.subplots(figsize=(10, 6))
    
    obs_sorted = np.sort(valid_obs)[::-1]
    sim_sorted = np.sort(valid_sim)[::-1]
    exceedance = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100
    
    ax.semilogy(exceedance, obs_sorted, 'k-', label='Observed', linewidth=2)
    ax.semilogy(exceedance, sim_sorted, 'b-', label='Simulated', linewidth=2)
    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel('Discharge (mm/day)')
    ax.set_title('Flow Duration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "flow_duration_curve.png", dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced cFUSE Basin Optimization')
    
    # Data args
    parser.add_argument('--basin', type=str, default="Bow_at_Banff_lumped_era5", help='Basin ID')
    parser.add_argument('--base-path', type=str, 
                        default=str(DEFAULT_DATA_PATH),
                        help=f'Path to basin data (default: relative to repo root)')
    
    # Optimization args
    parser.add_argument('--iterations', type=int, default=500, help='Number of iterations')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd', 'lbfgs'])
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--lr-schedule', type=str, default='cosine', 
                        choices=['cosine', 'step', 'exponential', 'warmup_cosine'])
    parser.add_argument('--warmup-steps', type=int, default=10, help='LR warmup steps')
    
    # Gradient backend
    parser.add_argument('--gradient-backend', type=str, default='enzyme', choices=['enzyme', 'cvodes'],
                        help='Gradient computation backend: enzyme (Euler+AD) or cvodes (BDF+adjoint)')
    
    # Loss args
    parser.add_argument('--loss', type=str, default='kge', choices=['nse', 'nse_log', 'kge', 'rmse'])
    parser.add_argument('--spinup-days', type=int, default=365, help='Spinup days to exclude')
    
    # Early stopping
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
    # L-BFGS specific args
    parser.add_argument('--lbfgs-max-iter', type=int, default=20, 
                        help='L-BFGS max iterations per step (line search evaluations)')
    parser.add_argument('--lbfgs-history', type=int, default=10,
                        help='L-BFGS history size (number of past updates to store)')
    parser.add_argument('--lbfgs-line-search', type=str, default='strong_wolfe',
                        choices=['strong_wolfe', 'none'],
                        help='L-BFGS line search function (use "none" if strong_wolfe fails)')
    
    # Other
    parser.add_argument('--grad-clip', type=float, default=10.0, help='Gradient clipping norm')
    parser.add_argument('--output-dir', type=str, default='optimization_results')
    
    args = parser.parse_args()
    
    # Create config
    config = OptimizationConfig(
        basin_id=args.basin,
        base_path=args.base_path,
        n_iterations=args.iterations,
        optimizer=args.optimizer,
        lr_initial=args.lr,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.warmup_steps,
        gradient_backend=args.gradient_backend,
        loss_type=args.loss,
        spinup_days=args.spinup_days,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        grad_clip_norm=args.grad_clip,
        output_dir=args.output_dir,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history_size=args.lbfgs_history,
        lbfgs_line_search=args.lbfgs_line_search
    )
    
    # Run optimization
    model, history, best_params = run_optimization(config)


if __name__ == "__main__":
    main()
