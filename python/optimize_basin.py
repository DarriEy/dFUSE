"""
Enhanced dFUSE Basin Optimization with Enzyme AD

Features:
- Learning rate scheduling (cosine annealing, step decay, warmup)
- Spinup period exclusion from loss
- Early stopping with patience
- Gradient clipping
- Parameter monitoring and logging
- Multiple optimizer options (Adam, AdamW, SGD with momentum)
- Checkpoint saving/loading
- Multi-objective loss options (NSE, KGE, RMSE)
"""

import torch
import dfuse_core
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json
import argparse

from dfuse import FUSEConfig, VIC_CONFIG
from dfuse_netcdf import read_fuse_forcing, parse_file_manager, read_elevation_bands


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for optimization run"""
    # Data paths
    basin_id: str = "Klondike_Bonanza_Creek"
    base_path: str = "/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Klondike_Bonanza_Creek"
    
    # Optimization settings
    n_iterations: int = 200
    optimizer: str = "adam"  # adam, adamw, sgd
    lr_initial: float = 0.1
    lr_min: float = 0.001
    lr_schedule: str = "cosine"  # cosine, step, exponential, warmup_cosine
    lr_warmup_steps: int = 10
    lr_step_size: int = 50  # for step schedule
    lr_gamma: float = 0.5  # decay factor for step/exponential
    
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
    'S1_max': 100.0, 'S2_max': 1000.0,
    'f_tens': 0.5, 'f_rchr': 0.5, 'f_base': 0.5,
    'r1': 0.5, 'ku': 500.0, 'c': 10.5,
    'alpha': 125.5, 'psi': 3.0, 'kappa': 0.5,
    'ki': 500.0, 'ks': 50.0, 'n': 5.0,
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
    @staticmethod
    def forward(ctx, params_physical, forcing, state_init, elev_bands, config_dict, 
                ref_elev, route_shape):
        params_np = params_physical.detach().numpy().astype(np.float32)
        forcing_np = forcing.detach().numpy().astype(np.float32)
        state_np = state_init.detach().numpy().astype(np.float32)
        
        # Run Forward
        res = dfuse_core.run_fuse_elevation_bands(
            state_np, forcing_np, params_np, config_dict,
            elev_bands.area_frac.astype(np.float32),
            elev_bands.mean_elev.astype(np.float32),
            float(ref_elev),
            None, 1.0, False, False, 1, "euler"
        )
        _, runoff, _ = res
        
        # Routing
        p_map = {name: i for i, name in enumerate(PARAM_NAMES)}
        delay = float(params_np[p_map['mu_t']])
        
        runoff_routed = dfuse_core.route_runoff(runoff, route_shape, delay, 1.0)
        
        ctx.save_for_backward(params_physical, forcing, state_init)
        ctx.elev_bands = elev_bands
        ctx.config_dict = config_dict
        ctx.ref_elev = ref_elev
        ctx.shape = route_shape
        ctx.delay = delay
        
        return torch.from_numpy(runoff_routed)

    @staticmethod
    def backward(ctx, grad_output):
        params_physical, forcing, state_init = ctx.saved_tensors
        
        s_soil = state_init.detach().numpy().astype(np.float32)
        full_state = np.zeros(9 + 30, dtype=np.float32)
        full_state[0] = s_soil[0]
        full_state[5] = s_soil[1]
        
        grad_np = grad_output.detach().numpy().astype(np.float32)
        params_np = params_physical.detach().numpy().astype(np.float32)
        forcing_np = forcing.detach().numpy().astype(np.float32)
        
        grad_params = dfuse_core.compute_gradient_adjoint_bands(
            full_state, forcing_np, params_np, grad_np,
            ctx.config_dict,
            ctx.elev_bands.area_frac.astype(np.float32),
            ctx.elev_bands.mean_elev.astype(np.float32),
            float(ctx.ref_elev), float(ctx.shape), float(ctx.delay), 1.0
        )
        
        return torch.from_numpy(grad_params), None, None, None, None, None, None


# =============================================================================
# MODEL WRAPPER
# =============================================================================

class CppFUSEModel(torch.nn.Module):
    def __init__(self, init_params: dict, elev_bands, config, ref_elev: float, route_shape: float):
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
            self.elev_bands, self.config_dict, self.ref_elev, self.route_shape
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
    
    # Initialize model
    model = CppFUSEModel(
        init_params=DEFAULT_INIT_PARAMS,
        elev_bands=elev_bands,
        config=VIC_CONFIG,
        ref_elev=config.ref_elev,
        route_shape=config.route_shape
    )
    
    # Create optimizer
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_initial, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_initial, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr_initial, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Create LR scheduler
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
    print(f"  Early stopping: {config.early_stopping} (patience={config.patience})")
    print()
    
    pbar = tqdm(range(config.n_iterations), desc="Optimizing")
    
    for i in pbar:
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
            nse_loss = compute_nse(q_sim, obs_tensor, mask)
            nse = 1 - nse_loss.item()
            kge_loss = compute_kge(q_sim, obs_tensor, mask)
            kge = 1 - kge_loss.item()
        
        # Track history
        current_lr = optimizer.param_groups[0]['lr']
        history['loss'].append(loss.item())
        history['nse'].append(nse)
        history['kge'].append(kge)
        history['lr'].append(current_lr)
        history['grad_norm'].append(grad_norm.item())
        
        # Track best
        if nse > history['best_nse']:
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
            tqdm.write(f"Iter {i}: NSE={nse:.4f}, KGE={kge:.4f}, LR={current_lr:.2e}, GradNorm={grad_norm:.2f}")
        
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
        
        # Early stopping
        if early_stopper is not None:
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
    parser = argparse.ArgumentParser(description='Enhanced dFUSE Basin Optimization')
    
    # Data args
    parser.add_argument('--basin', type=str, default="Klondike_Bonanza_Creek", help='Basin ID')
    parser.add_argument('--base-path', type=str, 
                        default="/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Klondike_Bonanza_Creek")
    
    # Optimization args
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--lr-schedule', type=str, default='warmup_cosine', 
                        choices=['cosine', 'step', 'exponential', 'warmup_cosine'])
    parser.add_argument('--warmup-steps', type=int, default=10, help='LR warmup steps')
    
    # Loss args
    parser.add_argument('--loss', type=str, default='nse', choices=['nse', 'nse_log', 'kge', 'rmse'])
    parser.add_argument('--spinup-days', type=int, default=365, help='Spinup days to exclude')
    
    # Early stopping
    parser.add_argument('--no-early-stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    
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
        loss_type=args.loss,
        spinup_days=args.spinup_days,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        grad_clip_norm=args.grad_clip,
        output_dir=args.output_dir
    )
    
    # Run optimization
    model, history, best_params = run_optimization(config)


if __name__ == "__main__":
    main()
