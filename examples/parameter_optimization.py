#!/usr/bin/env python3
"""
dFUSE Example: Parameter Optimization with PyTorch

This example demonstrates how to:
1. Set up a FUSE model with configurable physics
2. Generate synthetic "observed" data
3. Optimize parameters using gradient descent
4. Compare different model structures

Based on Clark et al. (2008) WRR, doi:10.1029/2007WR006735
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Import dFUSE
from dfuse import (
    FUSE, FUSEConfig, create_model,
    UpperLayerArch, LowerLayerArch, BaseflowType,
    PercolationType, SurfaceRunoffType, EvaporationType,
    VIC_CONFIG, TOPMODEL_CONFIG, SACRAMENTO_CONFIG, PRMS_CONFIG,
    PARAM_NAMES, PARAM_BOUNDS, get_default_params
)


def generate_synthetic_forcing(
    n_days: int = 365,
    seed: int = 42
) -> torch.Tensor:
    """
    Generate realistic synthetic forcing data.
    
    Returns:
        forcing: Tensor of shape [n_days, 3] with (precip, pet, temp)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Day of year
    doy = torch.arange(n_days, dtype=torch.float32)
    phase = 2 * np.pi * doy / 365
    
    # Temperature: seasonal cycle + noise
    temp_mean = 15.0 + 12.0 * torch.sin(phase - np.pi/2)  # Peak in summer
    temp = temp_mean + 3.0 * torch.randn(n_days)
    
    # PET: follows temperature with lag
    pet_base = 2.0 + 3.0 * torch.sigmoid((temp - 10) / 5)
    pet = pet_base + 0.5 * torch.randn(n_days).abs()
    
    # Precipitation: seasonal pattern + storm events
    precip_base = 3.0 + 2.0 * torch.sin(phase + np.pi)  # More in winter
    
    # Add random storm events
    storm_prob = 0.15
    storms = torch.bernoulli(torch.full((n_days,), storm_prob))
    storm_amount = torch.exponential(torch.full((n_days,), 15.0))
    
    precip = precip_base + storms * storm_amount
    precip = torch.clamp(precip, min=0)
    
    forcing = torch.stack([precip, pet, temp], dim=1)
    return forcing


def generate_observations(
    model: FUSE,
    forcing: torch.Tensor,
    noise_std: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic observations from a model.
    
    Args:
        model: FUSE model with "true" parameters
        forcing: Forcing data
        noise_std: Standard deviation of observation noise
        
    Returns:
        initial_state: Initial model state
        obs_runoff: Observed runoff with noise
    """
    initial_state = model.get_initial_state(S1_init=150.0, S2_init=800.0)
    
    with torch.no_grad():
        true_runoff = model(initial_state, forcing)
    
    # Add observation noise
    noise = noise_std * torch.randn_like(true_runoff)
    obs_runoff = torch.clamp(true_runoff + noise, min=0)
    
    return initial_state, obs_runoff


def nse_loss(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    Nash-Sutcliffe Efficiency loss (1 - NSE for minimization).
    """
    ss_res = torch.sum((pred - obs) ** 2)
    ss_tot = torch.sum((obs - obs.mean()) ** 2)
    nse = 1 - ss_res / (ss_tot + 1e-10)
    return 1 - nse  # Return loss (lower is better)


def mse_loss(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Root Mean Square Error loss."""
    return torch.sqrt(torch.mean((pred - obs) ** 2))


def optimize_parameters(
    model: FUSE,
    forcing: torch.Tensor,
    obs_runoff: torch.Tensor,
    initial_state: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 0.1,
    verbose: bool = True
) -> dict:
    """
    Optimize model parameters to match observations.
    
    Args:
        model: FUSE model (parameters will be modified)
        forcing: Forcing data
        obs_runoff: Observed runoff
        initial_state: Initial state
        n_epochs: Number of optimization epochs
        lr: Learning rate
        verbose: Print progress
        
    Returns:
        history: Dictionary with optimization history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
    )
    
    history = {
        'loss': [],
        'nse': [],
        'params': []
    }
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        pred_runoff = model(initial_state.clone(), forcing)
        
        # Compute loss
        loss = nse_loss(pred_runoff, obs_runoff)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step(loss)
        
        # Record history
        with torch.no_grad():
            nse = 1 - loss.item()
            history['loss'].append(loss.item())
            history['nse'].append(nse)
            history['params'].append(model.params.clone().numpy())
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, NSE = {nse:.4f}")
    
    return history


def compare_model_structures(
    forcing: torch.Tensor,
    obs_runoff: torch.Tensor,
    initial_state: torch.Tensor,
    n_epochs: int = 50
) -> dict:
    """
    Compare calibration results for different model structures.
    """
    configs = {
        'VIC': VIC_CONFIG,
        'TOPMODEL': TOPMODEL_CONFIG,
        'Sacramento': SACRAMENTO_CONFIG,
        'PRMS': PRMS_CONFIG
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Calibrating {name} structure...")
        print(f"{'='*50}")
        
        model = FUSE(config=config, dt=1.0, learnable_params=True)
        
        # Adjust initial state for this config
        state = model.get_initial_state(S1_init=150.0, S2_init=800.0)
        
        history = optimize_parameters(
            model, forcing, obs_runoff, state,
            n_epochs=n_epochs, lr=0.1, verbose=True
        )
        
        final_nse = history['nse'][-1]
        print(f"\nFinal NSE: {final_nse:.4f}")
        
        results[name] = {
            'model': model,
            'history': history,
            'final_nse': final_nse
        }
    
    return results


def plot_results(
    forcing: torch.Tensor,
    obs_runoff: torch.Tensor,
    results: dict,
    initial_state: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plot calibration results.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Forcing data
    ax = axes[0]
    days = np.arange(len(forcing))
    ax.bar(days, forcing[:, 0].numpy(), alpha=0.6, label='Precipitation', color='blue')
    ax.plot(days, forcing[:, 1].numpy(), 'g-', label='PET', linewidth=1)
    ax.set_ylabel('mm/day')
    ax.set_title('Forcing Data')
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(forcing))
    
    # Plot 2: Runoff comparison
    ax = axes[1]
    ax.plot(days, obs_runoff.numpy(), 'k-', label='Observed', linewidth=1.5)
    
    colors = ['blue', 'red', 'green', 'orange']
    for (name, result), color in zip(results.items(), colors):
        model = result['model']
        with torch.no_grad():
            state = model.get_initial_state(S1_init=150.0, S2_init=800.0)
            pred = model(state, forcing)
        ax.plot(days, pred.numpy(), '--', label=f'{name} (NSE={result["final_nse"]:.3f})', 
                color=color, alpha=0.8)
    
    ax.set_ylabel('Runoff (mm/day)')
    ax.set_title('Runoff: Observed vs Calibrated Models')
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(forcing))
    
    # Plot 3: NSE convergence
    ax = axes[2]
    for (name, result), color in zip(results.items(), colors):
        epochs = np.arange(1, len(result['history']['nse']) + 1)
        ax.plot(epochs, result['history']['nse'], '-', label=name, color=color)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NSE')
    ax.set_title('Calibration Convergence')
    ax.legend(loc='lower right')
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good fit threshold')
    ax.set_ylim(-0.5, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()


def main():
    """Main example workflow."""
    print("="*60)
    print("dFUSE Parameter Optimization Example")
    print("="*60)
    
    # 1. Generate synthetic forcing
    print("\n1. Generating synthetic forcing data...")
    n_days = 365
    forcing = generate_synthetic_forcing(n_days=n_days, seed=42)
    print(f"   Generated {n_days} days of forcing")
    print(f"   Precip: mean={forcing[:, 0].mean():.1f}, max={forcing[:, 0].max():.1f} mm/day")
    print(f"   PET: mean={forcing[:, 1].mean():.1f} mm/day")
    print(f"   Temp: mean={forcing[:, 2].mean():.1f}°C")
    
    # 2. Create "true" model and generate observations
    print("\n2. Generating synthetic observations from VIC model...")
    true_model = create_model('vic', learnable_params=False)
    
    # Set "true" parameters (slightly different from defaults)
    with torch.no_grad():
        true_model.raw_params[0] = 350.0   # S1_max
        true_model.raw_params[1] = 1200.0  # S2_max
        true_model.raw_params[6] = 15.0    # ku
        true_model.raw_params[12] = 35.0   # ks
    
    initial_state, obs_runoff = generate_observations(
        true_model, forcing, noise_std=0.3
    )
    print(f"   Observed runoff: mean={obs_runoff.mean():.2f} mm/day")
    
    # 3. Single model calibration demo
    print("\n3. Calibrating a single VIC model...")
    cal_model = create_model('vic', learnable_params=True)
    history = optimize_parameters(
        cal_model, forcing, obs_runoff, initial_state.clone(),
        n_epochs=50, lr=0.1, verbose=True
    )
    
    # 4. Compare model structures
    print("\n4. Comparing different model structures...")
    results = compare_model_structures(
        forcing, obs_runoff, initial_state.clone(), n_epochs=30
    )
    
    # 5. Summary
    print("\n" + "="*60)
    print("SUMMARY: Model Structure Comparison")
    print("="*60)
    print(f"{'Model':<15} {'Final NSE':>12} {'Rank':>8}")
    print("-"*40)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_nse'], reverse=True)
    for rank, (name, result) in enumerate(sorted_results, 1):
        print(f"{name:<15} {result['final_nse']:>12.4f} {rank:>8}")
    
    # 6. Plot results
    print("\n5. Plotting results...")
    try:
        plot_results(forcing, obs_runoff, results, initial_state, 
                    save_path='dfuse_calibration_results.png')
    except Exception as e:
        print(f"   Could not create plot: {e}")
        print("   (This may happen if matplotlib backend is not available)")
    
    print("\nExample completed!")
    return results


if __name__ == '__main__':
    results = main()
