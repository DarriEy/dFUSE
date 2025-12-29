#!/usr/bin/env python3
"""
compare_fortran.py - Compare cFUSE outputs against Fortran FUSE

This script provides comprehensive comparison between cFUSE (C++) and 
Fortran FUSE model outputs, including:
- Time series comparison plots
- Statistical metrics (RMSE, bias, correlation)
- Flux-by-flux breakdown
- State variable comparison
- Identification of largest discrepancies

Usage:
    python compare_fortran.py <cfuse_output> <fortran_output> [--plot] [--run-both]
    
Example:
    python compare_fortran.py \
        /path/to/cfuse_output.nc \
        /path/to/fortran_runs_def.nc \
        --plot --save-fig comparison.png
"""

import argparse
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    print("Error: netCDF4 required. Install with: pip install netCDF4")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")


@dataclass
class ComparisonStats:
    """Statistics for comparing two time series."""
    rmse: float
    bias: float
    mae: float
    correlation: float
    max_abs_diff: float
    max_diff_idx: int
    rel_rmse: float  # RMSE as fraction of mean
    
    def __str__(self):
        return (
            f"  RMSE:        {self.rmse:.6f}\n"
            f"  Bias:        {self.bias:.6f}\n"
            f"  MAE:         {self.mae:.6f}\n"
            f"  Correlation: {self.correlation:.6f}\n"
            f"  Max |diff|:  {self.max_abs_diff:.6f} (at index {self.max_diff_idx})\n"
            f"  Rel RMSE:    {self.rel_rmse:.2%}"
        )


def compute_stats(a: np.ndarray, b: np.ndarray) -> ComparisonStats:
    """Compute comparison statistics between two arrays."""
    # Handle NaN values
    valid = ~(np.isnan(a) | np.isnan(b))
    a_valid = a[valid]
    b_valid = b[valid]
    
    if len(a_valid) == 0:
        return ComparisonStats(
            rmse=np.nan, bias=np.nan, mae=np.nan,
            correlation=np.nan, max_abs_diff=np.nan,
            max_diff_idx=-1, rel_rmse=np.nan
        )
    
    diff = a_valid - b_valid
    rmse = np.sqrt(np.mean(diff**2))
    bias = np.mean(diff)
    mae = np.mean(np.abs(diff))
    
    # Correlation
    if np.std(a_valid) > 0 and np.std(b_valid) > 0:
        correlation = np.corrcoef(a_valid, b_valid)[0, 1]
    else:
        correlation = np.nan
    
    # Max difference (on full arrays with NaN handling)
    full_diff = np.abs(a - b)
    full_diff = np.where(np.isnan(full_diff), 0, full_diff)
    max_diff_idx = int(np.argmax(full_diff))
    max_abs_diff = full_diff[max_diff_idx]
    
    # Relative RMSE
    mean_val = np.mean(np.abs(b_valid))
    rel_rmse = rmse / mean_val if mean_val > 0 else np.nan
    
    return ComparisonStats(
        rmse=rmse, bias=bias, mae=mae,
        correlation=correlation, max_abs_diff=max_abs_diff,
        max_diff_idx=max_diff_idx, rel_rmse=rel_rmse
    )


def load_dfuse_output(filepath: Path) -> Dict[str, np.ndarray]:
    """Load cFUSE NetCDF output."""
    data = {}
    with nc.Dataset(filepath, 'r') as ds:
        # Print available variables for debugging
        print(f"\ncFUSE variables: {list(ds.variables.keys())}")
        
        for var in ds.variables:
            data[var] = ds.variables[var][:]
            
    return data


def load_fortran_output(filepath: Path) -> Dict[str, np.ndarray]:
    """Load Fortran FUSE NetCDF output."""
    data = {}
    with nc.Dataset(filepath, 'r') as ds:
        # Print available variables for debugging
        print(f"\nFortran FUSE variables: {list(ds.variables.keys())}")
        
        for var in ds.variables:
            arr = ds.variables[var][:]
            # Squeeze out singleton dimensions (common in Fortran FUSE output)
            if hasattr(arr, 'squeeze'):
                arr = arr.squeeze()
            data[var] = arr
            
    return data


def find_matching_variables(dfuse_data: Dict, fortran_data: Dict) -> List[Tuple[str, str]]:
    """Find matching variable pairs between cFUSE and Fortran outputs."""
    # Known mappings between cFUSE and Fortran variable names
    mappings = [
        # (dfuse_name, fortran_name)
        ('q_routed', 'q_instnt'),      # Instantaneous runoff
        ('q_routed', 'q_routed'),      # Routed runoff
        ('streamflow', 'q_routed'),
        ('q_total', 'q_instnt'),
        ('qsx', 'qsatexcs'),           # Surface runoff
        ('qb', 'qbasflow'),            # Baseflow  
        ('qif', 'qintflw'),            # Interflow
        ('q12', 'qpercolat'),          # Percolation
        ('e_total', 'evap_1a'),        # Evaporation
        ('S1', 'watr_1'),              # Upper layer storage
        ('S2', 'watr_2'),              # Lower layer storage
        ('SWE', 'swe_tot'),            # Snow water equivalent
    ]
    
    matches = []
    for dfuse_name, fortran_name in mappings:
        if dfuse_name in dfuse_data and fortran_name in fortran_data:
            matches.append((dfuse_name, fortran_name))
    
    return matches


def compare_outputs(
    dfuse_path: Path,
    fortran_path: Path,
    verbose: bool = True
) -> Dict[str, ComparisonStats]:
    """Compare cFUSE and Fortran FUSE outputs."""
    
    print(f"\n{'='*60}")
    print("cFUSE vs Fortran FUSE Comparison")
    print(f"{'='*60}")
    print(f"\ncFUSE output:   {dfuse_path}")
    print(f"Fortran output: {fortran_path}")
    
    # Load data
    dfuse_data = load_dfuse_output(dfuse_path)
    fortran_data = load_fortran_output(fortran_path)
    
    # Find the main runoff variable
    # cFUSE typically outputs q_routed, Fortran outputs q_instnt or q_routed
    dfuse_q_var = None
    for var in ['q_routed', 'streamflow', 'q_total', 'runoff']:
        if var in dfuse_data:
            dfuse_q_var = var
            break
    
    fortran_q_var = None
    for var in ['q_instnt', 'q_routed', 'q_sim']:
        if var in fortran_data:
            fortran_q_var = var
            break
    
    if dfuse_q_var is None or fortran_q_var is None:
        print("\nError: Could not find runoff variables in outputs")
        return {}
    
    print(f"\nComparing: cFUSE[{dfuse_q_var}] vs Fortran[{fortran_q_var}]")
    
    # Get the time series
    dfuse_q = np.array(dfuse_data[dfuse_q_var]).flatten()
    fortran_q = np.array(fortran_data[fortran_q_var]).flatten()
    
    # Handle length mismatch
    min_len = min(len(dfuse_q), len(fortran_q))
    if len(dfuse_q) != len(fortran_q):
        print(f"\nWarning: Length mismatch - cFUSE: {len(dfuse_q)}, Fortran: {len(fortran_q)}")
        print(f"         Comparing first {min_len} timesteps")
    
    dfuse_q = dfuse_q[:min_len]
    fortran_q = fortran_q[:min_len]
    
    # Compute statistics
    stats = compute_stats(dfuse_q, fortran_q)
    
    print(f"\n{'='*60}")
    print("RUNOFF COMPARISON STATISTICS")
    print(f"{'='*60}")
    print(stats)
    
    # Summary statistics for each series
    print(f"\n{'='*60}")
    print("TIME SERIES SUMMARIES")
    print(f"{'='*60}")
    print(f"\ncFUSE [{dfuse_q_var}]:")
    print(f"  Mean:   {np.nanmean(dfuse_q):.4f} mm/day")
    print(f"  Std:    {np.nanstd(dfuse_q):.4f} mm/day")
    print(f"  Max:    {np.nanmax(dfuse_q):.4f} mm/day")
    print(f"  Min:    {np.nanmin(dfuse_q):.4f} mm/day")
    
    print(f"\nFortran [{fortran_q_var}]:")
    print(f"  Mean:   {np.nanmean(fortran_q):.4f} mm/day")
    print(f"  Std:    {np.nanstd(fortran_q):.4f} mm/day")
    print(f"  Max:    {np.nanmax(fortran_q):.4f} mm/day")
    print(f"  Min:    {np.nanmin(fortran_q):.4f} mm/day")
    
    # Find worst discrepancies
    diff = dfuse_q - fortran_q
    abs_diff = np.abs(diff)
    worst_indices = np.argsort(abs_diff)[-10:][::-1]
    
    print(f"\n{'='*60}")
    print("TOP 10 LARGEST DISCREPANCIES")
    print(f"{'='*60}")
    print(f"{'Index':>8} {'cFUSE':>12} {'Fortran':>12} {'Diff':>12} {'%Diff':>10}")
    print("-" * 56)
    for idx in worst_indices:
        pct_diff = 100 * diff[idx] / fortran_q[idx] if fortran_q[idx] != 0 else np.inf
        print(f"{idx:>8} {dfuse_q[idx]:>12.4f} {fortran_q[idx]:>12.4f} {diff[idx]:>12.4f} {pct_diff:>9.1f}%")
    
    # Check for systematic patterns
    print(f"\n{'='*60}")
    print("DIAGNOSTIC CHECKS")
    print(f"{'='*60}")
    
    # Check spinup (first year)
    spinup_days = 365
    if min_len > spinup_days:
        spinup_stats = compute_stats(dfuse_q[:spinup_days], fortran_q[:spinup_days])
        post_spinup_stats = compute_stats(dfuse_q[spinup_days:], fortran_q[spinup_days:])
        
        print(f"\nSpinup period (first {spinup_days} days):")
        print(f"  RMSE: {spinup_stats.rmse:.6f}, Bias: {spinup_stats.bias:.6f}")
        print(f"\nPost-spinup period:")
        print(f"  RMSE: {post_spinup_stats.rmse:.6f}, Bias: {post_spinup_stats.bias:.6f}")
        
        if spinup_stats.rmse > 2 * post_spinup_stats.rmse:
            print("\n  ⚠️  Spinup RMSE much larger than post-spinup - likely initialization difference")
    
    # Check for scale issues
    scale_ratio = np.nanmean(dfuse_q) / np.nanmean(fortran_q) if np.nanmean(fortran_q) != 0 else np.nan
    print(f"\nScale ratio (cFUSE/Fortran): {scale_ratio:.4f}")
    if abs(scale_ratio - 1.0) > 0.1:
        print("  ⚠️  Significant scale difference detected - check flux aggregation")
    
    # Check timing (peak alignment)
    dfuse_peak = np.argmax(dfuse_q)
    fortran_peak = np.argmax(fortran_q)
    print(f"\nPeak timing: cFUSE at day {dfuse_peak}, Fortran at day {fortran_peak}")
    if abs(dfuse_peak - fortran_peak) > 5:
        print("  ⚠️  Peak timing differs by more than 5 days")
    
    # Check initial states from Fortran output
    print(f"\n{'='*60}")
    print("FORTRAN INITIAL STATES (day 0)")
    print(f"{'='*60}")
    state_vars = ['tens_1', 'tens_1a', 'tens_1b', 'free_1', 'watr_1', 
                  'tens_2', 'free_2', 'free_2a', 'free_2b', 'watr_2']
    for var in state_vars:
        if var in fortran_data:
            val = np.array(fortran_data[var]).flatten()
            if len(val) > 0:
                print(f"  {var:12s}: {val[0]:8.3f} mm (initial) -> {val[-1]:8.3f} mm (final)")
    
    return {
        'runoff': stats,
        'dfuse_q': dfuse_q,
        'fortran_q': fortran_q,
        'dfuse_data': dfuse_data,
        'fortran_data': fortran_data
    }


def plot_comparison(
    results: Dict,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """Create comparison plots."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    dfuse_q = results['dfuse_q']
    fortran_q = results['fortran_q']
    n_days = len(dfuse_q)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 1. Full time series comparison
    ax = axes[0, 0]
    ax.plot(fortran_q, 'b-', alpha=0.7, linewidth=0.5, label='Fortran FUSE')
    ax.plot(dfuse_q, 'r-', alpha=0.7, linewidth=0.5, label='cFUSE')
    ax.set_xlabel('Day')
    ax.set_ylabel('Runoff (mm/day)')
    ax.set_title('Full Time Series Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. First year (spinup) detail
    ax = axes[0, 1]
    spinup_days = min(365, n_days)
    ax.plot(fortran_q[:spinup_days], 'b-', linewidth=1, label='Fortran FUSE')
    ax.plot(dfuse_q[:spinup_days], 'r-', linewidth=1, label='cFUSE')
    ax.set_xlabel('Day')
    ax.set_ylabel('Runoff (mm/day)')
    ax.set_title('First Year (Spinup Period)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Scatter plot
    ax = axes[1, 0]
    ax.scatter(fortran_q, dfuse_q, alpha=0.3, s=1)
    max_val = max(np.nanmax(fortran_q), np.nanmax(dfuse_q))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='1:1 line')
    ax.set_xlabel('Fortran FUSE (mm/day)')
    ax.set_ylabel('cFUSE (mm/day)')
    ax.set_title('Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 4. Difference time series
    ax = axes[1, 1]
    diff = dfuse_q - fortran_q
    ax.plot(diff, 'g-', linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(y=np.mean(diff), color='r', linestyle='-', linewidth=1, label=f'Mean bias: {np.mean(diff):.3f}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Difference (mm/day)')
    ax.set_title('cFUSE - Fortran FUSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Histogram of differences
    ax = axes[2, 0]
    ax.hist(diff, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax.axvline(x=np.mean(diff), color='r', linestyle='-', linewidth=2, label=f'Mean: {np.mean(diff):.3f}')
    ax.set_xlabel('Difference (mm/day)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Differences')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Cumulative runoff
    ax = axes[2, 1]
    ax.plot(np.cumsum(fortran_q), 'b-', linewidth=1, label='Fortran FUSE')
    ax.plot(np.cumsum(dfuse_q), 'r-', linewidth=1, label='cFUSE')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Runoff (mm)')
    ax.set_title('Cumulative Runoff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    if show:
        plt.show()


def run_both_models(
    file_manager: Path,
    basin_id: str,
    run_mode: str = "run_def",
    cfuse_binary: Optional[Path] = None,
    fortran_binary: Optional[Path] = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """Run both cFUSE and Fortran FUSE models."""
    
    print(f"\n{'='*60}")
    print("Running Both Models")
    print(f"{'='*60}")
    
    dfuse_output = None
    fortran_output = None
    
    # Run cFUSE
    if cfuse_binary and cfuse_binary.exists():
        print(f"\nRunning cFUSE: {cfuse_binary}")
        try:
            result = subprocess.run(
                [str(cfuse_binary), str(file_manager), basin_id, run_mode],
                capture_output=True, text=True, timeout=300
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"cFUSE error: {result.stderr}")
        except Exception as e:
            print(f"cFUSE failed: {e}")
    
    # Run Fortran FUSE
    if fortran_binary and fortran_binary.exists():
        print(f"\nRunning Fortran FUSE: {fortran_binary}")
        try:
            result = subprocess.run(
                [str(fortran_binary), str(file_manager), basin_id, run_mode],
                capture_output=True, text=True, timeout=300
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"Fortran FUSE error: {result.stderr}")
        except Exception as e:
            print(f"Fortran FUSE failed: {e}")
    
    return dfuse_output, fortran_output


def main():
    parser = argparse.ArgumentParser(
        description='Compare cFUSE and Fortran FUSE model outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare existing outputs
  python compare_fortran.py cfuse_output.nc fortran_runs_def.nc --plot
  
  # Compare and save figure
  python compare_fortran.py cfuse_output.nc fortran_runs_def.nc --plot --save-fig comparison.png
        """
    )
    
    parser.add_argument('dfuse_output', type=Path,
                        help='Path to cFUSE NetCDF output file')
    parser.add_argument('fortran_output', type=Path,
                        help='Path to Fortran FUSE NetCDF output file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots')
    parser.add_argument('--save-fig', type=Path, default=None,
                        help='Save figure to this path')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots interactively')
    parser.add_argument('--run-both', action='store_true',
                        help='Run both models before comparing')
    parser.add_argument('--file-manager', type=Path, default=None,
                        help='Path to FUSE file manager (for --run-both)')
    parser.add_argument('--basin-id', type=str, default=None,
                        help='Basin ID (for --run-both)')
    parser.add_argument('--cfuse-binary', type=Path, default=None,
                        help='Path to cFUSE binary')
    parser.add_argument('--fortran-binary', type=Path, default=None,
                        help='Path to Fortran FUSE binary')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.dfuse_output.exists():
        print(f"Error: cFUSE output not found: {args.dfuse_output}")
        sys.exit(1)
    if not args.fortran_output.exists():
        print(f"Error: Fortran output not found: {args.fortran_output}")
        sys.exit(1)
    
    # Run comparison
    results = compare_outputs(args.dfuse_output, args.fortran_output)
    
    # Generate plots if requested
    if args.plot and results:
        plot_comparison(
            results,
            save_path=args.save_fig,
            show=not args.no_show
        )
    
    print(f"\n{'='*60}")
    print("Comparison complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
