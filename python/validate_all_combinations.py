#!/usr/bin/env python3
"""
validate_all_combinations.py - Comprehensive cFUSE vs Fortran FUSE Validation

This script systematically tests all valid combinations of FUSE model decisions,
comparing cFUSE outputs against Fortran FUSE to validate the implementation.

Features:
- Tests all valid model decision combinations
- Measures wall-clock execution time for both implementations
- Computes comparison statistics (RMSE, bias, correlation)
- Generates summary report and visualization
- Identifies which physics implementations need attention

Usage:
    python validate_all_combinations.py \
        --cfuse-binary /path/to/cfuse \
        --fortran-binary /path/to/fuse.exe \
        --file-manager /path/to/fm_catch.txt \
        --basin-id Bow_at_Banff_lumped_era5 \
        --output-dir ./validation_results
"""

import argparse
import subprocess
import sys
import time
import shutil
import tempfile
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import json

try:
    import netCDF4 as nc
except ImportError:
    print("Error: netCDF4 required. Install with: pip install netCDF4")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")


# ============================================================================
# MODEL DECISION OPTIONS
# ============================================================================

MODEL_DECISIONS = {
    'RFERR': ['additive_e', 'multiplc_e'],
    # Note: tension2_1 is NOT reliably supported by Fortran FUSE - runs in 0.01s vs 0.7s normal
    # This indicates Fortran is not actually computing with this architecture
    'ARCH1': ['onestate_1', 'tension1_1'],
    # Note: topmdexp_2 is NOT supported by Fortran FUSE (only pre-defined combinations)
    # cFUSE implements it but cannot be validated against Fortran
    'ARCH2': ['fixedsiz_2', 'unlimfrc_2', 'unlimpow_2', 'tens2pll_2'],
    # Note: tmdl_param requires TOPMODEL baseflow which Fortran doesn't support
    'QSURF': ['arno_x_vic', 'prms_varnt'],
    'QPERC': ['perc_f2sat', 'perc_w2sat', 'perc_lower'],
    'ESOIL': ['sequential', 'rootweight'],
    'QINTF': ['intflwnone', 'intflwsome'],
    'Q_TDH': ['no_routing', 'rout_gamma'],
    'SNOWM': ['no_snowmod', 'temp_index'],
}

# Total combinations: 2 × 4 × 2 × 3 × 2 × 2 × 2 × 2 = 768 (with rferr=additive only: 384)

def is_valid_combination(decisions: Dict[str, str]) -> bool:
    """Check if a combination of decisions is valid for Fortran FUSE.
    
    Based on validation results, certain combinations are not supported
    by Fortran FUSE (it runs in 0.01s instead of 0.7s and produces stale output).
    
    Unsupported combinations include:
    - tension1_1 + perc_w2sat: Fortran doesn't support this architecture+percolation combo
    - tension2_1: Not in our test set (removed from ARCH1 options)
    """
    # tension1_1 + perc_w2sat is not supported by Fortran
    if decisions['ARCH1'] == 'tension1_1' and decisions['QPERC'] == 'perc_w2sat':
        return False
    
    return True


@dataclass
class ValidationResult:
    """Results from validating a single model configuration."""
    decisions: Dict[str, str]
    config_name: str
    dfuse_time: float = 0.0
    fortran_time: float = 0.0
    rmse: float = np.nan
    bias: float = np.nan
    correlation: float = np.nan
    max_diff: float = np.nan
    dfuse_mean: float = np.nan
    fortran_mean: float = np.nan
    dfuse_success: bool = False
    fortran_success: bool = False
    error_message: str = ""


def generate_decisions_file(
    template_path: Path,
    output_path: Path,
    decisions: Dict[str, str]
) -> None:
    """Generate a FUSE decisions file with specified options."""
    lines = [
        "(a10,1x,a5)          ! format code (model component, model decision)",
        f"{decisions['RFERR']:10s} RFERR ! (1) rainfall error",
        f"{decisions['ARCH1']:10s} ARCH1 ! (2) upper-layer architecture",
        f"{decisions['ARCH2']:10s} ARCH2 ! (3) lower-layer architecture and baseflow",
        f"{decisions['QSURF']:10s} QSURF ! (4) surface runoff",
        f"{decisions['QPERC']:10s} QPERC ! (5) percolation",
        f"{decisions['ESOIL']:10s} ESOIL ! (6) evaporation",
        f"{decisions['QINTF']:10s} QINTF ! (7) interflow",
        f"{decisions['Q_TDH']:10s} Q_TDH ! (8) time delay in runoff",
        f"{decisions['SNOWM']:10s} SNOWM ! (9) snow model",
        "0                    ! naming convention for model (0=full, 1=index + numrx)",
    ]
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_model(
    binary_path: Path,
    file_manager: Path,
    basin_id: str,
    run_mode: str = "run_def",
    timeout: int = 300
) -> Tuple[bool, float, str]:
    """Run a model binary and return (success, elapsed_time, error_message)."""
    try:
        start_time = time.perf_counter()
        result = subprocess.run(
            [str(binary_path), str(file_manager), basin_id, run_mode],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.perf_counter() - start_time
        
        if result.returncode != 0:
            return False, elapsed, result.stderr[:500]
        
        return True, elapsed, ""
        
    except subprocess.TimeoutExpired:
        return False, timeout, "Timeout expired"
    except Exception as e:
        return False, 0.0, str(e)


def compute_comparison_stats(
    dfuse_path: Path,
    fortran_path: Path
) -> Dict[str, float]:
    """Compute comparison statistics between cFUSE and Fortran outputs."""
    stats = {
        'rmse': np.nan,
        'bias': np.nan,
        'correlation': np.nan,
        'max_diff': np.nan,
        'dfuse_mean': np.nan,
        'fortran_mean': np.nan,
    }
    
    try:
        # Load cFUSE output
        with nc.Dataset(dfuse_path, 'r') as ds:
            dfuse_q = np.array(ds.variables['q_routed'][:]).flatten()
        
        # Load Fortran output
        with nc.Dataset(fortran_path, 'r') as ds:
            # Try different variable names
            for var_name in ['q_instnt', 'q_routed', 'q_sim']:
                if var_name in ds.variables:
                    fortran_q = np.array(ds.variables[var_name][:]).squeeze()
                    break
            else:
                return stats
        
        # Handle length mismatch
        min_len = min(len(dfuse_q), len(fortran_q))
        dfuse_q = dfuse_q[:min_len]
        fortran_q = fortran_q[:min_len]
        
        # Handle NaN values
        valid = ~(np.isnan(dfuse_q) | np.isnan(fortran_q))
        if not np.any(valid):
            return stats
        
        dfuse_valid = dfuse_q[valid]
        fortran_valid = fortran_q[valid]
        
        diff = dfuse_valid - fortran_valid
        
        stats['rmse'] = np.sqrt(np.mean(diff**2))
        stats['bias'] = np.mean(diff)
        stats['max_diff'] = np.max(np.abs(diff))
        stats['dfuse_mean'] = np.mean(dfuse_valid)
        stats['fortran_mean'] = np.mean(fortran_valid)
        
        if np.std(dfuse_valid) > 0 and np.std(fortran_valid) > 0:
            stats['correlation'] = np.corrcoef(dfuse_valid, fortran_valid)[0, 1]
        
    except Exception as e:
        print(f"  Warning: Could not compute stats: {e}")
    
    return stats


def validate_configuration(
    decisions: Dict[str, str],
    cfuse_binary: Path,
    fortran_binary: Path,
    file_manager: Path,
    basin_id: str,
    settings_dir: Path,
    output_dir: Path,
    decisions_file: str = "fuse_zDecisions_run_1.txt"
) -> ValidationResult:
    """Validate a single model configuration."""
    
    # Create config name (abbreviated for readability)
    config_name = "_".join([
        decisions['ARCH1'][:4],
        decisions['ARCH2'][:4],
        decisions['QSURF'][:4],
        decisions['QPERC'][5:9],  # f2sat, w2sat, lowe
        decisions['ESOIL'][:4],
        decisions['QINTF'][6:10],  # none, some
        decisions['Q_TDH'][:4],   # no_r, rout
        decisions['SNOWM'][:4],   # no_s, temp
    ])
    
    result = ValidationResult(
        decisions=decisions.copy(),
        config_name=config_name
    )
    
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    # Backup original decisions file
    decisions_path = settings_dir / decisions_file
    backup_path = settings_dir / f"{decisions_file}.backup"
    
    try:
        shutil.copy(decisions_path, backup_path)
        
        # Generate new decisions file
        generate_decisions_file(decisions_path, decisions_path, decisions)
        
        # Run Fortran FUSE
        print("  Running Fortran FUSE...")
        result.fortran_success, result.fortran_time, error = run_model(
            fortran_binary, file_manager, basin_id
        )
        if not result.fortran_success:
            result.error_message = f"Fortran: {error}"
            print(f"  ❌ Fortran failed: {error[:100]}")
        else:
            print(f"  ✓ Fortran completed in {result.fortran_time:.3f}s")
        
        # Run cFUSE
        print("  Running cFUSE...")
        result.dfuse_success, result.dfuse_time, error = run_model(
            cfuse_binary, file_manager, basin_id
        )
        if not result.dfuse_success:
            result.error_message += f" cFUSE: {error}"
            print(f"  ❌ cFUSE failed: {error[:100]}")
        else:
            print(f"  ✓ cFUSE completed in {result.dfuse_time:.3f}s")
        
        # Compare outputs if both succeeded
        if result.fortran_success and result.dfuse_success:
            # Find output files
            dfuse_output = output_dir / f"{basin_id}_run_1_cfuse.nc"
            fortran_output = output_dir / f"{basin_id}_run_1_runs_def.nc"
            
            if dfuse_output.exists() and fortran_output.exists():
                stats = compute_comparison_stats(dfuse_output, fortran_output)
                result.rmse = stats['rmse']
                result.bias = stats['bias']
                result.correlation = stats['correlation']
                result.max_diff = stats['max_diff']
                result.dfuse_mean = stats['dfuse_mean']
                result.fortran_mean = stats['fortran_mean']
                
                print(f"  Comparison: RMSE={result.rmse:.4f}, Bias={result.bias:.4f}, r={result.correlation:.4f}")
            else:
                print(f"  Warning: Output files not found")
                
    finally:
        # Restore original decisions file
        if backup_path.exists():
            shutil.copy(backup_path, decisions_path)
            backup_path.unlink()
    
    return result


def generate_all_combinations(
    skip_snow: bool = False,
    skip_interflow: bool = False,
    skip_routing: bool = False,
    skip_rferr: bool = True,  # Usually just use additive
) -> List[Dict[str, str]]:
    """Generate all valid model decision combinations.
    
    Note: topmdexp_2 and tmdl_param are NOT included because Fortran FUSE
    does not support these as standalone options (only pre-defined combinations).
    cFUSE implements TOPMODEL but cannot be validated against Fortran.
    
    Full combinations: 1 × 3 × 4 × 2 × 3 × 2 × 2 × 2 × 2 = 576 (default, rferr=additive only)
    With all RFERR: 2 × 3 × 4 × 2 × 3 × 2 × 2 × 2 × 2 = 1,152
    """
    
    rferr_options = MODEL_DECISIONS['RFERR'] if not skip_rferr else ['additive_e']
    arch1_options = MODEL_DECISIONS['ARCH1']
    arch2_options = MODEL_DECISIONS['ARCH2']
    qsurf_options = MODEL_DECISIONS['QSURF']
    qperc_options = MODEL_DECISIONS['QPERC']
    esoil_options = MODEL_DECISIONS['ESOIL']
    qintf_options = MODEL_DECISIONS['QINTF'] if not skip_interflow else ['intflwnone']
    q_tdh_options = MODEL_DECISIONS['Q_TDH'] if not skip_routing else ['no_routing']
    snowm_options = MODEL_DECISIONS['SNOWM'] if not skip_snow else ['no_snowmod']
    
    combinations = []
    
    for rferr, arch1, arch2, qsurf, qperc, esoil, qintf, q_tdh, snowm in itertools.product(
        rferr_options, arch1_options, arch2_options, qsurf_options, qperc_options,
        esoil_options, qintf_options, q_tdh_options, snowm_options
    ):
        decisions = {
            'RFERR': rferr,
            'ARCH1': arch1,
            'ARCH2': arch2,
            'QSURF': qsurf,
            'QPERC': qperc,
            'ESOIL': esoil,
            'QINTF': qintf,
            'Q_TDH': q_tdh,
            'SNOWM': snowm,
        }
        
        if is_valid_combination(decisions):
            combinations.append(decisions)
    
    return combinations


def generate_summary_report(
    results: List[ValidationResult],
    output_path: Path
) -> None:
    """Generate a summary report of all validation results."""
    
    successful = [r for r in results if r.dfuse_success and r.fortran_success]
    failed = [r for r in results if not (r.dfuse_success and r.fortran_success)]
    
    # Sort by RMSE
    successful.sort(key=lambda r: r.rmse if not np.isnan(r.rmse) else float('inf'))
    
    lines = [
        "=" * 80,
        "cFUSE vs Fortran FUSE Validation Report",
        "=" * 80,
        "",
        f"Total configurations tested: {len(results)}",
        f"Successful: {len(successful)}",
        f"Failed: {len(failed)}",
        "",
    ]
    
    if successful:
        # Timing summary
        dfuse_times = [r.dfuse_time for r in successful]
        fortran_times = [r.fortran_time for r in successful]
        
        lines.extend([
            "=" * 80,
            "TIMING SUMMARY",
            "=" * 80,
            f"cFUSE:   mean={np.mean(dfuse_times):.4f}s, total={np.sum(dfuse_times):.2f}s",
            f"Fortran: mean={np.mean(fortran_times):.4f}s, total={np.sum(fortran_times):.2f}s",
            f"Speedup: {np.mean(fortran_times)/np.mean(dfuse_times):.1f}x",
            "",
        ])
        
        # Best matches
        lines.extend([
            "=" * 80,
            "BEST MATCHES (lowest RMSE)",
            "=" * 80,
        ])
        for r in successful[:10]:
            lines.append(
                f"  {r.config_name}: RMSE={r.rmse:.4f}, bias={r.bias:.4f}, r={r.correlation:.4f}"
            )
        
        # Worst matches
        lines.extend([
            "",
            "=" * 80,
            "WORST MATCHES (highest RMSE)",
            "=" * 80,
        ])
        for r in successful[-10:]:
            lines.append(
                f"  {r.config_name}: RMSE={r.rmse:.4f}, bias={r.bias:.4f}, r={r.correlation:.4f}"
            )
        
        # By architecture
        lines.extend([
            "",
            "=" * 80,
            "SUMMARY BY UPPER LAYER ARCHITECTURE",
            "=" * 80,
        ])
        for arch in MODEL_DECISIONS['ARCH1']:
            arch_results = [r for r in successful if r.decisions['ARCH1'] == arch]
            if arch_results:
                rmses = [r.rmse for r in arch_results if not np.isnan(r.rmse)]
                if rmses:
                    lines.append(f"  {arch}: n={len(arch_results)}, mean_RMSE={np.mean(rmses):.4f}")
        
        lines.extend([
            "",
            "=" * 80,
            "SUMMARY BY LOWER LAYER ARCHITECTURE",
            "=" * 80,
        ])
        for arch in MODEL_DECISIONS['ARCH2']:
            arch_results = [r for r in successful if r.decisions['ARCH2'] == arch]
            if arch_results:
                rmses = [r.rmse for r in arch_results if not np.isnan(r.rmse)]
                if rmses:
                    lines.append(f"  {arch}: n={len(arch_results)}, mean_RMSE={np.mean(rmses):.4f}")
        
        lines.extend([
            "",
            "=" * 80,
            "SUMMARY BY ROUTING",
            "=" * 80,
        ])
        for routing in MODEL_DECISIONS['Q_TDH']:
            routing_results = [r for r in successful if r.decisions.get('Q_TDH') == routing]
            if routing_results:
                rmses = [r.rmse for r in routing_results if not np.isnan(r.rmse)]
                if rmses:
                    lines.append(f"  {routing}: n={len(routing_results)}, mean_RMSE={np.mean(rmses):.4f}")
    
    if failed:
        lines.extend([
            "",
            "=" * 80,
            "FAILED CONFIGURATIONS",
            "=" * 80,
        ])
        for r in failed:
            lines.append(f"  {r.config_name}: {r.error_message[:60]}")
    
    # Full results table
    lines.extend([
        "",
        "=" * 80,
        "FULL RESULTS TABLE",
        "=" * 80,
        f"{'Config':<50} {'RMSE':>10} {'Bias':>10} {'Corr':>8} {'cFUSE(s)':>10} {'Fortran(s)':>12}",
        "-" * 100,
    ])
    
    for r in results:
        lines.append(
            f"{r.config_name:<50} "
            f"{r.rmse:>10.4f} "
            f"{r.bias:>10.4f} "
            f"{r.correlation:>8.4f} "
            f"{r.dfuse_time:>10.4f} "
            f"{r.fortran_time:>12.4f}"
        )
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nReport saved to: {output_path}")


def plot_validation_summary(
    results: List[ValidationResult],
    output_path: Path
) -> None:
    """Generate visualization of validation results."""
    if not HAS_MATPLOTLIB:
        return
    
    successful = [r for r in results if r.dfuse_success and r.fortran_success and not np.isnan(r.rmse)]
    
    if not successful:
        print("No successful results to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. RMSE distribution
    ax = axes[0, 0]
    rmses = [r.rmse for r in successful]
    ax.hist(rmses, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(rmses), color='r', linestyle='--', label=f'Median: {np.median(rmses):.3f}')
    ax.set_xlabel('RMSE (mm/day)')
    ax.set_ylabel('Count')
    ax.set_title('RMSE Distribution')
    ax.legend()
    
    # 2. Bias distribution
    ax = axes[0, 1]
    biases = [r.bias for r in successful]
    ax.hist(biases, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(np.median(biases), color='r', linestyle='--', label=f'Median: {np.median(biases):.3f}')
    ax.set_xlabel('Bias (mm/day)')
    ax.set_ylabel('Count')
    ax.set_title('Bias Distribution')
    ax.legend()
    
    # 3. Correlation distribution
    ax = axes[0, 2]
    corrs = [r.correlation for r in successful if not np.isnan(r.correlation)]
    ax.hist(corrs, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(corrs), color='r', linestyle='--', label=f'Median: {np.median(corrs):.3f}')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Correlation Distribution')
    ax.legend()
    
    # 4. Timing comparison
    ax = axes[1, 0]
    dfuse_times = [r.dfuse_time for r in successful]
    fortran_times = [r.fortran_time for r in successful]
    ax.scatter(fortran_times, dfuse_times, alpha=0.5, s=20)
    max_time = max(max(dfuse_times), max(fortran_times))
    ax.plot([0, max_time], [0, max_time], 'k--', linewidth=0.5, label='1:1')
    ax.set_xlabel('Fortran Time (s)')
    ax.set_ylabel('cFUSE Time (s)')
    ax.set_title('Execution Time Comparison')
    ax.legend()
    
    # 5. RMSE by architecture
    ax = axes[1, 1]
    arch_data = {}
    for arch in MODEL_DECISIONS['ARCH1']:
        arch_rmses = [r.rmse for r in successful if r.decisions['ARCH1'] == arch]
        if arch_rmses:
            arch_data[arch] = arch_rmses
    
    if arch_data:
        positions = range(len(arch_data))
        bp = ax.boxplot(arch_data.values(), positions=positions)
        ax.set_xticks(positions)
        ax.set_xticklabels(arch_data.keys(), rotation=45, ha='right')
        ax.set_ylabel('RMSE (mm/day)')
        ax.set_title('RMSE by Upper Layer Architecture')
    
    # 6. RMSE by lower architecture
    ax = axes[1, 2]
    arch_data = {}
    for arch in MODEL_DECISIONS['ARCH2']:
        arch_rmses = [r.rmse for r in successful if r.decisions['ARCH2'] == arch]
        if arch_rmses:
            arch_data[arch] = arch_rmses
    
    if arch_data:
        positions = range(len(arch_data))
        bp = ax.boxplot(arch_data.values(), positions=positions)
        ax.set_xticks(positions)
        ax.set_xticklabels(arch_data.keys(), rotation=45, ha='right')
        ax.set_ylabel('RMSE (mm/day)')
        ax.set_title('RMSE by Lower Layer Architecture')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate cFUSE against Fortran FUSE for all model combinations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--cfuse-binary', type=Path, required=True,
                        help='Path to cFUSE binary')
    parser.add_argument('--fortran-binary', type=Path, required=True,
                        help='Path to Fortran FUSE binary')
    parser.add_argument('--file-manager', type=Path, required=True,
                        help='Path to FUSE file manager')
    parser.add_argument('--basin-id', type=str, required=True,
                        help='Basin identifier')
    parser.add_argument('--output-dir', type=Path, default=Path('./validation_results'),
                        help='Output directory for results')
    parser.add_argument('--skip-snow', action='store_true',
                        help='Skip snow model variations (use no_snowmod only)')
    parser.add_argument('--skip-interflow', action='store_true',
                        help='Skip interflow variations (use intflwnone only)')
    parser.add_argument('--skip-routing', action='store_true',
                        help='Skip routing variations (use no_routing only)')
    parser.add_argument('--include-rferr', action='store_true',
                        help='Include rainfall error variations (default: additive_e only)')
    parser.add_argument('--max-configs', type=int, default=None,
                        help='Maximum number of configurations to test')
    parser.add_argument('--config-filter', type=str, default=None,
                        help='Filter configs by substring (e.g., "tension1")')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.cfuse_binary.exists():
        print(f"Error: cFUSE binary not found: {args.cfuse_binary}")
        sys.exit(1)
    if not args.fortran_binary.exists():
        print(f"Error: Fortran binary not found: {args.fortran_binary}")
        sys.exit(1)
    if not args.file_manager.exists():
        print(f"Error: File manager not found: {args.file_manager}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse file manager to get paths
    with open(args.file_manager, 'r') as f:
        content = f.read()
    
    # Extract settings and output paths from file manager
    import re
    paths = re.findall(r"'([^']+)'", content)
    if len(paths) < 3:
        print("Error: Could not parse file manager")
        sys.exit(1)
    
    settings_dir = Path(paths[0])
    output_dir = Path(paths[2])
    
    print(f"Settings directory: {settings_dir}")
    print(f"Model output directory: {output_dir}")
    
    # Generate all combinations
    combinations = generate_all_combinations(
        skip_snow=args.skip_snow,
        skip_interflow=args.skip_interflow,
        skip_routing=args.skip_routing,
        skip_rferr=not args.include_rferr,
    )
    
    print(f"\nCombination space:")
    print(f"  RFERR: {'all (2)' if args.include_rferr else 'additive_e only'}")
    print(f"  ARCH1: all (3)")
    print(f"  ARCH2: 4 (excluding topmdexp_2 - not supported by Fortran)")
    print(f"  QSURF: 2 (excluding tmdl_param - requires TOPMODEL)")
    print(f"  QPERC: all (3)")
    print(f"  ESOIL: all (2)")
    print(f"  QINTF: {'all (2)' if not args.skip_interflow else 'intflwnone only'}")
    print(f"  Q_TDH: {'all (2)' if not args.skip_routing else 'no_routing only'}")
    print(f"  SNOWM: {'all (2)' if not args.skip_snow else 'no_snowmod only'}")
    print(f"\nTotal valid combinations: {len(combinations)}")
    
    # Apply filter if specified
    if args.config_filter:
        combinations = [c for c in combinations 
                       if args.config_filter.lower() in str(c).lower()]
    
    # Limit if specified
    if args.max_configs:
        combinations = combinations[:args.max_configs]
    
    print(f"\nTesting {len(combinations)} configurations...")
    
    # Run validation
    results = []
    for i, decisions in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}]", end="")
        
        result = validate_configuration(
            decisions=decisions,
            cfuse_binary=args.cfuse_binary,
            fortran_binary=args.fortran_binary,
            file_manager=args.file_manager,
            basin_id=args.basin_id,
            settings_dir=settings_dir,
            output_dir=output_dir,
        )
        results.append(result)
    
    # Generate reports
    report_path = args.output_dir / "validation_report.txt"
    generate_summary_report(results, report_path)
    
    # Generate plots
    plot_path = args.output_dir / "validation_summary.png"
    plot_validation_summary(results, plot_path)
    
    # Save raw results as JSON
    json_path = args.output_dir / "validation_results.json"
    json_results = []
    for r in results:
        json_results.append({
            'config_name': r.config_name,
            'decisions': r.decisions,
            'dfuse_time': r.dfuse_time,
            'fortran_time': r.fortran_time,
            'rmse': float(r.rmse) if not np.isnan(r.rmse) else None,
            'bias': float(r.bias) if not np.isnan(r.bias) else None,
            'correlation': float(r.correlation) if not np.isnan(r.correlation) else None,
            'max_diff': float(r.max_diff) if not np.isnan(r.max_diff) else None,
            'dfuse_success': r.dfuse_success,
            'fortran_success': r.fortran_success,
            'error_message': r.error_message,
        })
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Print summary
    successful = [r for r in results if r.dfuse_success and r.fortran_success]
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    
    if successful:
        rmses = [r.rmse for r in successful if not np.isnan(r.rmse)]
        biases = [r.bias for r in successful if not np.isnan(r.bias)]
        
        print(f"\nRMSE:  mean={np.mean(rmses):.4f}, median={np.median(rmses):.4f}")
        print(f"Bias:  mean={np.mean(biases):.4f}, median={np.median(biases):.4f}")
        
        dfuse_total = sum(r.dfuse_time for r in successful)
        fortran_total = sum(r.fortran_time for r in successful)
        print(f"\nTotal cFUSE time:   {dfuse_total:.2f}s")
        print(f"Total Fortran time: {fortran_total:.2f}s")
        print(f"Speedup: {fortran_total/dfuse_total:.1f}x")


if __name__ == '__main__':
    main()
