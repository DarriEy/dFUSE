#!/usr/bin/env python
"""
dFUSE Command Line Interface

Drop-in replacement for Fortran FUSE executable.

Usage:
    python dfuse_cli.py <fileManager> <basinID> <runMode>
    
    Or if installed:
    dfuse <fileManager> <basinID> <runMode>

Arguments:
    fileManager: Path to FUSE file manager (fm_*.txt)
    basinID: Basin identifier (e.g., 'Klondike_Bonanza_Creek')
    runMode: Run mode ('run_def' for default parameters, 'run_pre' for preset)

Example:
    python dfuse_cli.py /path/to/fm_catch.txt Klondike_Bonanza_Creek run_def
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dfuse_netcdf import (
    read_fuse_forcing,
    read_elevation_bands,
    parse_fuse_decisions,
    parse_file_manager,
    parse_fortran_constraints,
    write_fuse_output,
    FortranParameters
)


def run_dfuse(
    file_manager_path: str,
    basin_id: str,
    run_mode: str = 'run_def',
    verbose: bool = True
) -> dict:
    """
    Run dFUSE model from Fortran FUSE file manager.
    
    Args:
        file_manager_path: Path to file manager
        basin_id: Basin identifier
        run_mode: 'run_def' (default params), 'run_pre' (preset)
        verbose: Print progress
        
    Returns:
        Dictionary with results
    """
    from dfuse import (
        FUSEConfig,
        UpperLayerArch, LowerLayerArch, BaseflowType, PercolationType,
        SurfaceRunoffType, EvaporationType, InterflowType
    )
    import dfuse_core
    
    file_manager_path = Path(file_manager_path)
    
    if verbose:
        print(f"dFUSE v{dfuse_core.__version__}")
        print("=" * 60)
    
    # Parse file manager
    fm = parse_file_manager(file_manager_path)
    
    input_path = Path(fm['input_path'])
    output_path = Path(fm['output_path'])
    setngs_path = Path(fm['setngs_path'])
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Basin: {basin_id}")
        print(f"Simulation: {fm['date_start_sim']} to {fm['date_end_sim']}")
        print(f"Output: {output_path}")
    
    # Parse model decisions
    decisions_path = setngs_path / fm['m_decisions']
    decisions = parse_fuse_decisions(decisions_path)
    config_dict = decisions.to_config_dict()
    
    if verbose:
        print(f"\nModel structure:")
        print(f"  Upper layer: {decisions.arch1}")
        print(f"  Lower layer: {decisions.arch2}")
        print(f"  Surface runoff: {decisions.qsurf}")
        print(f"  Baseflow: {decisions.arch2}")
        print(f"  Snow: {decisions.snowmod}")
    
    # Create FUSEConfig
    config = FUSEConfig(
        upper_arch=UpperLayerArch(config_dict['upper_arch']),
        lower_arch=LowerLayerArch(config_dict['lower_arch']),
        baseflow=BaseflowType(config_dict['baseflow']),
        percolation=PercolationType(config_dict['percolation']),
        surface_runoff=SurfaceRunoffType(config_dict['surface_runoff']),
        evaporation=EvaporationType(config_dict['evaporation']),
        interflow=InterflowType(config_dict['interflow']),
        enable_snow=config_dict['enable_snow']
    )
    
    # Parse parameters
    constraints_path = setngs_path / fm['constraints']
    fortran_params = parse_fortran_constraints(constraints_path)
    params_np = fortran_params.to_dfuse_params()
    
    if verbose:
        print(f"\nParameters:")
        print(f"  S1_max: {params_np[0]:.1f} mm")
        print(f"  S2_max: {params_np[1]:.1f} mm")
        print(f"  ks: {params_np[12]:.1f} mm/day")
        print(f"  n: {params_np[13]:.1f}")
        print(f"  T_rain: {params_np[22]:.1f} °C")
        print(f"  melt_rate: {params_np[23]:.2f} mm/°C/day")
    
    # Load forcing
    forcing_file = input_path / f"{basin_id}{fm['suffix_forcing']}"
    forcing = read_fuse_forcing(forcing_file)
    
    if verbose:
        print(f"\nForcing loaded: {forcing.n_timesteps} timesteps")
        print(f"  Precip: {np.nanmean(forcing.precip):.2f} mm/day (mean)")
        print(f"  Temp: {np.nanmean(forcing.temp):.1f} °C (mean)")
    
    # Filter to simulation period
    # TODO: Implement date filtering based on date_start_sim, date_end_sim
    
    # Initialize state
    n_states = config.num_states
    state_np = np.zeros(n_states, dtype=np.float32)
    state_np[0] = fortran_params.MAXWATR_1 * 0.5  # S1 at 50%
    s2_idx = config.num_upper_states
    state_np[s2_idx] = fortran_params.MAXWATR_2 * 0.5  # S2 at 50%
    
    # Prepare forcing
    forcing_np = np.stack([forcing.precip, forcing.pet, forcing.temp], axis=1).astype(np.float32)
    
    # Replace NaN/fill values with 0
    forcing_np = np.nan_to_num(forcing_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    if verbose:
        print(f"\nRunning dFUSE...")
    
    # Run model
    start_time = datetime.now()
    final_state, runoff = dfuse_core.run_fuse(
        state_np, forcing_np, params_np, config.to_dict(), 1.0
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    
    if verbose:
        print(f"  Completed in {elapsed:.2f} seconds")
        print(f"  Runoff: mean={np.mean(runoff):.3f}, max={np.max(runoff):.3f} mm/day")
    
    # TODO: Apply gamma routing if q_tdh == 'rout_gamma'
    runoff_routed = runoff  # Placeholder - routing not yet implemented
    
    # Save output
    output_file = output_path / f"{basin_id}_{fm['fmodel_id']}_dfuse.nc"
    
    write_fuse_output(
        output_file,
        time=forcing.time,
        runoff=runoff_routed,
        states={
            'watr_1': np.full(len(runoff), final_state[0]),  # Would need trajectory
            'watr_2': np.full(len(runoff), final_state[s2_idx]),
        },
        time_units=forcing.time_units or "days since 1970-01-01"
    )
    
    if verbose:
        print(f"\nOutput saved: {output_file}")
    
    # Compute metrics if observed data available
    results = {
        'basin_id': basin_id,
        'n_timesteps': len(runoff),
        'runoff_mean': float(np.mean(runoff)),
        'runoff_max': float(np.max(runoff)),
        'output_file': str(output_file),
        'elapsed_seconds': elapsed
    }
    
    if forcing.q_obs is not None:
        valid = ~np.isnan(forcing.q_obs) & (forcing.q_obs > -9000) & ~np.isnan(runoff)
        if valid.sum() > 0:
            q_obs = forcing.q_obs[valid]
            q_sim = runoff[valid]
            
            # NSE
            ss_res = np.sum((q_sim - q_obs)**2)
            ss_tot = np.sum((q_obs - np.mean(q_obs))**2)
            nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            
            # KGE
            r = np.corrcoef(q_sim, q_obs)[0, 1]
            alpha = np.std(q_sim) / np.std(q_obs) if np.std(q_obs) > 0 else np.nan
            beta = np.mean(q_sim) / np.mean(q_obs) if np.mean(q_obs) > 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
            results['nse'] = float(nse)
            results['kge'] = float(kge)
            results['bias'] = float(np.mean(q_sim - q_obs))
            
            if verbose:
                print(f"\nPerformance vs observed:")
                print(f"  NSE: {nse:.3f}")
                print(f"  KGE: {kge:.3f}")
                print(f"  Bias: {results['bias']:.3f} mm/day")
    
    if verbose:
        print("\n" + "=" * 60)
        print("dFUSE completed successfully")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='dFUSE - Differentiable FUSE hydrological model',
        epilog='Drop-in replacement for Fortran FUSE executable.'
    )
    parser.add_argument('fileManager', help='Path to FUSE file manager (fm_*.txt)')
    parser.add_argument('basinID', help='Basin identifier')
    parser.add_argument('runMode', choices=['run_def', 'run_pre', 'calib_sce'],
                        help='Run mode (run_def=default params, run_pre=preset)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress output')
    
    args = parser.parse_args()
    
    if args.runMode == 'calib_sce':
        print("Error: Calibration mode not yet implemented in dFUSE")
        print("Use PyTorch optimization instead - see examples/parameter_optimization.py")
        sys.exit(1)
    
    try:
        results = run_dfuse(
            args.fileManager,
            args.basinID,
            args.runMode,
            verbose=not args.quiet
        )
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
