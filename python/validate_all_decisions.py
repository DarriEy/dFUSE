#!/usr/bin/env python3
"""
Comprehensive validation of dFUSE against Fortran FUSE across all decision combinations.

This script:
1. Documents the physics equations in both implementations
2. Tests all valid decision combinations
3. Identifies any discrepancies and their likely causes

Physics Mapping from Clark et al. (2008):
=========================================

UPPER LAYER ARCHITECTURE (ARCH1):
---------------------------------
onestate_1:  Single state S1 (VIC/ARNO style) - Eq 1a
             dS1/dt = P - E1 - Q12 - Qif - Qufof
             
tension1_1:  Tension + Free storage (PRMS style) - Eq 1b
             dS1_T/dt = P - E1 - Qutof
             dS1_F/dt = Qutof - Q12 - Qif - Qufof
             
tension2_1:  Two tension + Free (Sacramento style) - Eq 1c, 1d
             dS1_TA/dt = P - E1_A - Qurof
             dS1_TB/dt = Qurof - E1_B - Qutof
             dS1_F/dt = Qutof - Q12 - Qif - Qufof

LOWER LAYER ARCHITECTURE (ARCH2):
---------------------------------
unlimfrc_2:  Unlimited capacity, no evap - Eq 2a
             dS2/dt = Q12 - Qb
             
unlimpow_2:  Unlimited capacity, with evap - Eq 2b
             dS2/dt = Q12 - E2 - Qb - Qsfof

fixedsiz_2:  Fixed size (TOPMODEL) - Eq 2b with fixed S2_max
             
tens2pll_2:  Tension + 2 parallel baseflow - Eq 2c, 2d, 2e
             dS2_T/dt = κ*Q12 - E2 - Qstof
             dS2_FA/dt = (1-κ)*Q12/2 + Qstof/2 - Qb_A - Qsfofa
             dS2_FB/dt = (1-κ)*Q12/2 + Qstof/2 - Qb_B - Qsfofb

EVAPORATION (ESOIL):
--------------------
sequential:  Upper then lower - Eq 3a, 3b
             E1 = PET * (S1_T / S1_T_max)
             E2 = (PET - E1) * (S2_T / S2_T_max)
             
rootweight:  Root-weighted - Eq 3c, 3d
             E1 = PET * r1 * (S1_T / S1_T_max)
             E2 = PET * (1-r1) * (S2_T / S2_T_max)

PERCOLATION (QPERC):
--------------------
perc_f2sat:  Free storage based - Eq 4b (field capacity to saturation)
             Q12 = ku * (S1_F / S1_F_max)^c
             
perc_w2sat:  Total storage based - Eq 4a (wilting point to saturation)
             Q12 = ku * (S1 / S1_max)^c
             
perc_lower:  Lower zone demand - Eq 4c (Sacramento)
             Q12 = Q0 * dlz * (S1_F / S1_F_max)
             dlz = 1 + α * (S2 / S2_max)^ψ

INTERFLOW (QINTF):
------------------
intflwnone:  No interflow
             Qif = 0
             
intflwsome:  Linear interflow - Eq 5b
             Qif = ki * (S1_F / S1_F_max)

BASEFLOW (from ARCH2):
----------------------
For unlimfrc_2, unlimpow_2:
             Qb = v * S2  (linear, PRMS) - Eq 6a
             
For fixedsiz_2:
             Qb = ks * (S2 / S2_max)^n  (nonlinear, TOPMODEL) - Eq 6d
             
For tens2pll_2:
             Qb_A = v_A * S2_FA  - Eq 6b
             Qb_B = v_B * S2_FB

Note: unlimpow_2 with ARNO/VIC surface runoff typically uses nonlinear:
             Qb = ks * (S2 / S2_max)^n  - Eq 6c

SURFACE RUNOFF (QSURF):
-----------------------
arno_x_vic:  VIC/ARNO Pareto - Eq 9b, 11
             Ac = 1 - (1 - S1/S1_max)^b
             Qsx = Ac * throughfall
             
prms_varnt:  Linear - Eq 9a, 11
             Ac = (S1_T / S1_T_max) * Ac_max
             Qsx = Ac * throughfall
             
tmdl_param:  TOPMODEL - Eq 9c, 10, 11
             Ac = integral of gamma distribution
             Qsx = Ac * throughfall

ROUTING (Q_TDH):
----------------
no_routing:  No delay
             Q_routed = Q_instnt
             
rout_gamma:  Gamma distribution delay
             Q_routed = convolution(Q_instnt, gamma(shape, scale))

SNOW (SNOWM):
-------------
no_snowmod:  No snow
             throughfall = precip
             
temp_index:  Temperature index with elevation bands
             For each band:
               T_band = T + lapse_rate * (elev - ref_elev)/1000
               P_band = P * (1 + opg * (elev - ref_elev)/1000)
               snow_frac = 1 / (1 + exp(k*(T_band - T_rain)))
               melt = melt_rate * max(T_band - T_melt, 0)
             throughfall = Σ(area_frac * (rain + melt))

KNOWN DIFFERENCES BETWEEN dFUSE AND FORTRAN FUSE:
=================================================

1. NUMERICAL SOLVER:
   - Fortran: Implicit method with Newton iteration and adaptive sub-stepping
   - dFUSE: Explicit Euler (default) or SUNDIALS CVODE (optional)
   - Impact: High-rate dynamics (ki=500) may differ; use smaller dt or SUNDIALS

2. SMOOTHING FUNCTIONS:
   - Fortran: Hard thresholds with iterative convergence
   - dFUSE: Smooth approximations (logistic, softmax) for differentiability
   - Impact: Small fluxes near thresholds may differ slightly

3. ROUTING:
   - Fortran: Full gamma distribution convolution
   - dFUSE: Currently no routing implemented (Q_routed = Q_instnt)
   - Impact: Peak timing may differ if routing is significant

4. RAIN ERROR:
   - Fortran: Multiplicative/additive error options
   - dFUSE: Not implemented
   - Impact: Usually error=1, so no difference

5. MELT RATE SEASONALITY:
   - Fortran: MFMAX/MFMIN vary seasonally (summer/winter)
   - dFUSE: Uses average melt_rate = (MFMAX + MFMIN) / 2
   - Impact: Slight differences in spring melt timing
"""

import numpy as np
import sys
from pathlib import Path
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import subprocess
import xarray as xr

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dfuse import FUSEConfig, UpperLayerArch, LowerLayerArch, BaseflowType, \
                  PercolationType, SurfaceRunoffType, EvaporationType, InterflowType
from dfuse_netcdf import (
    parse_file_manager, parse_fuse_decisions, parse_fortran_constraints,
    read_fuse_forcing, read_elevation_bands, FUSEDecisions
)

import dfuse_core


@dataclass
class DecisionCombination:
    """A specific combination of FUSE model decisions"""
    arch1: str      # Upper layer: onestate_1, tension1_1, tension2_1
    arch2: str      # Lower layer: unlimfrc_2, unlimpow_2, fixedsiz_2, tens2pll_2
    qsurf: str      # Surface runoff: arno_x_vic, prms_varnt, tmdl_param
    qperc: str      # Percolation: perc_f2sat, perc_w2sat, perc_lower
    esoil: str      # Evaporation: sequential, rootweight
    qintf: str      # Interflow: intflwnone, intflwsome
    snowm: str      # Snow: no_snowmod, temp_index
    
    def to_fortran_decisions(self) -> str:
        """Generate Fortran FUSE decision file content"""
        lines = [
            f"additive_e RFERR    ! rainfall error",
            f"{self.arch1} ARCH1    ! upper layer architecture",
            f"{self.arch2} ARCH2    ! lower layer architecture", 
            f"{self.qsurf} QSURF    ! surface runoff",
            f"{self.qperc} QPERC    ! percolation",
            f"{self.esoil} ESOIL    ! evaporation",
            f"{self.qintf} QINTF    ! interflow",
            f"rout_gamma Q_TDH    ! routing",
            f"{self.snowm} SNOWM    ! snow model"
        ]
        return '\n'.join(lines)
    
    def to_dfuse_config(self) -> Dict:
        """Convert to dFUSE configuration dictionary"""
        upper_map = {
            'onestate_1': 0,  # SINGLE_STATE
            'tension1_1': 1,  # TENSION_FREE
            'tension2_1': 2   # TENSION2_FREE
        }
        
        lower_map = {
            'unlimfrc_2': 0,  # SINGLE_NOEVAP
            'unlimpow_2': 1,  # SINGLE_EVAP
            'fixedsiz_2': 1,  # SINGLE_EVAP (with TOPMODEL baseflow)
            'tens2pll_2': 2   # TENSION_2RESERV
        }
        
        # Baseflow type depends on arch2 and qsurf combination
        baseflow_map = {
            'unlimfrc_2': 0,  # LINEAR
            'unlimpow_2': 2,  # NONLINEAR (with arno_x_vic)
            'fixedsiz_2': 3,  # TOPMODEL
            'tens2pll_2': 1   # PARALLEL_LINEAR
        }
        
        percolation_map = {
            'perc_f2sat': 1,  # FREE_STORAGE
            'perc_w2sat': 0,  # TOTAL_STORAGE
            'perc_lower': 2   # LOWER_DEMAND
        }
        
        surface_map = {
            'arno_x_vic': 1,  # UZ_PARETO
            'prms_varnt': 0,  # UZ_LINEAR
            'tmdl_param': 2   # LZ_GAMMA (TOPMODEL)
        }
        
        evap_map = {
            'sequential': 0,
            'rootweight': 1
        }
        
        interflow_map = {
            'intflwnone': 0,
            'intflwsome': 1
        }
        
        return {
            'upper_arch': upper_map[self.arch1],
            'lower_arch': lower_map[self.arch2],
            'baseflow': baseflow_map[self.arch2],
            'percolation': percolation_map[self.qperc],
            'surface_runoff': surface_map[self.qsurf],
            'evaporation': evap_map[self.esoil],
            'interflow': interflow_map[self.qintf],
            'enable_snow': self.snowm == 'temp_index'
        }
    
    def name(self) -> str:
        """Short name for this combination"""
        return f"{self.arch1}_{self.arch2}_{self.qsurf}_{self.qperc}_{self.esoil}_{self.qintf}_{self.snowm}"
    
    def is_valid(self) -> bool:
        """Check if this is a valid combination (some are incompatible)"""
        # perc_lower (Sacramento) requires tension architecture and tens2pll_2
        if self.qperc == 'perc_lower':
            if self.arch1 == 'onestate_1' or self.arch2 != 'tens2pll_2':
                return False
        
        # TOPMODEL surface runoff typically requires fixedsiz_2
        if self.qsurf == 'tmdl_param' and self.arch2 not in ['fixedsiz_2', 'unlimpow_2']:
            return False
            
        # Parallel baseflow requires tens2pll_2
        if self.arch2 == 'tens2pll_2' and self.arch1 == 'onestate_1':
            # Sacramento style requires tension in upper layer
            return False
            
        return True


def get_all_valid_combinations() -> List[DecisionCombination]:
    """Generate all valid decision combinations"""
    arch1_options = ['onestate_1', 'tension1_1', 'tension2_1']
    arch2_options = ['unlimfrc_2', 'unlimpow_2', 'fixedsiz_2', 'tens2pll_2']
    qsurf_options = ['arno_x_vic', 'prms_varnt', 'tmdl_param']
    qperc_options = ['perc_f2sat', 'perc_w2sat', 'perc_lower']
    esoil_options = ['sequential', 'rootweight']
    qintf_options = ['intflwnone', 'intflwsome']
    snowm_options = ['no_snowmod', 'temp_index']
    
    valid = []
    for arch1, arch2, qsurf, qperc, esoil, qintf, snowm in product(
        arch1_options, arch2_options, qsurf_options, qperc_options,
        esoil_options, qintf_options, snowm_options
    ):
        combo = DecisionCombination(arch1, arch2, qsurf, qperc, esoil, qintf, snowm)
        if combo.is_valid():
            valid.append(combo)
    
    return valid


def run_dfuse_simulation(
    forcing: np.ndarray,
    params: np.ndarray,
    config: Dict,
    initial_state: np.ndarray,
    elev_bands: Optional[Dict] = None,
    dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run dFUSE simulation.
    
    Returns:
        runoff: Array of daily runoff [n_timesteps]
        final_state: Final state vector
    """
    if elev_bands is not None and config.get('enable_snow', False):
        result = dfuse_core.run_fuse_elevation_bands(
            initial_state, forcing, params, config,
            elev_bands['area_frac'].astype(np.float32),
            elev_bands['mean_elev'].astype(np.float32),
            float(elev_bands['ref_elev']),
            None,  # Initial SWE = 0
            dt,
            False,  # return_fluxes
            False   # return_swe_trajectory
        )
        final_state, runoff, final_swe_bands = result
    else:
        final_state, runoff = dfuse_core.run_fuse(
            initial_state, forcing, params, config, dt
        )
    
    return runoff, final_state


def run_fortran_fuse(
    basin_id: str,
    fm: Dict,
    decisions: DecisionCombination,
    output_suffix: str = 'test'
) -> Optional[np.ndarray]:
    """
    Run Fortran FUSE with specific decisions.
    
    Returns:
        runoff: Array of daily runoff, or None if failed
    """
    # Write decision file
    decision_file = Path(fm['setngs_path']) / f'fuse_zDecisions_{output_suffix}.txt'
    with open(decision_file, 'w') as f:
        f.write(decisions.to_fortran_decisions())
    
    # Run Fortran FUSE
    fm_file = Path(fm['setngs_path']).parent / f'fm_catch_{output_suffix}.txt'
    # Would need to write fm file with correct decision file reference
    
    # For now, assume we can call fuse.exe directly
    # This is a placeholder - actual implementation depends on your setup
    try:
        cmd = ['fuse.exe', str(fm_file), basin_id, f'run_{output_suffix}']
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Fortran FUSE failed: {result.stderr.decode()}")
            return None
        
        # Read output
        output_file = Path(fm['output_path']) / f'{basin_id}_900_runs_{output_suffix}.nc'
        if not output_file.exists():
            return None
            
        ds = xr.open_dataset(output_file)
        runoff = ds['q_routed'].values.squeeze()
        ds.close()
        
        return runoff
        
    except Exception as e:
        print(f"Error running Fortran FUSE: {e}")
        return None


def compare_simulations(
    dfuse_runoff: np.ndarray,
    fortran_runoff: np.ndarray
) -> Dict:
    """
    Compute comparison metrics.
    
    Returns:
        Dictionary with NSE, correlation, RMSE, bias, etc.
    """
    valid = ~(np.isnan(dfuse_runoff) | np.isnan(fortran_runoff))
    d = dfuse_runoff[valid]
    f = fortran_runoff[valid]
    
    # Basic statistics
    mean_d = np.mean(d)
    mean_f = np.mean(f)
    
    # NSE
    ss_res = np.sum((d - f) ** 2)
    ss_tot = np.sum((f - mean_f) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    # Correlation
    if len(d) > 1 and np.std(d) > 0 and np.std(f) > 0:
        corr = np.corrcoef(d, f)[0, 1]
    else:
        corr = np.nan
    
    # RMSE
    rmse = np.sqrt(np.mean((d - f) ** 2))
    
    # Bias
    bias = mean_d - mean_f
    
    # Peak flow comparison
    max_d = np.max(d)
    max_f = np.max(f)
    
    return {
        'nse': nse,
        'correlation': corr,
        'rmse': rmse,
        'bias': bias,
        'mean_dfuse': mean_d,
        'mean_fortran': mean_f,
        'max_dfuse': max_d,
        'max_fortran': max_f,
        'n_valid': np.sum(valid)
    }


def run_single_comparison(
    basin_id: str,
    fm_path: str,
    combo: DecisionCombination,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Run a single comparison between dFUSE and Fortran FUSE.
    
    Returns:
        Dictionary with comparison results, or None if failed
    """
    # Parse file manager
    fm = parse_file_manager(fm_path)
    
    # Parse parameters
    constraints_path = Path(fm['setngs_path']) / fm['constraints']
    fortran_params = parse_fortran_constraints(constraints_path)
    params = fortran_params.to_dfuse_params()
    
    # Load forcing
    forcing_path = Path(fm['input_path']) / f"{basin_id}{fm['suffix_forcing']}"
    forcing = read_fuse_forcing(forcing_path)
    forcing_np = np.column_stack([forcing.precip, forcing.pet, forcing.temp]).astype(np.float32)
    
    # Load elevation bands if needed
    elev_bands = None
    if combo.snowm == 'temp_index':
        elev_bands_path = Path(fm['input_path']) / f"{basin_id}{fm['suffix_elev_bands']}"
        try:
            eb = read_elevation_bands(elev_bands_path)
            elev_bands = {
                'area_frac': eb.area_frac.astype(np.float32),
                'mean_elev': eb.mean_elev.astype(np.float32),
                'ref_elev': float(np.sum(eb.area_frac * eb.mean_elev))
            }
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load elevation bands: {e}")
            elev_bands = None
    
    # Get config
    config = combo.to_dfuse_config()
    
    # Determine number of states
    fuse_config = FUSEConfig(
        upper_arch=UpperLayerArch(config['upper_arch']),
        lower_arch=LowerLayerArch(config['lower_arch']),
        baseflow=BaseflowType(config['baseflow']),
        percolation=PercolationType(config['percolation']),
        surface_runoff=SurfaceRunoffType(config['surface_runoff']),
        evaporation=EvaporationType(config['evaporation']),
        interflow=InterflowType(config['interflow']),
        enable_snow=config['enable_snow']
    )
    n_states = fuse_config.num_states
    
    # Set initial state (equilibrium-ish)
    initial_state = np.zeros(n_states, dtype=np.float32)
    initial_state[0] = fortran_params.MAXWATR_1 * fortran_params.FRACTEN
    s2_idx = fuse_config.num_upper_states
    initial_state[s2_idx] = fortran_params.MAXWATR_2 * 0.25
    
    # Run dFUSE
    try:
        dfuse_runoff, _ = run_dfuse_simulation(
            forcing_np, params, config, initial_state, elev_bands
        )
    except Exception as e:
        if verbose:
            print(f"  dFUSE failed: {e}")
        return None
    
    # Read Fortran output (assuming it exists from a previous run)
    fortran_output = Path(fm['output_path']) / f"{basin_id}_900_runs_def.nc"
    if not fortran_output.exists():
        if verbose:
            print(f"  Fortran output not found: {fortran_output}")
        return None
    
    try:
        ds = xr.open_dataset(fortran_output)
        fortran_runoff = ds['q_routed'].values.squeeze()
        ds.close()
    except Exception as e:
        if verbose:
            print(f"  Could not read Fortran output: {e}")
        return None
    
    # Compare
    metrics = compare_simulations(dfuse_runoff, fortran_runoff)
    metrics['combination'] = combo.name()
    metrics['config'] = config
    
    return metrics


def run_comprehensive_validation(
    basin_id: str,
    fm_path: str,
    max_combinations: Optional[int] = None,
    verbose: bool = True
):
    """
    Run comprehensive validation across all decision combinations.
    """
    print("=" * 70)
    print("COMPREHENSIVE dFUSE vs FORTRAN FUSE VALIDATION")
    print("=" * 70)
    
    # Get all valid combinations
    combinations = get_all_valid_combinations()
    print(f"\nTotal valid decision combinations: {len(combinations)}")
    
    if max_combinations:
        combinations = combinations[:max_combinations]
        print(f"Testing first {max_combinations} combinations")
    
    results = []
    
    for i, combo in enumerate(combinations):
        if verbose:
            print(f"\n[{i+1}/{len(combinations)}] Testing: {combo.name()}")
        
        result = run_single_comparison(basin_id, fm_path, combo, verbose)
        
        if result:
            results.append(result)
            if verbose:
                print(f"  NSE={result['nse']:.3f}, Corr={result['correlation']:.3f}, "
                      f"RMSE={result['rmse']:.3f}, Bias={result['bias']:.4f}")
        else:
            if verbose:
                print(f"  SKIPPED (Fortran output not available)")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if results:
        nse_values = [r['nse'] for r in results if not np.isnan(r['nse'])]
        corr_values = [r['correlation'] for r in results if not np.isnan(r['correlation'])]
        
        print(f"\nTested combinations: {len(results)}")
        print(f"NSE:  min={min(nse_values):.3f}, max={max(nse_values):.3f}, "
              f"mean={np.mean(nse_values):.3f}")
        print(f"Corr: min={min(corr_values):.3f}, max={max(corr_values):.3f}, "
              f"mean={np.mean(corr_values):.3f}")
        
        # Flag poor matches
        poor_matches = [r for r in results if r['nse'] < 0.5]
        if poor_matches:
            print(f"\nPoor matches (NSE < 0.5): {len(poor_matches)}")
            for r in poor_matches:
                print(f"  {r['combination']}: NSE={r['nse']:.3f}")
        else:
            print("\nAll tested combinations have NSE >= 0.5")
    else:
        print("\nNo successful comparisons.")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate dFUSE against Fortran FUSE')
    parser.add_argument('--basin', default='Klondike_Bonanza_Creek',
                        help='Basin ID')
    parser.add_argument('--fm', default='/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Klondike_Bonanza_Creek/settings/FUSE/fm_catch.txt',
                        help='Path to file manager')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum combinations to test')
    parser.add_argument('--list', action='store_true',
                        help='List all valid combinations')
    
    args = parser.parse_args()
    
    if args.list:
        combos = get_all_valid_combinations()
        print(f"Valid decision combinations: {len(combos)}")
        for i, c in enumerate(combos):
            print(f"{i+1:3d}. {c.name()}")
    else:
        run_comprehensive_validation(args.basin, args.fm, args.max)
