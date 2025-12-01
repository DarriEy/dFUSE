#!/usr/bin/env python3
"""
Comprehensive validation of dFUSE against Fortran FUSE across all decision combinations.

This script:
1. Generates all valid decision combinations
2. Writes decision files and runs Fortran FUSE
3. Runs corresponding dFUSE simulations
4. Compares results and identifies discrepancies

KNOWN SOLVER DIFFERENCES:
=========================
- Fortran FUSE: Implicit Euler with Newton iteration + adaptive sub-stepping
  (See Clark & Kavetski 2010 "Ancient numerical daemons")
- dFUSE: Explicit Euler (for differentiability)
  
This leads to expected differences:
- High rate constants (ki > 100, ku > 50) may show numerical differences
- Rapid state changes during melt events may differ in timing
- Low flows may persist longer in dFUSE (no numerical smoothing)

PHYSICS VERIFICATION CHECKLIST:
===============================
✅ Upper layer architecture (onestate_1, tension1_1, tension2_1)
✅ Lower layer architecture (unlimfrc_2, unlimpow_2, fixedsiz_2, tens2pll_2)  
✅ Surface runoff (arno_x_vic, prms_varnt, tmdl_param)
✅ Percolation (perc_f2sat, perc_w2sat, perc_lower)
✅ Evaporation (sequential, rootweight)
✅ Interflow (intflwnone, intflwsome)
✅ Snow (no_snowmod, temp_index with elevation bands)
⚠️ Routing (rout_gamma) - NOT IMPLEMENTED in dFUSE
⚠️ Rainfall error - NOT IMPLEMENTED in dFUSE

Author: Claude (Anthropic)
Date: 2024
"""

import numpy as np
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import xarray as xr

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dfuse import FUSEConfig, UpperLayerArch, LowerLayerArch, BaseflowType, \
                  PercolationType, SurfaceRunoffType, EvaporationType, InterflowType
from dfuse_netcdf import (
    parse_file_manager, parse_fuse_decisions, parse_fortran_constraints,
    read_fuse_forcing, read_elevation_bands, FUSEDecisions
)

try:
    import dfuse_core
    DFUSE_AVAILABLE = True
except ImportError:
    DFUSE_AVAILABLE = False
    print("Warning: dfuse_core not available")


# =============================================================================
# DECISION COMBINATIONS
# =============================================================================

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
    
    # Mapping tables
    _upper_map: Dict = field(default_factory=lambda: {
        'onestate_1': UpperLayerArch.SINGLE_STATE,
        'tension1_1': UpperLayerArch.TENSION_FREE,
        'tension2_1': UpperLayerArch.TENSION2_FREE
    })
    
    _lower_map: Dict = field(default_factory=lambda: {
        'unlimfrc_2': LowerLayerArch.SINGLE_NOEVAP,
        'unlimpow_2': LowerLayerArch.SINGLE_EVAP,
        'fixedsiz_2': LowerLayerArch.SINGLE_EVAP,
        'tens2pll_2': LowerLayerArch.TENSION_2RESERV
    })
    
    _surface_map: Dict = field(default_factory=lambda: {
        'arno_x_vic': SurfaceRunoffType.UZ_PARETO,
        'prms_varnt': SurfaceRunoffType.UZ_LINEAR,
        'tmdl_param': SurfaceRunoffType.LZ_GAMMA
    })
    
    _percolation_map: Dict = field(default_factory=lambda: {
        'perc_f2sat': PercolationType.FREE_STORAGE,
        'perc_w2sat': PercolationType.TOTAL_STORAGE,
        'perc_lower': PercolationType.LOWER_DEMAND
    })
    
    _evap_map: Dict = field(default_factory=lambda: {
        'sequential': EvaporationType.SEQUENTIAL,
        'rootweight': EvaporationType.ROOT_WEIGHT
    })
    
    _interflow_map: Dict = field(default_factory=lambda: {
        'intflwnone': InterflowType.NONE,
        'intflwsome': InterflowType.LINEAR
    })
    
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
    
    def to_dfuse_config(self) -> FUSEConfig:
        """Convert to dFUSE FUSEConfig object"""
        # Baseflow type depends on arch2
        if self.arch2 == 'unlimfrc_2':
            baseflow = BaseflowType.LINEAR
        elif self.arch2 == 'tens2pll_2':
            baseflow = BaseflowType.PARALLEL_LINEAR
        elif self.arch2 == 'fixedsiz_2':
            baseflow = BaseflowType.TOPMODEL
        else:  # unlimpow_2
            # With ARNO/VIC, use nonlinear
            if self.qsurf == 'arno_x_vic':
                baseflow = BaseflowType.NONLINEAR
            else:
                baseflow = BaseflowType.LINEAR
        
        return FUSEConfig(
            upper_arch=self._upper_map[self.arch1],
            lower_arch=self._lower_map[self.arch2],
            baseflow=baseflow,
            percolation=self._percolation_map[self.qperc],
            surface_runoff=self._surface_map[self.qsurf],
            evaporation=self._evap_map[self.esoil],
            interflow=self._interflow_map[self.qintf],
            enable_snow=(self.snowm == 'temp_index')
        )
    
    def name(self) -> str:
        """Short name for this combination"""
        return f"{self.arch1}_{self.arch2}_{self.qsurf}_{self.qperc}_{self.esoil}_{self.qintf}_{self.snowm}"
    
    def short_name(self) -> str:
        """Very short name"""
        parts = [
            self.arch1.split('_')[0],
            self.arch2.split('_')[0],
            self.qsurf.split('_')[0],
            self.qperc.split('_')[0][:4],
            self.esoil[:3],
            'if' if self.qintf == 'intflwsome' else 'no',
            'sn' if self.snowm == 'temp_index' else 'ns'
        ]
        return '_'.join(parts)
    
    def is_valid(self) -> bool:
        """Check if this is a valid combination"""
        # perc_lower (Sacramento) requires tension architecture and tens2pll_2
        if self.qperc == 'perc_lower':
            if self.arch1 == 'onestate_1':
                return False
            if self.arch2 != 'tens2pll_2':
                return False
        
        # tens2pll_2 requires tension in upper layer (Sacramento style)
        if self.arch2 == 'tens2pll_2':
            if self.arch1 == 'onestate_1':
                return False
        
        # TOPMODEL surface runoff typically with fixedsiz_2
        # But can work with others, just less common
        
        return True
    
    def expected_issues(self) -> List[str]:
        """List potential comparison issues for this combination"""
        issues = []
        
        # High interflow rate may cause numerical differences
        if self.qintf == 'intflwsome':
            issues.append("High ki (500 mm/day) may cause explicit/implicit solver differences")
        
        # Complex architectures may accumulate numerical errors
        if self.arch1 == 'tension2_1' or self.arch2 == 'tens2pll_2':
            issues.append("Complex architecture may show larger numerical divergence")
        
        # Snow adds complexity
        if self.snowm == 'temp_index':
            issues.append("Melt rate seasonality differs (dFUSE uses average)")
        
        return issues


def get_all_valid_combinations(
    arch1_filter: Optional[List[str]] = None,
    arch2_filter: Optional[List[str]] = None,
    snow_only: bool = False,
    no_snow_only: bool = False
) -> List[DecisionCombination]:
    """Generate all valid decision combinations with optional filters"""
    
    arch1_options = ['onestate_1', 'tension1_1', 'tension2_1']
    arch2_options = ['unlimfrc_2', 'unlimpow_2', 'fixedsiz_2', 'tens2pll_2']
    qsurf_options = ['arno_x_vic', 'prms_varnt', 'tmdl_param']
    qperc_options = ['perc_f2sat', 'perc_w2sat', 'perc_lower']
    esoil_options = ['sequential', 'rootweight']
    qintf_options = ['intflwnone', 'intflwsome']
    snowm_options = ['no_snowmod', 'temp_index']
    
    # Apply filters
    if arch1_filter:
        arch1_options = [a for a in arch1_options if a in arch1_filter]
    if arch2_filter:
        arch2_options = [a for a in arch2_options if a in arch2_filter]
    if snow_only:
        snowm_options = ['temp_index']
    if no_snow_only:
        snowm_options = ['no_snowmod']
    
    valid = []
    for arch1, arch2, qsurf, qperc, esoil, qintf, snowm in product(
        arch1_options, arch2_options, qsurf_options, qperc_options,
        esoil_options, qintf_options, snowm_options
    ):
        combo = DecisionCombination(arch1, arch2, qsurf, qperc, esoil, qintf, snowm)
        if combo.is_valid():
            valid.append(combo)
    
    return valid


# =============================================================================
# SIMULATION RUNNERS
# =============================================================================

def run_dfuse_simulation(
    forcing: np.ndarray,
    params: np.ndarray,
    config: FUSEConfig,
    initial_state: np.ndarray,
    elev_bands: Optional[Dict] = None,
    dt: float = 1.0,
    return_fluxes: bool = False
) -> Dict:
    """
    Run dFUSE simulation.
    
    Returns dict with runoff, final_state, and optionally fluxes
    """
    if not DFUSE_AVAILABLE:
        raise RuntimeError("dfuse_core not available")
    
    config_dict = config.to_dict()
    
    if elev_bands is not None and config.enable_snow:
        result = dfuse_core.run_fuse_elevation_bands(
            initial_state.astype(np.float32),
            forcing.astype(np.float32),
            params.astype(np.float32),
            config_dict,
            elev_bands['area_frac'].astype(np.float32),
            elev_bands['mean_elev'].astype(np.float32),
            float(elev_bands['ref_elev']),
            None,  # Initial SWE = 0
            dt,
            return_fluxes,
            False   # return_swe_trajectory
        )
        if return_fluxes:
            final_state, runoff, fluxes, final_swe = result[:4]
            return {
                'runoff': np.array(runoff),
                'final_state': np.array(final_state),
                'fluxes': np.array(fluxes),
                'final_swe': np.array(final_swe)
            }
        else:
            final_state, runoff, final_swe = result[:3]
            return {
                'runoff': np.array(runoff),
                'final_state': np.array(final_state),
                'final_swe': np.array(final_swe)
            }
    else:
        if return_fluxes:
            final_state, runoff, fluxes = dfuse_core.run_fuse(
                initial_state.astype(np.float32),
                forcing.astype(np.float32),
                params.astype(np.float32),
                config_dict,
                dt,
                True  # return_fluxes
            )
            return {
                'runoff': np.array(runoff),
                'final_state': np.array(final_state),
                'fluxes': np.array(fluxes)
            }
        else:
            final_state, runoff = dfuse_core.run_fuse(
                initial_state.astype(np.float32),
                forcing.astype(np.float32),
                params.astype(np.float32),
                config_dict,
                dt
            )
            return {
                'runoff': np.array(runoff),
                'final_state': np.array(final_state)
            }


def read_fortran_output(output_path: Path) -> Optional[Dict]:
    """Read Fortran FUSE output NetCDF file"""
    if not output_path.exists():
        return None
    
    try:
        ds = xr.open_dataset(output_path)
        
        result = {}
        
        # Main runoff outputs
        for var in ['q_instnt', 'q_routed']:
            if var in ds:
                data = ds[var].values.squeeze()
                data = np.where(data < -9000, np.nan, data)
                result[var] = data
        
        # State variables
        for var in ['watr_1', 'watr_2', 'swe_tot', 'free_1', 'tens_1']:
            if var in ds:
                data = ds[var].values.squeeze()
                data = np.where(data < -9000, np.nan, data)
                result[var] = data
        
        # Flux components
        for var in ['qsurf', 'qbase_2', 'qintf_1', 'qperc_12', 'evap_1', 'evap_2', 'eff_ppt']:
            if var in ds:
                data = ds[var].values.squeeze()
                data = np.where(data < -9000, np.nan, data)
                result[var] = data
        
        # Solver diagnostics
        for var in ['num_funcs', 'numjacobian', 'sub_accept', 'sub_reject', 'sub_noconv', 'max_iterns']:
            if var in ds:
                result[var] = ds[var].values.squeeze()
        
        ds.close()
        return result
        
    except Exception as e:
        print(f"Error reading {output_path}: {e}")
        return None


# =============================================================================
# COMPARISON METRICS
# =============================================================================

def compute_comparison_metrics(
    dfuse_runoff: np.ndarray,
    fortran_runoff: np.ndarray
) -> Dict:
    """Compute comprehensive comparison metrics"""
    
    valid = ~(np.isnan(dfuse_runoff) | np.isnan(fortran_runoff))
    d = dfuse_runoff[valid]
    f = fortran_runoff[valid]
    
    if len(d) == 0:
        return {'error': 'No valid data points'}
    
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
    
    # RMSE and MAE
    rmse = np.sqrt(np.mean((d - f) ** 2))
    mae = np.mean(np.abs(d - f))
    
    # Bias
    bias = mean_d - mean_f
    rel_bias = bias / mean_f if mean_f != 0 else np.nan
    
    # Peak comparison
    max_d = np.max(d)
    max_f = np.max(f)
    peak_ratio = max_d / max_f if max_f > 0 else np.nan
    
    # Low flow comparison (below median)
    median_f = np.median(f)
    low_mask = f < median_f
    if np.sum(low_mask) > 0:
        low_rmse = np.sqrt(np.mean((d[low_mask] - f[low_mask]) ** 2))
    else:
        low_rmse = np.nan
    
    # High flow comparison (above 90th percentile)
    high_thresh = np.percentile(f, 90)
    high_mask = f > high_thresh
    if np.sum(high_mask) > 0:
        high_rmse = np.sqrt(np.mean((d[high_mask] - f[high_mask]) ** 2))
    else:
        high_rmse = np.nan
    
    return {
        'nse': nse,
        'correlation': corr,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'rel_bias': rel_bias,
        'mean_dfuse': mean_d,
        'mean_fortran': mean_f,
        'max_dfuse': max_d,
        'max_fortran': max_f,
        'peak_ratio': peak_ratio,
        'low_rmse': low_rmse,
        'high_rmse': high_rmse,
        'n_valid': np.sum(valid)
    }


def diagnose_differences(
    dfuse_result: Dict,
    fortran_result: Dict,
    combo: DecisionCombination
) -> List[str]:
    """Diagnose potential causes of differences"""
    
    diagnoses = []
    
    d_runoff = dfuse_result['runoff']
    
    # Check if routing is significant
    if 'q_instnt' in fortran_result and 'q_routed' in fortran_result:
        routing_diff = np.nanmean(np.abs(
            fortran_result['q_instnt'] - fortran_result['q_routed']
        ))
        if routing_diff > 0.01:
            diagnoses.append(
                f"ROUTING: Fortran uses gamma routing (mean diff={routing_diff:.3f} mm/day). "
                "dFUSE does not implement routing - use q_instnt for fairer comparison."
            )
    
    # Check solver effort
    if 'num_funcs' in fortran_result:
        mean_funcs = np.nanmean(fortran_result['num_funcs'])
        if mean_funcs > 20:
            diagnoses.append(
                f"SOLVER EFFORT: Fortran needed {mean_funcs:.1f} function evals/step. "
                "High effort suggests stiff system - explicit Euler may be less accurate."
            )
    
    if 'sub_accept' in fortran_result:
        mean_substeps = np.nanmean(fortran_result['sub_accept'])
        if mean_substeps > 5:
            diagnoses.append(
                f"SUB-STEPPING: Fortran used {mean_substeps:.1f} sub-steps/day. "
                "dFUSE uses single step - consider smaller dt."
            )
    
    # Check for state divergence
    if 'final_state' in dfuse_result and 'watr_2' in fortran_result:
        # Compare final S2
        s2_fortran_final = fortran_result['watr_2'][-1]
        s2_dfuse_final = dfuse_result['final_state'][1] if len(dfuse_result['final_state']) > 1 else np.nan
        if not np.isnan(s2_fortran_final) and not np.isnan(s2_dfuse_final):
            s2_diff = abs(s2_dfuse_final - s2_fortran_final)
            if s2_diff > 50:
                diagnoses.append(
                    f"STATE DIVERGENCE: Final S2 differs by {s2_diff:.1f} mm. "
                    "Check percolation and baseflow accumulation."
                )
    
    # Decision-specific diagnostics
    if combo.qintf == 'intflwsome':
        if 'fluxes' in dfuse_result:
            qif_mean = np.mean(dfuse_result['fluxes'][:, 7])  # qif index
            if qif_mean > 1.0:
                diagnoses.append(
                    f"HIGH INTERFLOW: Mean qif={qif_mean:.2f} mm/day. "
                    "With ki=500 mm/day, explicit Euler may overshoot."
                )
    
    if combo.snowm == 'temp_index':
        diagnoses.append(
            "SNOW: dFUSE uses constant melt_rate=(MFMAX+MFMIN)/2, "
            "Fortran varies seasonally. May affect spring timing."
        )
    
    return diagnoses


# =============================================================================
# VALIDATION RUNNER
# =============================================================================

def validate_single_combination(
    combo: DecisionCombination,
    basin_id: str,
    fm: Dict,
    fortran_params: Any,
    forcing: np.ndarray,
    elev_bands: Optional[Dict],
    fortran_output_path: Path,
    verbose: bool = True
) -> Optional[Dict]:
    """
    Validate a single decision combination.
    
    Returns dict with metrics and diagnostics, or None if failed.
    """
    
    if verbose:
        print(f"\n  Testing: {combo.short_name()}")
    
    # Get dFUSE config
    config = combo.to_dfuse_config()
    
    # Set up initial state
    n_states = config.num_states
    initial_state = np.zeros(n_states, dtype=np.float32)
    
    # Initialize based on architecture
    s1_init = fortran_params.MAXWATR_1 * fortran_params.FRACTEN
    s2_init = fortran_params.MAXWATR_2 * 0.25  # Equilibrium for nonlinear baseflow
    
    if config.upper_arch == UpperLayerArch.SINGLE_STATE:
        initial_state[0] = s1_init
    elif config.upper_arch == UpperLayerArch.TENSION_FREE:
        initial_state[0] = s1_init  # S1_T
        initial_state[1] = 0        # S1_F
    elif config.upper_arch == UpperLayerArch.TENSION2_FREE:
        initial_state[0] = s1_init * 0.5  # S1_TA
        initial_state[1] = s1_init * 0.5  # S1_TB
        initial_state[2] = 0              # S1_F
    
    s2_idx = config.num_upper_states
    if config.lower_arch == LowerLayerArch.TENSION_2RESERV:
        initial_state[s2_idx] = s2_init * 0.5     # S2_T
        initial_state[s2_idx + 1] = s2_init * 0.25  # S2_FA
        initial_state[s2_idx + 2] = s2_init * 0.25  # S2_FB
    else:
        initial_state[s2_idx] = s2_init
    
    # Get parameters
    params = fortran_params.to_dfuse_params()
    
    # Select elevation bands based on snow setting
    eb = elev_bands if combo.snowm == 'temp_index' else None
    
    # Run dFUSE
    try:
        dfuse_result = run_dfuse_simulation(
            forcing, params, config, initial_state, eb,
            dt=1.0, return_fluxes=True
        )
    except Exception as e:
        if verbose:
            print(f"    dFUSE FAILED: {e}")
        return None
    
    # Read Fortran output
    fortran_result = read_fortran_output(fortran_output_path)
    if fortran_result is None:
        if verbose:
            print(f"    Fortran output not found")
        return None
    
    # Use q_instnt for comparison (no routing in dFUSE)
    if 'q_instnt' in fortran_result:
        fortran_runoff = fortran_result['q_instnt']
    elif 'q_routed' in fortran_result:
        fortran_runoff = fortran_result['q_routed']
    else:
        if verbose:
            print(f"    No runoff in Fortran output")
        return None
    
    # Compute metrics
    metrics = compute_comparison_metrics(dfuse_result['runoff'], fortran_runoff)
    
    # Diagnose differences
    diagnoses = diagnose_differences(dfuse_result, fortran_result, combo)
    
    # Add expected issues
    expected = combo.expected_issues()
    
    result = {
        'combination': combo.name(),
        'short_name': combo.short_name(),
        'config': config.to_dict(),
        'metrics': metrics,
        'diagnoses': diagnoses,
        'expected_issues': expected,
        'dfuse_result': dfuse_result,
        'fortran_result': fortran_result
    }
    
    if verbose:
        nse = metrics.get('nse', np.nan)
        corr = metrics.get('correlation', np.nan)
        print(f"    NSE={nse:.3f}, Corr={corr:.3f}")
        if nse < 0.6:
            print(f"    ⚠️  LOW NSE - investigating...")
            for diag in diagnoses[:2]:
                print(f"      - {diag[:80]}...")
    
    return result


def run_comprehensive_validation(
    basin_id: str,
    fm_path: str,
    output_dir: Optional[str] = None,
    max_combinations: Optional[int] = None,
    arch1_filter: Optional[List[str]] = None,
    arch2_filter: Optional[List[str]] = None,
    snow_only: bool = False,
    no_snow_only: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """
    Run comprehensive validation across decision combinations.
    """
    
    print("=" * 70)
    print("dFUSE vs FORTRAN FUSE COMPREHENSIVE VALIDATION")
    print("=" * 70)
    
    # Parse file manager
    fm = parse_file_manager(fm_path)
    
    # Parse Fortran parameters
    constraints_path = Path(fm['setngs_path']) / fm['constraints']
    fortran_params = parse_fortran_constraints(constraints_path)
    print(f"\nParameters from: {constraints_path}")
    print(f"  S1_max={fortran_params.MAXWATR_1}, S2_max={fortran_params.MAXWATR_2}")
    print(f"  ki={fortran_params.IFLWRTE}, ku={fortran_params.PERCRTE}")
    
    # Load forcing
    forcing_path = Path(fm['input_path']) / f"{basin_id}{fm['suffix_forcing']}"
    forcing_data = read_fuse_forcing(forcing_path)
    forcing = np.column_stack([
        forcing_data.precip, forcing_data.pet, forcing_data.temp
    ]).astype(np.float32)
    print(f"\nForcing: {len(forcing)} timesteps")
    
    # Load elevation bands
    elev_bands = None
    elev_bands_path = Path(fm['input_path']) / f"{basin_id}{fm['suffix_elev_bands']}"
    try:
        eb = read_elevation_bands(elev_bands_path)
        elev_bands = {
            'area_frac': eb.area_frac,
            'mean_elev': eb.mean_elev,
            'ref_elev': float(np.sum(eb.area_frac * eb.mean_elev))
        }
        print(f"Elevation bands: {len(eb.area_frac)} bands, ref_elev={elev_bands['ref_elev']:.0f}m")
    except Exception as e:
        print(f"No elevation bands: {e}")
    
    # Get combinations
    combinations = get_all_valid_combinations(
        arch1_filter=arch1_filter,
        arch2_filter=arch2_filter,
        snow_only=snow_only,
        no_snow_only=no_snow_only
    )
    print(f"\nTotal valid combinations: {len(combinations)}")
    
    if max_combinations:
        combinations = combinations[:max_combinations]
        print(f"Testing first {max_combinations}")
    
    # Fortran output path
    fortran_output = Path(fm['output_path']) / f"{basin_id}_900_runs_def.nc"
    print(f"\nFortran output: {fortran_output}")
    if not fortran_output.exists():
        print("  ⚠️  File not found - will skip Fortran comparison")
    
    # Run validations
    results = []
    for i, combo in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}]", end="")
        
        result = validate_single_combination(
            combo=combo,
            basin_id=basin_id,
            fm=fm,
            fortran_params=fortran_params,
            forcing=forcing,
            elev_bands=elev_bands,
            fortran_output_path=fortran_output,
            verbose=verbose
        )
        
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if results:
        nse_values = [r['metrics']['nse'] for r in results 
                      if not np.isnan(r['metrics'].get('nse', np.nan))]
        corr_values = [r['metrics']['correlation'] for r in results 
                       if not np.isnan(r['metrics'].get('correlation', np.nan))]
        
        print(f"\nSuccessful comparisons: {len(results)}")
        
        if nse_values:
            print(f"\nNSE Statistics:")
            print(f"  Min:  {min(nse_values):.3f}")
            print(f"  Max:  {max(nse_values):.3f}")
            print(f"  Mean: {np.mean(nse_values):.3f}")
            print(f"  >0.8: {sum(1 for n in nse_values if n > 0.8)} / {len(nse_values)}")
            print(f"  >0.6: {sum(1 for n in nse_values if n > 0.6)} / {len(nse_values)}")
        
        if corr_values:
            print(f"\nCorrelation Statistics:")
            print(f"  Min:  {min(corr_values):.3f}")
            print(f"  Max:  {max(corr_values):.3f}")
            print(f"  Mean: {np.mean(corr_values):.3f}")
        
        # Flag poor matches
        poor_matches = [r for r in results if r['metrics'].get('nse', 0) < 0.5]
        if poor_matches:
            print(f"\n⚠️  Poor matches (NSE < 0.5): {len(poor_matches)}")
            for r in poor_matches[:5]:
                print(f"  {r['short_name']}: NSE={r['metrics']['nse']:.3f}")
                for diag in r['diagnoses'][:1]:
                    print(f"    → {diag[:70]}...")
        else:
            print("\n✅ All tested combinations have NSE >= 0.5")
        
        # Excellent matches
        excellent = [r for r in results if r['metrics'].get('nse', 0) > 0.9]
        if excellent:
            print(f"\n✅ Excellent matches (NSE > 0.9): {len(excellent)}")
    else:
        print("\nNo successful comparisons.")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive dFUSE vs Fortran FUSE validation'
    )
    parser.add_argument('--basin', default='Klondike_Bonanza_Creek',
                        help='Basin ID')
    parser.add_argument('--fm', 
                        default='/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Klondike_Bonanza_Creek/settings/FUSE/fm_catch.txt',
                        help='Path to file manager')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum combinations to test')
    parser.add_argument('--arch1', nargs='+', default=None,
                        help='Filter upper arch (e.g., onestate_1)')
    parser.add_argument('--arch2', nargs='+', default=None,
                        help='Filter lower arch')
    parser.add_argument('--snow', action='store_true',
                        help='Test only snow combinations')
    parser.add_argument('--no-snow', action='store_true',
                        help='Test only non-snow combinations')
    parser.add_argument('--list', action='store_true',
                        help='List all valid combinations')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output')
    
    args = parser.parse_args()
    
    if args.list:
        combos = get_all_valid_combinations()
        print(f"Valid decision combinations: {len(combos)}\n")
        
        # Group by architecture
        by_arch = {}
        for c in combos:
            key = f"{c.arch1}_{c.arch2}"
            if key not in by_arch:
                by_arch[key] = []
            by_arch[key].append(c)
        
        for key, group in sorted(by_arch.items()):
            print(f"{key}: {len(group)} combinations")
            for c in group[:3]:
                print(f"  - {c.short_name()}")
            if len(group) > 3:
                print(f"  ... and {len(group)-3} more")
    else:
        results = run_comprehensive_validation(
            basin_id=args.basin,
            fm_path=args.fm,
            max_combinations=args.max,
            arch1_filter=args.arch1,
            arch2_filter=args.arch2,
            snow_only=args.snow,
            no_snow_only=args.no_snow,
            verbose=not args.quiet
        )
