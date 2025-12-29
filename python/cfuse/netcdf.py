"""
cFUSE NetCDF I/O Module

Provides NetCDF reading/writing for FUSE input/output files,
compatible with the Fortran FUSE file formats.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
import re

# Try to import netCDF4
try:
    import netCDF4 as nc
    HAS_NETCDF = True
except ImportError:
    try:
        from scipy.io import netcdf_file
        HAS_NETCDF = True
        nc = None  # Use scipy fallback
    except ImportError:
        HAS_NETCDF = False
        nc = None

# Try to import xarray for convenience
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


@dataclass
class FUSEForcing:
    """Container for FUSE forcing data"""
    time: np.ndarray          # Time in days since reference
    precip: np.ndarray        # Precipitation [mm/day]
    pet: np.ndarray           # Potential evapotranspiration [mm/day]
    temp: np.ndarray          # Temperature [°C]
    q_obs: Optional[np.ndarray] = None  # Observed discharge [mm/day]
    
    # Metadata
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    time_units: Optional[str] = None
    
    @property
    def n_timesteps(self) -> int:
        return len(self.time)
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert to PyTorch tensor [n_timesteps, 3] for (precip, pet, temp)"""
        forcing = np.stack([self.precip, self.pet, self.temp], axis=1)
        return torch.from_numpy(forcing.astype(np.float32)).to(device)
    
    def subset(self, start_idx: int, end_idx: int) -> 'FUSEForcing':
        """Get subset of forcing data"""
        return FUSEForcing(
            time=self.time[start_idx:end_idx],
            precip=self.precip[start_idx:end_idx],
            pet=self.pet[start_idx:end_idx],
            temp=self.temp[start_idx:end_idx],
            q_obs=self.q_obs[start_idx:end_idx] if self.q_obs is not None else None,
            start_date=self.start_date,
            end_date=self.end_date,
            time_units=self.time_units
        )


@dataclass
class FUSEElevationBands:
    """Container for elevation band data"""
    n_bands: int
    area_frac: np.ndarray     # Fraction of catchment per band
    mean_elev: np.ndarray     # Mean elevation per band [m]
    prec_frac: np.ndarray     # Precipitation fraction per band


@dataclass 
class FortranParameters:
    """Container for Fortran FUSE parameters"""
    # Storage parameters
    MAXWATR_1: float = 100.0   # Upper layer depth [mm]
    MAXWATR_2: float = 1000.0  # Lower layer depth [mm]
    FRACTEN: float = 0.5       # Fraction in tension storage
    FRCHZNE: float = 0.5       # Fraction tension in recharge zone
    FPRIMQB: float = 0.5       # Fraction in 1st baseflow reservoir
    RTFRAC1: float = 0.75      # Fraction of roots in upper layer
    
    # Percolation parameters
    PERCRTE: float = 100.0     # Percolation rate [mm/day]
    PERCEXP: float = 5.0       # Percolation exponent
    SACPMLT: float = 10.0      # SAC percolation multiplier
    SACPEXP: float = 5.0       # SAC percolation exponent
    PERCFRAC: float = 0.5      # Fraction percolation to tension
    FRACLOWZ: float = 0.5      # Fraction excess to lower zone
    
    # Flow parameters
    IFLWRTE: float = 500.0     # Interflow rate [mm/day]
    BASERTE: float = 50.0      # Baseflow rate [mm/day]
    QB_POWR: float = 5.0       # Baseflow exponent
    QB_PRMS: float = 0.01      # Baseflow depletion rate [day-1]
    QBRATE_2A: float = 0.025   # Baseflow rate 1st reservoir [day-1]
    QBRATE_2B: float = 0.01    # Baseflow rate 2nd reservoir [day-1]
    
    # Surface runoff parameters
    SAREAMAX: float = 0.25     # Maximum saturated area
    AXV_BEXP: float = 0.3      # ARNO/VIC 'b' exponent
    LOGLAMB: float = 7.5       # Mean topographic index
    TISHAPE: float = 3.0       # Topo index Gamma shape
    
    # Routing
    TIMEDELAY: float = 0.9     # Time delay [days]
    
    # Snow parameters
    MBASE: float = 1.0         # Base melt temperature [°C]
    MFMAX: float = 4.2         # Max melt factor [mm/°C/day]
    MFMIN: float = 2.4         # Min melt factor [mm/°C/day]
    PXTEMP: float = 1.0        # Rain-snow partition temp [°C]
    OPG: float = 0.5           # Precip elevation gradient [km-1]
    LAPSE: float = -5.0        # Temperature lapse rate [°C/km]
    
    # Rainfall error
    RFERR_ADD: float = 0.0     # Additive rainfall error [mm]
    RFERR_MLT: float = 1.0     # Multiplicative rainfall error
    
    def to_cfuse_params(self, arch2: str = 'unlimpow_2') -> np.ndarray:
        """
        Convert Fortran FUSE parameters to cFUSE parameter array.
        
        cFUSE parameter order (from PARAM_NAMES):
        0: S1_max, 1: S2_max, 2: f_tens, 3: f_rchr, 4: f_base, 5: r1,
        6: ku, 7: c, 8: alpha, 9: psi, 10: kappa, 11: ki,
        12: ks, 13: n, 14: v, 15: v_A, 16: v_B,
        17: Ac_max, 18: b, 19: lambda, 20: chi, 21: mu_t,
        22: T_rain, 23: T_melt, 24: melt_rate, 25: lapse_rate, 26: opg
        27: MFMAX, 28: MFMIN
        
        Args:
            arch2: Lower layer architecture (unlimfrc_2, unlimpow_2, fixedsiz_2, tens2pll_2)
                   For 'unlim*' architectures, S2_max is set to a large value.
        """
        params = np.zeros(29, dtype=np.float32)
        
        # Storage
        params[0] = self.MAXWATR_1           # S1_max
        
        # For "unlimited" lower layer architectures, use a very large S2_max
        # This makes baseflow negligible: qb = ks * (S2/S2_max)^n → 0 when S2_max >> S2
        if arch2 in ('unlimfrc_2', 'unlimpow_2'):
            params[1] = 1e10  # Effectively infinite for baseflow computation
        else:
            params[1] = self.MAXWATR_2       # S2_max
        
        params[2] = self.FRACTEN             # f_tens (tension fraction)
        params[3] = self.FRCHZNE             # f_rchr (recharge zone fraction)
        params[4] = self.FPRIMQB             # f_base (baseflow reservoir fraction)
        
        # Evaporation
        params[5] = self.RTFRAC1             # r1 (root fraction)
        
        # Percolation
        params[6] = self.PERCRTE             # ku (percolation rate)
        params[7] = self.PERCEXP             # c (percolation exponent)
        params[8] = self.SACPMLT             # alpha (SAC multiplier)
        params[9] = self.SACPEXP             # psi (SAC exponent)
        params[10] = self.PERCFRAC           # kappa (percolation fraction)
        
        # Interflow
        params[11] = self.IFLWRTE            # ki (interflow rate)
        
        # Baseflow
        params[12] = self.BASERTE            # ks (baseflow rate)
        params[13] = self.QB_POWR            # n (baseflow exponent)
        params[14] = self.QB_PRMS            # v (linear baseflow rate)
        params[15] = self.QBRATE_2A          # v_A (parallel reservoir A)
        params[16] = self.QBRATE_2B          # v_B (parallel reservoir B)
        
        # Surface runoff
        params[17] = self.SAREAMAX           # Ac_max (max saturated area)
        params[18] = self.AXV_BEXP           # b (VIC exponent)
        
        # TOPMODEL
        params[19] = self.LOGLAMB            # lambda (topographic index)
        params[20] = self.TISHAPE            # chi (shape parameter)
        
        # Routing
        params[21] = self.TIMEDELAY          # mu_t (time delay)
        
        # Snow
        params[22] = self.PXTEMP             # T_rain (rain-snow threshold)
        params[23] = self.MBASE              # T_melt (base melt temperature)
        params[24] = (self.MFMAX + self.MFMIN) / 2.0  # melt_rate (fallback if MFMAX/MFMIN not used)
        params[25] = self.LAPSE              # lapse_rate (°C/km)
        params[26] = self.OPG                # opg (km-1)
        params[27] = self.MFMAX              # MFMAX (max melt factor on June 21)
        params[28] = self.MFMIN              # MFMIN (min melt factor on Dec 21)
        
        return params

    def to_dfuse_params(self, arch2: str = 'unlimpow_2') -> np.ndarray:
        """Backward-compatible alias for to_cfuse_params."""
        return self.to_cfuse_params(arch2=arch2)
    
    def get_melt_rate(self) -> float:
        """Get average melt rate from MFMAX/MFMIN"""
        return (self.MFMAX + self.MFMIN) / 2.0


def parse_fortran_constraints(filepath: Union[str, Path]) -> FortranParameters:
    """
    Parse Fortran FUSE parameter constraints file.
    
    Args:
        filepath: Path to fuse_zConstraints*.txt file
        
    Returns:
        FortranParameters with default values from file
    """
    filepath = Path(filepath)
    params = FortranParameters()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip header and comments
            if not line or line.startswith('(') or line.startswith('*') or line.startswith('!'):
                continue
            
            parts = line.split()
            if len(parts) < 6:
                continue
            
            try:
                # Format: fit_flag stoch_flag default lower upper ... param_name
                # Find parameter name (usually near end, all caps with underscore)
                param_name = None
                for part in parts:
                    if part.isupper() and '_' in part and not part.startswith('NO_'):
                        param_name = part
                        break
                    elif part in ['FRACTEN', 'FRCHZNE', 'FPRIMQB', 'RTFRAC1', 'PERCEXP', 
                                  'SACPMLT', 'SACPEXP', 'PERCFRAC', 'FRACLOWZ', 'IFLWRTE',
                                  'BASERTE', 'SAREAMAX', 'LOGLAMB', 'TISHAPE', 'TIMEDELAY',
                                  'MBASE', 'MFMAX', 'MFMIN', 'PXTEMP', 'OPG', 'LAPSE']:
                        param_name = part
                        break
                
                if param_name and hasattr(params, param_name):
                    # Default value is the 3rd numeric value (index 2 after fit and stoch flags)
                    default_val = float(parts[2])
                    setattr(params, param_name, default_val)
                    
            except (ValueError, IndexError):
                continue
    
    return params


@dataclass
class FUSEDecisions:
    """FUSE model structure decisions"""
    # Upper layer architecture
    rferr: str = 'additive_e'      # Rainfall error (additive_e, multiplc_e)
    arch1: str = 'onestate_1'      # Upper layer (onestate_1, tension1_1, tension2_1)
    arch2: str = 'unlimfrc_2'      # Lower layer (unlimfrc_2, unlimpow_2, fixedsiz_2, tens2pll_2)
    qsurf: str = 'arno_x_vic'      # Surface runoff (arno_x_vic, prms_varnt, tmdl_param)
    qperc: str = 'perc_f2sat'      # Percolation (perc_f2sat, perc_lower, perc_w2sat)
    esoil: str = 'rootweight'      # Evaporation (rootweight, sequential)
    qintf: str = 'intflwsome'      # Interflow (intflwnone, intflwsome)
    q_tdh: str = 'rout_gamma'      # Routing (no_routing, rout_gamma)
    snowmod: str = 'temp_index'    # Snow (temp_index, no_snowmod)
    
    def to_config_dict(self) -> Dict:
        """Convert decisions to cFUSE config dict"""
        # Map FUSE decision strings to cFUSE enum values
        
        # Upper layer architecture
        upper_arch_map = {
            'onestate_1': 0,   # SINGLE_STATE
            'tension1_1': 1,   # TENSION_FREE
            'tension2_1': 2,   # TENSION2_FREE
        }
        
        # Lower layer architecture  
        lower_arch_map = {
            'unlimfrc_2': 1,   # SINGLE_EVAP (unlimited)
            'unlimpow_2': 1,   # SINGLE_EVAP (power)
            'fixedsiz_2': 0,   # SINGLE_NOEVAP
            'tens2pll_2': 2,   # TENSION_2RESERV
            'topmdexp_2': 1,   # SINGLE_EVAP with TOPMODEL
        }
        
        # Surface runoff
        qsurf_map = {
            'arno_x_vic': 1,   # UZ_PARETO (VIC/Arno)
            'prms_varnt': 0,   # UZ_LINEAR (PRMS)
            'tmdl_param': 2,   # LZ_GAMMA (TOPMODEL)
        }
        
        # Baseflow
        baseflow_map = {
            'unlimfrc_2': 0,   # LINEAR
            'unlimpow_2': 2,   # NONLINEAR
            'fixedsiz_2': 0,   # LINEAR
            'tens2pll_2': 1,   # PARALLEL_LINEAR
            'topmdexp_2': 3,   # TOPMODEL
        }
        
        # Percolation
        perc_map = {
            'perc_f2sat': 1,   # FREE_STORAGE - percolation from free storage only
            'perc_w2sat': 0,   # TOTAL_STORAGE - percolation from total water content
            'perc_lower': 2,   # LOWER_DEMAND - driven by lower zone (Sacramento)
        }
        
        # Evaporation
        evap_map = {
            'sequential': 0,   # SEQUENTIAL
            'rootweight': 1,   # ROOT_WEIGHT
        }
        
        # Interflow
        interflow_map = {
            'intflwnone': 0,   # NONE
            'intflwsome': 1,   # LINEAR
        }
        
        return {
            'upper_arch': upper_arch_map.get(self.arch1, 0),
            'lower_arch': lower_arch_map.get(self.arch2, 1),
            'baseflow': baseflow_map.get(self.arch2, 2),
            'percolation': perc_map.get(self.qperc, 0),
            'surface_runoff': qsurf_map.get(self.qsurf, 1),
            'evaporation': evap_map.get(self.esoil, 0),
            'interflow': interflow_map.get(self.qintf, 0),
            'enable_snow': self.snowmod != 'no_snowmod',
        }


def read_fuse_forcing(filepath: Union[str, Path]) -> FUSEForcing:
    """
    Read FUSE forcing NetCDF file.
    
    Args:
        filepath: Path to NetCDF file with pr, temp, pet variables
        
    Returns:
        FUSEForcing object with forcing data
    """
    filepath = Path(filepath)
    
    if not HAS_NETCDF:
        raise ImportError("netCDF4 or scipy required. Install with: pip install netCDF4")
    
    if HAS_XARRAY:
        # Use xarray for convenience
        ds = xr.open_dataset(filepath)
        
        # Handle different dimension orders
        def get_1d_var(var):
            data = ds[var].values
            # Squeeze out singleton dimensions
            return data.squeeze()
        
        forcing = FUSEForcing(
            time=ds['time'].values.astype('datetime64[D]').astype(float),
            precip=get_1d_var('pr'),
            pet=get_1d_var('pet'),
            temp=get_1d_var('temp'),
            q_obs=get_1d_var('q_obs') if 'q_obs' in ds else None,
            time_units=ds['time'].attrs.get('units', None)
        )
        ds.close()
        return forcing
    
    else:
        # Use netCDF4 directly
        with nc.Dataset(filepath, 'r') as ds:
            time = ds.variables['time'][:]
            precip = ds.variables['pr'][:].squeeze()
            pet = ds.variables['pet'][:].squeeze()
            temp = ds.variables['temp'][:].squeeze()
            q_obs = ds.variables['q_obs'][:].squeeze() if 'q_obs' in ds.variables else None
            time_units = ds.variables['time'].units if hasattr(ds.variables['time'], 'units') else None
            
        # Handle masked arrays
        if hasattr(precip, 'filled'):
            precip = precip.filled(np.nan)
        if hasattr(pet, 'filled'):
            pet = pet.filled(np.nan)
        if hasattr(temp, 'filled'):
            temp = temp.filled(np.nan)
        if q_obs is not None and hasattr(q_obs, 'filled'):
            q_obs = q_obs.filled(np.nan)
            
        return FUSEForcing(
            time=time,
            precip=precip,
            pet=pet,
            temp=temp,
            q_obs=q_obs,
            time_units=time_units
        )


def read_elevation_bands(filepath: Union[str, Path]) -> FUSEElevationBands:
    """Read FUSE elevation bands NetCDF file."""
    filepath = Path(filepath)
    
    if HAS_XARRAY:
        ds = xr.open_dataset(filepath)
        # Use np.atleast_1d to handle single-band (lumped) basins
        area_frac = np.atleast_1d(ds['area_frac'].values.squeeze())
        mean_elev = np.atleast_1d(ds['mean_elev'].values.squeeze())
        prec_frac = np.atleast_1d(ds['prec_frac'].values.squeeze())
        bands = FUSEElevationBands(
            n_bands=len(area_frac),
            area_frac=area_frac,
            mean_elev=mean_elev,
            prec_frac=prec_frac
        )
        ds.close()
        return bands
    else:
        with nc.Dataset(filepath, 'r') as ds:
            area_frac = np.atleast_1d(ds.variables['area_frac'][:].squeeze())
            mean_elev = np.atleast_1d(ds.variables['mean_elev'][:].squeeze())
            prec_frac = np.atleast_1d(ds.variables['prec_frac'][:].squeeze())
            return FUSEElevationBands(
                n_bands=len(area_frac),
                area_frac=area_frac,
                mean_elev=mean_elev,
                prec_frac=prec_frac
            )


def parse_fuse_decisions(filepath: Union[str, Path]) -> FUSEDecisions:
    """
    Parse FUSE model decisions file.
    
    Fortran FUSE format:
        option_name DECISION_NAME ! comment
    
    Args:
        filepath: Path to fuse_zDecisions*.txt file
        
    Returns:
        FUSEDecisions object
    """
    filepath = Path(filepath)
    decisions = FUSEDecisions()
    
    # Map Fortran decision names to our attribute names
    decision_map = {
        'RFERR': 'rferr',
        'ARCH1': 'arch1',
        'ARCH2': 'arch2',
        'QSURF': 'qsurf',
        'QPERC': 'qperc',
        'ESOIL': 'esoil',
        'QINTF': 'qintf',
        'Q_TDH': 'q_tdh',
        'SNOWM': 'snowmod',
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('(') or line.startswith('-'):
                continue
            
            # Stop at the break line
            if line.startswith('0'):
                break
            
            # Parse: option_name DECISION_NAME ! comment
            parts = line.split()
            if len(parts) >= 2:
                option_name = parts[0].lower()
                decision_name = parts[1].upper()
                
                if decision_name in decision_map:
                    attr_name = decision_map[decision_name]
                    setattr(decisions, attr_name, option_name)
    
    return decisions


def parse_file_manager(filepath: Union[str, Path]) -> Dict:
    """
    Parse FUSE file manager.
    
    Returns dict with paths and settings.
    """
    filepath = Path(filepath)
    
    keys = [
        'setngs_path', 'input_path', 'output_path',
        'suffix_forcing', 'suffix_elev_bands',
        'forcing_info', 'constraints', 'mod_numerix', 'm_decisions',
        'fmodel_id', 'q_only',
        'date_start_sim', 'date_end_sim', 'date_start_eval', 'date_end_eval',
        'numtim_sub', 'metric', 'transfo', 'maxn', 'kstop', 'pcento'
    ]
    
    fm = {}
    key_idx = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('FUSE_FILEMANAGER'):
                continue
            
            # Extract value (handle quoted strings)
            match = re.match(r"'([^']*)'", line)
            if match and key_idx < len(keys):
                fm[keys[key_idx]] = match.group(1).strip()
                key_idx += 1
    
    return fm


def write_fuse_output(
    filepath: Union[str, Path],
    time: np.ndarray,
    runoff: np.ndarray,
    states: Optional[Dict[str, np.ndarray]] = None,
    fluxes: Optional[Dict[str, np.ndarray]] = None,
    time_units: str = "days since 1970-01-01"
):
    """
    Write cFUSE output to NetCDF file.
    
    Args:
        filepath: Output path
        time: Time array
        runoff: Total runoff [mm/day]
        states: Optional dict of state variables
        fluxes: Optional dict of flux variables
        time_units: CF-compliant time units
    """
    if not HAS_NETCDF or nc is None:
        raise ImportError("netCDF4 required for writing. Install with: pip install netCDF4")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with nc.Dataset(filepath, 'w', format='NETCDF4') as ds:
        # Dimensions
        ds.createDimension('time', len(time))
        
        # Time variable
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var[:] = time
        time_var.units = time_units
        time_var.long_name = 'time'
        
        # Runoff
        q_var = ds.createVariable('q_routed', 'f4', ('time',), fill_value=-9999.0)
        q_var[:] = runoff
        q_var.units = 'mm/day'
        q_var.long_name = 'Simulated discharge'
        
        # States
        if states:
            for name, data in states.items():
                var = ds.createVariable(name, 'f4', ('time',), fill_value=-9999.0)
                var[:] = data
                var.long_name = name
        
        # Fluxes
        if fluxes:
            for name, data in fluxes.items():
                var = ds.createVariable(name, 'f4', ('time',), fill_value=-9999.0)
                var[:] = data
                var.long_name = name
        
        # Global attributes
        ds.title = 'cFUSE model output'
        ds.source = 'cFUSE - Differentiable FUSE'
        ds.history = f'Created by cFUSE'


class FUSERunner:
    """
    High-level interface for running cFUSE with NetCDF I/O.
    
    Provides compatibility with Fortran FUSE file structure.
    """
    
    def __init__(self, file_manager_path: Union[str, Path]):
        """
        Initialize runner from FUSE file manager.
        
        Args:
            file_manager_path: Path to fm_*.txt file
        """
        self.fm_path = Path(file_manager_path)
        self.fm = parse_file_manager(self.fm_path)
        
        # Parse decisions
        decisions_path = Path(self.fm['setngs_path']) / self.fm['m_decisions']
        self.decisions = parse_fuse_decisions(decisions_path)
        self.config_dict = self.decisions.to_config_dict()
        
        # Paths
        self.input_path = Path(self.fm['input_path'])
        self.output_path = Path(self.fm['output_path'])
        self.setngs_path = Path(self.fm['setngs_path'])
        
    def load_forcing(self, basin_id: str) -> FUSEForcing:
        """Load forcing data for a basin."""
        forcing_file = self.input_path / f"{basin_id}{self.fm['suffix_forcing']}"
        return read_fuse_forcing(forcing_file)
    
    def load_elevation_bands(self, basin_id: str) -> FUSEElevationBands:
        """Load elevation bands for a basin."""
        bands_file = self.input_path / f"{basin_id}{self.fm['suffix_elev_bands']}"
        return read_elevation_bands(bands_file)
    
    def run(
        self,
        basin_id: str,
        params: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_states: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Run cFUSE for a basin.
        
        Args:
            basin_id: Basin identifier
            params: Model parameters (uses defaults if None)
            return_states: Whether to return state trajectories
            
        Returns:
            runoff: Simulated runoff [mm/day]
            states: State trajectories if return_states=True
        """
        # Import here to avoid circular imports
        from cfuse.legacy import FUSE, FUSEConfig, get_default_constrained_params
        
        # Load forcing
        forcing_data = self.load_forcing(basin_id)
        forcing = forcing_data.to_tensor()
        
        # Create model
        config = FUSEConfig(**self.config_dict)
        model = FUSE(config=config, learnable_params=False)
        
        # Set parameters
        if params is not None:
            if isinstance(params, np.ndarray):
                params = torch.from_numpy(params.astype(np.float32))
            model.raw_params.data = params
        
        # Get initial state
        state = model.get_initial_state()
        
        # Run model
        with torch.no_grad():
            runoff = model(state, forcing)
        
        runoff_np = runoff.numpy()
        
        if return_states:
            # Would need to modify forward to return states
            return runoff_np, None
        else:
            return runoff_np, None
    
    def save_output(
        self,
        basin_id: str,
        runoff: np.ndarray,
        time: np.ndarray,
        states: Optional[Dict] = None
    ):
        """Save output to NetCDF."""
        output_file = self.output_path / f"{basin_id}_{self.fm['fmodel_id']}_output.nc"
        write_fuse_output(output_file, time, runoff, states)


def compare_with_fortran(
    file_manager_path: Union[str, Path],
    basin_id: str,
    fortran_exe: Union[str, Path],
    params: Optional[np.ndarray] = None,
    plot: bool = True
) -> Dict:
    """
    Run both cFUSE and Fortran FUSE and compare outputs.
    
    Args:
        file_manager_path: Path to FUSE file manager
        basin_id: Basin identifier
        fortran_exe: Path to Fortran FUSE executable
        params: Optional parameters (uses defaults if None)
        plot: Whether to generate comparison plots
        
    Returns:
        Dict with comparison metrics
    """
    import subprocess
    
    file_manager_path = Path(file_manager_path)
    fortran_exe = Path(fortran_exe)
    
    # Run Fortran FUSE
    print("Running Fortran FUSE...")
    result = subprocess.run(
        [str(fortran_exe), str(file_manager_path), basin_id, 'run_def'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Fortran FUSE failed: {result.stderr}")
    else:
        print("Fortran FUSE completed")
    
    # Run cFUSE
    print("Running cFUSE...")
    runner = FUSERunner(file_manager_path)
    forcing = runner.load_forcing(basin_id)
    runoff_dfuse, _ = runner.run(basin_id, params)
    print("cFUSE completed")
    
    # Load Fortran output
    fm = parse_file_manager(file_manager_path)
    output_path = Path(fm['output_path'])
    fortran_output_file = output_path / f"{basin_id}_{fm['fmodel_id']}_runs.nc"
    
    if fortran_output_file.exists():
        if HAS_XARRAY:
            ds = xr.open_dataset(fortran_output_file)
            runoff_fortran = ds['q_routed'].values.squeeze()
            ds.close()
        else:
            with nc.Dataset(fortran_output_file, 'r') as ds:
                runoff_fortran = ds.variables['q_routed'][:].squeeze()
    else:
        print(f"Warning: Fortran output not found at {fortran_output_file}")
        runoff_fortran = None
    
    # Compute metrics
    results = {
        'n_timesteps': len(runoff_dfuse),
        'cfuse_mean': float(np.nanmean(runoff_dfuse)),
        'cfuse_std': float(np.nanstd(runoff_dfuse)),
    }
    
    if runoff_fortran is not None:
        # Align lengths if needed
        min_len = min(len(runoff_dfuse), len(runoff_fortran))
        rd = runoff_dfuse[:min_len]
        rf = runoff_fortran[:min_len]
        
        # Remove NaNs for comparison
        valid = ~(np.isnan(rd) | np.isnan(rf))
        rd_valid = rd[valid]
        rf_valid = rf[valid]
        
        results['fortran_mean'] = float(np.nanmean(rf))
        results['fortran_std'] = float(np.nanstd(rf))
        results['rmse'] = float(np.sqrt(np.mean((rd_valid - rf_valid)**2)))
        results['mae'] = float(np.mean(np.abs(rd_valid - rf_valid)))
        results['correlation'] = float(np.corrcoef(rd_valid, rf_valid)[0, 1])
        results['bias'] = float(np.mean(rd_valid - rf_valid))
        
        # NSE
        ss_res = np.sum((rd_valid - rf_valid)**2)
        ss_tot = np.sum((rf_valid - np.mean(rf_valid))**2)
        results['nse'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        print(f"\nComparison Results:")
        print(f"  RMSE: {results['rmse']:.4f} mm/day")
        print(f"  MAE: {results['mae']:.4f} mm/day")
        print(f"  Correlation: {results['correlation']:.4f}")
        print(f"  NSE: {results['nse']:.4f}")
        print(f"  Bias: {results['bias']:.4f} mm/day")
        
        if plot:
            try:
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Time series
                ax = axes[0, 0]
                ax.plot(rf_valid[:365], label='Fortran FUSE', alpha=0.7)
                ax.plot(rd_valid[:365], label='cFUSE', alpha=0.7)
                ax.set_xlabel('Day')
                ax.set_ylabel('Runoff [mm/day]')
                ax.set_title('First Year Comparison')
                ax.legend()
                
                # Scatter plot
                ax = axes[0, 1]
                ax.scatter(rf_valid, rd_valid, alpha=0.3, s=5)
                max_val = max(rf_valid.max(), rd_valid.max())
                ax.plot([0, max_val], [0, max_val], 'r--', label='1:1')
                ax.set_xlabel('Fortran FUSE [mm/day]')
                ax.set_ylabel('cFUSE [mm/day]')
                ax.set_title(f'Scatter (R²={results["correlation"]**2:.3f})')
                ax.legend()
                
                # Residuals
                ax = axes[1, 0]
                residuals = rd_valid - rf_valid
                ax.hist(residuals, bins=50, edgecolor='black')
                ax.axvline(0, color='r', linestyle='--')
                ax.set_xlabel('Residual [mm/day]')
                ax.set_ylabel('Count')
                ax.set_title(f'Residuals (Bias={results["bias"]:.4f})')
                
                # Flow duration curve
                ax = axes[1, 1]
                exceedance = np.linspace(0, 100, len(rf_valid))
                ax.semilogy(exceedance, np.sort(rf_valid)[::-1], label='Fortran')
                ax.semilogy(exceedance, np.sort(rd_valid)[::-1], label='cFUSE')
                ax.set_xlabel('Exceedance [%]')
                ax.set_ylabel('Runoff [mm/day]')
                ax.set_title('Flow Duration Curve')
                ax.legend()
                
                plt.tight_layout()
                plt.savefig(output_path / f'{basin_id}_comparison.png', dpi=150)
                plt.show()
                print(f"\nPlot saved to: {output_path / f'{basin_id}_comparison.png'}")
                
            except ImportError:
                print("matplotlib not available for plotting")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m cfuse.netcdf <file_manager.txt> <basin_id> [fortran_exe]")
        sys.exit(1)
    
    fm_path = sys.argv[1]
    basin_id = sys.argv[2]
    
    if len(sys.argv) > 3:
        fortran_exe = sys.argv[3]
        results = compare_with_fortran(fm_path, basin_id, fortran_exe)
    else:
        # Just run cFUSE
        runner = FUSERunner(fm_path)
        forcing = runner.load_forcing(basin_id)
        print(f"Loaded forcing: {forcing.n_timesteps} timesteps")
        print(f"Precip range: {forcing.precip.min():.2f} - {forcing.precip.max():.2f} mm/day")
        print(f"Temp range: {forcing.temp.min():.2f} - {forcing.temp.max():.2f} °C")
        
        runoff, _ = runner.run(basin_id)
        print(f"Simulated runoff: mean={runoff.mean():.2f}, max={runoff.max():.2f} mm/day")
