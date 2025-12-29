"""
cFUSE Configuration Classes and Constants

This module provides model configuration classes, parameter definitions,
and preset configurations for different hydrological model structures.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict
import numpy as np


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
            'enable_snow': self.enable_snow,
        }


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRMS_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.TENSION2_FREE,
    lower_arch=LowerLayerArch.SINGLE_NOEVAP,
    baseflow=BaseflowType.LINEAR,
    percolation=PercolationType.FREE_STORAGE,
    surface_runoff=SurfaceRunoffType.UZ_LINEAR,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.LINEAR,
)

SACRAMENTO_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.TENSION_FREE,
    lower_arch=LowerLayerArch.TENSION_2RESERV,
    baseflow=BaseflowType.PARALLEL_LINEAR,
    percolation=PercolationType.LOWER_DEMAND,
    surface_runoff=SurfaceRunoffType.UZ_LINEAR,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.NONE,
)

TOPMODEL_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.SINGLE_STATE,
    lower_arch=LowerLayerArch.SINGLE_NOEVAP,
    baseflow=BaseflowType.TOPMODEL,
    percolation=PercolationType.TOTAL_STORAGE,
    surface_runoff=SurfaceRunoffType.LZ_GAMMA,
    evaporation=EvaporationType.SEQUENTIAL,
    interflow=InterflowType.NONE,
)

VIC_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.TENSION_FREE,
    lower_arch=LowerLayerArch.SINGLE_EVAP,
    baseflow=BaseflowType.NONLINEAR,
    percolation=PercolationType.TOTAL_STORAGE,
    surface_runoff=SurfaceRunoffType.UZ_PARETO,
    evaporation=EvaporationType.ROOT_WEIGHT,
    interflow=InterflowType.NONE,
)

ARNO_CONFIG = FUSEConfig(
    upper_arch=UpperLayerArch.TENSION_FREE,
    lower_arch=LowerLayerArch.SINGLE_EVAP,
    baseflow=BaseflowType.NONLINEAR,
    percolation=PercolationType.TOTAL_STORAGE,
    surface_runoff=SurfaceRunoffType.UZ_PARETO,
    evaporation=EvaporationType.ROOT_WEIGHT,
    interflow=InterflowType.NONE,
)


# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

PARAM_NAMES = [
    'S1_max', 'S2_max', 'f_tens', 'f_rchr', 'f_base', 'r1',
    'ku', 'c', 'alpha', 'psi', 'kappa', 'ki',
    'ks', 'n', 'v', 'v_A', 'v_B',
    'Ac_max', 'b', 'lambda', 'chi', 'mu_t',
    'T_rain', 'T_melt', 'melt_rate', 'lapse_rate', 'opg',
    'MFMAX', 'MFMIN'
]

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
    'lapse_rate': (-9.8, 0.0),
    'opg': (0.0, 1.0),
    'MFMAX': (1.0, 10.0),
    'MFMIN': (0.0, 10.0),
}

DEFAULT_PARAMS = {
    'S1_max': 200.0,
    'S2_max': 2000.0,
    'f_tens': 0.5,
    'f_rchr': 0.5,
    'f_base': 0.5,
    'r1': 0.5,
    'ku': 10.0,
    'c': 4.0,
    'alpha': 100.0,
    'psi': 2.0,
    'kappa': 0.5,
    'ki': 10.0,
    'ks': 100.0,
    'n': 2.0,
    'v': 0.1,
    'v_A': 0.1,
    'v_B': 0.1,
    'Ac_max': 0.5,
    'b': 1.0,
    'lambda': 7.5,
    'chi': 3.5,
    'mu_t': 1.0,
    'T_rain': 1.0,
    'T_melt': 0.0,
    'melt_rate': 3.0,
    'lapse_rate': -6.5,
    'opg': 0.0,
    'MFMAX': 4.0,
    'MFMIN': 1.0,
}


def get_default_params_array() -> np.ndarray:
    """Get default parameters as numpy array in correct order"""
    return np.array([DEFAULT_PARAMS[name] for name in PARAM_NAMES], dtype=np.float32)
