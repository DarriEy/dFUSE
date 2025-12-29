"""
cFUSE - Differentiable FUSE Hydrological Model

A differentiable implementation of the FUSE (Framework for Understanding 
Structural Errors) hydrological model with Enzyme automatic differentiation.

Example usage:
    from cfuse import FUSEConfig, VIC_CONFIG
    from cfuse.io import read_fuse_forcing, read_elevation_bands
"""

__version__ = "0.4.0"

# Import core configuration
from .config import (
    FUSEConfig,
    VIC_CONFIG,
    TOPMODEL_CONFIG,
    ARNO_CONFIG,
    PRMS_CONFIG,
    SACRAMENTO_CONFIG,
    PARAM_NAMES,
    PARAM_BOUNDS,
    DEFAULT_PARAMS,
    get_default_params_array,
    UpperLayerArch,
    LowerLayerArch,
    BaseflowType,
    PercolationType,
    SurfaceRunoffType,
    EvaporationType,
    InterflowType,
)

# Import I/O utilities
from .io import (
    read_fuse_forcing,
    read_elevation_bands,
    parse_file_manager,
    parse_fuse_decisions,
    FUSEForcing,
    FUSEElevationBands,
)

# Try to import the compiled C++ core
try:
    import cfuse_core as core
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    core = None

__all__ = [
    # Version
    "__version__",
    # Configuration classes
    "FUSEConfig",
    "UpperLayerArch",
    "LowerLayerArch",
    "BaseflowType",
    "PercolationType",
    "SurfaceRunoffType",
    "EvaporationType",
    "InterflowType",
    # Preset configs
    "VIC_CONFIG",
    "TOPMODEL_CONFIG", 
    "ARNO_CONFIG",
    "PRMS_CONFIG",
    "SACRAMENTO_CONFIG",
    # Parameters
    "PARAM_NAMES",
    "PARAM_BOUNDS",
    "DEFAULT_PARAMS",
    "get_default_params_array",
    # I/O
    "read_fuse_forcing",
    "read_elevation_bands",
    "parse_file_manager",
    "parse_fuse_decisions",
    "FUSEForcing",
    "FUSEElevationBands",
    # Core
    "core",
    "HAS_CORE",
]
