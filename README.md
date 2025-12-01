# dFUSE - Differentiable FUSE

[![CI](https://github.com/DarriEy/dFUSE/actions/workflows/ci.yml/badge.svg)](https://github.com/DarriEy/dFUSE/actions/workflows/ci.yml)
[![Python Tests](https://github.com/DarriEy/dFUSE/actions/workflows/python-tests.yml/badge.svg)](https://github.com/DarriEy/dFUSE/actions/workflows/python-tests.yml)

**Version 0.3.0** - A differentiable implementation of the FUSE hydrological model framework in C++, with Enzyme automatic differentiation.

> **Note:** dFUSE is in active development.

## Overview

dFUSE implements the modular FUSE (Framework for Understanding Structural Errors) hydrological model with automatic differentiation support via [Enzyme](https://enzyme.mit.edu/). This enables gradient-based parameter optimization using modern deep learning optimizers.

Based on [Clark et al. (2008)](http://dx.doi.org/10.1029/2007WR006735) "Framework for Understanding Structural Errors (FUSE): A modular framework to diagnose differences between hydrological models", Water Resources Research.

## Features

- **Modular Architecture**: 7 structural decisions with 2-4 options each (792 model configurations)
- **Enzyme AD**: Automatic differentiation through the full physics simulation
- **Elevation Bands**: Multi-band snow modeling with lapse rate corrections
- **Smooth Physics**: Logistic approximations for discontinuities (Kavetski & Kuczera, 2007)
- **PyTorch Integration**: Custom autograd function for gradient-based optimization
- **Unit Hydrograph Routing**: Gamma distribution-based flow routing

## Quick Start

### Prerequisites

- Python 3.9+ with NumPy, PyTorch, netCDF4
- C++17 compiler (for building the native module)
- LLVM 19 with Enzyme plugin (for gradient support)

### Installation

**Option 1: pip install (Python only)**
```bash
pip install -e .
```

This installs the Python package. You still need to build the C++ extension separately for gradient computation.

**Option 2: Full build with C++ extension**
```bash
# Build C++ module
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DDFUSE_BUILD_PYTHON=ON \
    -DDFUSE_USE_ENZYME=ON \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++
make -j

# Copy compiled module to package
cp dfuse_core*.so ../python/dfuse/

# Install Python package
cd ..
pip install -e .
```

### Running Optimization

**Command line:**
```bash
dfuse-optimize --help
```

**Python API:**
```python
from dfuse import FUSEConfig, VIC_CONFIG, PARAM_NAMES
from dfuse.io import read_fuse_forcing, read_elevation_bands
import dfuse_core  # C++ extension (must be built separately)

# Load data
forcing = read_fuse_forcing("forcing.nc")
bands = read_elevation_bands("bands.nc")

# Run simulation
result = dfuse_core.run_fuse_elevation_bands(
    initial_state, forcing, params, 
    VIC_CONFIG.to_dict(),
    bands.area_frac, bands.mean_elev, ref_elev
)
```

### Comparing with Fortran FUSE

```bash
python compare_fortran.py \
    --forcing /path/to/forcing.nc \
    --fuse-output /path/to/fuse_output.nc
```

## Project Structure

```
dFUSE/
├── include/dfuse/       # C++ headers
│   ├── config.hpp       # Model configuration enums
│   ├── state.hpp        # State and flux structures
│   ├── physics.hpp      # Physics computations
│   ├── enzyme_ad.hpp    # Enzyme AD integration
│   ├── kernels.hpp      # Simulation kernels
│   ├── solver.hpp       # ODE solvers
│   ├── routing.hpp      # Flow routing
│   ├── netcdf_io.hpp    # NetCDF I/O
│   └── simulation.hpp   # High-level simulation
├── src/
│   └── dfuse_cli.cpp    # Command-line interface
├── python/
│   ├── bindings.cpp     # pybind11 bindings
│   ├── dfuse.py         # Python model definitions
│   ├── dfuse_netcdf.py  # NetCDF data loading
│   ├── optimize_basin.py    # Gradient-based optimization
│   └── compare_fortran.py   # Validation against Fortran FUSE
├── tests/
│   └── test_comprehensive.cpp  # Unit tests
├── examples/
│   └── single_basin.cpp # Example usage
└── CMakeLists.txt
```

## Command-Line Interface

If built with NetCDF support, dFUSE provides a CLI for running simulations:

```bash
# Build with NetCDF
cmake .. -DDFUSE_USE_NETCDF=ON ...
make

# Run simulation
./dfuse --forcing forcing.nc --output output.nc --config config.txt
```

## Model Architecture

### Structural Decisions

| Decision | Options |
|----------|---------|
| Upper Layer | Single state, Tension/Free split, Tension cascade |
| Lower Layer | Single state, Tension/Free split, Split with multiple free states |
| Surface Runoff | ARNO/VIC, PRMS, TOPMODEL |
| Percolation | Drainage at field capacity, Gravity-driven, Saturation excess |
| Evaporation | Root weighting, Sequential |
| Interflow | Linear, Nonlinear |
| Baseflow | Linear, Nonlinear |

### Parameters

29 parameters total including bucket capacities, transfer rates, shape parameters, and snow/routing parameters.

## Smooth Physics

All physics computations use smooth approximations to avoid discontinuities that would break gradient computation:

```cpp
// Logistic smoothing for bucket overflow
Real logistic_overflow(Real S, Real S_max, Real w) {
    Real x = (S - S_max - w * 5) / w;
    return 1.0 / (1.0 + exp(-x));
}
```

## References

- Clark, M. P., et al. (2008). Framework for Understanding Structural Errors (FUSE). Water Resources Research, 44(12). [doi:10.1029/2007WR006735](http://dx.doi.org/10.1029/2007WR006735)
- Kavetski, D., & Kuczera, G. (2007). Model smoothing strategies to remove microscale discontinuities. Water Resources Research, 43(3).

## License

GNU General Public License v3.0 (same as original FUSE)

## Continuous Integration

The project uses GitHub Actions for CI:

- **CI**: Builds and tests on Ubuntu and macOS (without Enzyme for simplicity)
- **Python Tests**: Tests the Python package across Python 3.9-3.12
- **Enzyme Build** (optional): Builds with Enzyme AD on macOS

## Acknowledgments

- Original FUSE implementation by Martyn Clark and collaborators at NCAR
- [Enzyme AD project](https://enzyme.mit.edu/) for automatic differentiation support
