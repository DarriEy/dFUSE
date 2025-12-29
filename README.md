# cFUSE - Differentiable FUSE

[![CI](https://github.com/DarriEy/cFUSE/actions/workflows/ci.yml/badge.svg)](https://github.com/DarriEy/cFUSE/actions/workflows/ci.yml)
[![Python Tests](https://github.com/DarriEy/cFUSE/actions/workflows/python-tests.yml/badge.svg)](https://github.com/DarriEy/cFUSE/actions/workflows/python-tests.yml)

A differentiable implementation of the FUSE hydrological model framework with Enzyme automatic differentiation.

**Note cFUSE is in active development**

## Features

- **Differentiable physics**: Full gradient computation via Enzyme AD
- **PyTorch integration**: Custom autograd function for gradient-based calibration

## Quick Start

### 1. Install Dependencies

```bash
# Python dependencies
pip install numpy torch netCDF4 tqdm matplotlib

# macOS: Install LLVM 19 for Enzyme
brew install llvm@19
```

### 2. Build Enzyme (one-time setup)

```bash
git clone https://github.com/EnzymeAD/Enzyme.git
cd Enzyme && mkdir build && cd build
cmake ../enzyme -DLLVM_DIR=$(brew --prefix llvm@19)/lib/cmake/llvm
make -j
sudo cp Enzyme/ClangEnzyme-19.dylib /opt/homebrew/lib/
```

### 3. Build cFUSE

```bash
cd cFUSE
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DDFUSE_BUILD_PYTHON=ON \
    -DDFUSE_USE_NETCDF=ON \
    -DDFUSE_USE_ENZYME=ON \
    -DCMAKE_CXX_COMPILER=$(brew --prefix llvm@19)/bin/clang++
make -j
cp cfuse_core*.so ../python/
```

### 3b. Build Python package with Enzyme + NetCDF (optional)

```bash
# Use local Enzyme plugin and enable NetCDF in the build
CMAKE_ARGS="-DDFUSE_USE_ENZYME=ON -DDFUSE_USE_NETCDF=ON -DENZYME_PLUGIN=/path/to/ClangEnzyme-19.dylib" \
    python -m build --no-isolation --skip-dependency-check
```

### 4. Run  Example Optimization

```bash
cd path/to/cFUSE/python
python optimize_basin.py
```
## Command Line Options

```bash
python optimize_basin.py --help

# Examples:
python optimize_basin.py --iterations 700          # More iterations
python optimize_basin.py --lr 0.05                 # Lower learning rate
python optimize_basin.py --loss nse                # Optimize NSE instead of KGE
python optimize_basin.py --spinup-days 730         # 2-year spinup
```

## Test Data

Example data is distributed as a release asset starting with `v0.4.0`.
Download the Bow-at-Banff dataset from
https://github.com/DarriEy/cFUSE/releases/tag/v0.4.0 and extract it so the path
is `data/domain_Bow_at_Banff_lumped_era5/` before running examples.

- **Basin**: Bow River at Banff (2210 km²)
- **Forcing**: ERA5 reanalysis (precipitation, temperature, PET)
- **Observations**: Streamflow for validation

## Project Structure

```
cFUSE/
├── data/                    # Downloaded example data (release asset)
├── include/cfuse/          # C++ headers
├── python/
│   ├── optimize_basin.py   # Example optimization script
│   └── cfuse/              # Python package
├── build/                  # Build output
└── README.md
```

## License

MIT License

## Acknowledgments

- Original FUSE: Clark et al. (2008), Water Resources Research
- Enzyme AD: Moses & Churavy (2020), NeurIPS
