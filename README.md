# dFUSE: Differentiable Framework for Understanding Structural Errors

**Version 0.2.0** - A GPU-native, differentiable implementation of the FUSE hydrological model framework.

Based on [Clark et al. (2008) "Framework for Understanding Structural Errors (FUSE): A modular framework to diagnose differences between hydrological models"](http://dx.doi.org/10.1029/2007WR006735), Water Resources Research.

## Overview

dFUSE reimplements the classic FUSE modular hydrological modeling framework in modern C++ with:

- **GPU Acceleration**: CUDA kernels for parallel execution across thousands of HRUs
- **Automatic Differentiation**: Enzyme AD integration for gradient-based parameter optimization
- **Multiple ODE Solvers**: Explicit Euler, Implicit Euler, SUNDIALS BDF/Adams-Moulton
- **Gamma Distribution Routing**: Unit hydrograph convolution for runoff timing
- **PyTorch Integration**: Use FUSE as a differentiable layer in deep learning workflows
- **All 79 Configurations**: Support for all valid model structure combinations from Clark et al. (2008)

## What's New in v0.2.0

- **SUNDIALS Integration**: CVODE solver with BDF (stiff) and Adams-Moulton (non-stiff) methods
- **Gamma Distribution Routing**: Clark et al. (2008) equation 13 implementation
- **Enzyme AD Support**: Exact gradient computation via reverse-mode automatic differentiation
- **Adjoint Method**: Efficient gradient computation for long time series
- **CUDA Workspace Optimization**: Pre-allocated GPU memory for zero-allocation training loops
- **Clang-CUDA + Enzyme**: Configuration support for GPU automatic differentiation
- **Comprehensive Validation**: 17-test suite validating all physics modules and configurations
- **Default Parameter Values**: Sensible defaults for all parameters following Clark et al. (2008)

## Features

### Physics Options (from Clark et al. 2008)

| Decision | Options |
|----------|---------|
| Upper Layer Architecture | Single state, Tension+Free, Two tension+Free |
| Lower Layer Architecture | Single (no evap), Single (evap), Tension+2 reservoirs |
| Evaporation | Sequential, Root-weighted |
| Percolation | Total storage, Free storage, Lower demand |
| Baseflow | Linear, Parallel linear, Nonlinear, TOPMODEL |
| Surface Runoff | Linear, Pareto/VIC, TOPMODEL gamma |
| Interflow | None, Linear |

### Parent Model Configurations

- **VIC**: Variable Infiltration Capacity model
- **TOPMODEL**: Topographic model
- **Sacramento**: NWS Sacramento Soil Moisture Accounting
- **PRMS**: Precipitation-Runoff Modeling System

## Installation

### Requirements

- C++17 compiler (GCC 9+, Clang 10+)
- CMake 3.18+
- (Optional) CUDA Toolkit 11.0+ for GPU support
- (Optional) pybind11 for Python bindings
- (Optional) Enzyme for automatic differentiation

### Build from Source

```bash
git clone https://github.com/your-org/dfuse.git
cd dfuse
mkdir build && cd build

# Basic CPU build (header-only, no dependencies)
cmake .. -DDFUSE_USE_CUDA=OFF
make -j

# Full build with all features
cmake .. \
    -DDFUSE_USE_CUDA=ON \
    -DDFUSE_USE_SUNDIALS=ON \
    -DDFUSE_USE_ENZYME=ON \
    -DDFUSE_BUILD_PYTHON=ON \
    -DDFUSE_BUILD_TESTS=ON
make -j
```

### Quick Test (Header-only, no build required)

```bash
# Basic functionality test
g++ -std=c++17 -O2 -I include -o test_cpu tests/test_cpu.cpp -lm
./test_cpu

# Comprehensive validation suite (17 tests)
g++ -std=c++17 -O2 -I include -o test_validation tests/test_fortran_validation.cpp -lm
./test_validation
```

## Quick Start

### C++ API

```cpp
#include <dfuse/dfuse.hpp>

using namespace dfuse;

// Select model configuration
ModelConfig config = models::VIC;

// Initialize state
State state;
state.S1 = 150.0;  // Upper layer (mm)
state.S2 = 800.0;  // Lower layer (mm)
state.SWE = 0.0;   // Snow (mm)
state.sync_derived(config);

// Set parameters
Parameters params;
params.S1_max = 400.0;
params.S2_max = 1500.0;
params.ku = 12.0;
params.ks = 40.0;
// ... other parameters
params.compute_derived();

// Create forcing
Forcing forcing(10.0, 4.0, 15.0);  // precip, PET, temp

// Run single timestep
Flux flux;
fuse_step(state, forcing, params, config, 1.0, flux);

printf("Runoff: %.2f mm/day\n", flux.q_total);
```

### Python API

```python
from dfuse import FUSE, create_model, VIC_CONFIG
import torch

# Create model
model = create_model('vic', dt=1.0, learnable_params=True)

# Generate forcing [n_timesteps, 3] = (precip, pet, temp)
forcing = torch.randn(365, 3).abs()
forcing[:, 2] = 15 + 10 * torch.sin(torch.linspace(0, 2*3.14, 365))

# Initial state
state = model.get_initial_state(S1_init=150.0, S2_init=800.0)

# Forward pass
runoff = model(state, forcing)

# Backward pass for parameter optimization
loss = ((runoff - observed) ** 2).mean()
loss.backward()
```

### Parameter Optimization

```python
import torch.optim as optim

model = create_model('vic', learnable_params=True)
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    
    pred = model(initial_state, forcing)
    loss = nse_loss(pred, observed)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}: NSE = {1 - loss.item():.4f}")
```

### High-Performance CUDA Training

For maximum GPU training performance, use the workspace-based batch execution to avoid `cudaMalloc`/`cudaFree` overhead:

```python
from dfuse import FUSE, create_workspace, VIC_CONFIG
import torch

# Create model and workspace (allocate GPU memory once)
model = FUSE(config=VIC_CONFIG, learnable_params=True)
workspace = create_workspace(
    n_hru=1000,        # Number of basins
    n_states=10,       # State variables per basin  
    n_timesteps=365,   # Simulation length
    device='cuda'
)

# Training loop - no GPU memory allocation inside!
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    
    # Use optimized batch forward (reuses workspace)
    runoff = model.forward_batch_optimized(
        initial_states,   # [n_hru, n_states]
        forcing,          # [n_timesteps, 3] shared
        workspace
    )
    
    loss = ((runoff - observed) ** 2).mean()
    loss.backward()
    optimizer.step()

print(f"Workspace size: {workspace}")  # e.g., CUDAWorkspace(n_hru=1000, size=14.5KB)
```

This eliminates ~50% of GPU wait time compared to the standard `forward_batch(use_cuda=True)`.

## Architecture

```
dfuse/
├── include/dfuse/
│   ├── config.hpp      # Physics configuration and enums
│   ├── state.hpp       # State, flux, parameter structures
│   ├── physics.hpp     # Flux computation functions
│   ├── solver.hpp      # ODE solvers (Euler, SUNDIALS)
│   ├── routing.hpp     # Gamma distribution routing
│   ├── enzyme_ad.hpp   # Enzyme automatic differentiation
│   ├── kernels.hpp     # GPU/CPU kernels
│   └── dfuse.hpp       # Main include header
├── python/
│   ├── dfuse.py        # PyTorch interface
│   └── bindings.cpp    # pybind11 bindings
├── tests/
│   ├── test_cpu.cpp              # Basic CPU tests
│   └── test_fortran_validation.cpp  # Comprehensive validation suite
└── examples/
    ├── single_basin.cpp          # C++ example
    └── parameter_optimization.py  # Python optimization example
```

## Solver Options

| Solver | Method | Use Case |
|--------|--------|----------|
| EXPLICIT_EULER | Forward Euler with adaptive substepping | Fast, simple problems |
| IMPLICIT_EULER | Backward Euler with Newton iteration | Stiff systems |
| SUNDIALS_BDF | CVODE BDF (1-5 order) | Stiff ODEs, high accuracy |
| SUNDIALS_ADAMS | CVODE Adams-Moulton | Non-stiff, smooth solutions |

```cpp
#include <dfuse/solver.hpp>

solver::SolverConfig config;
config.method = solver::SolverMethod::SUNDIALS_BDF;
config.rel_tol = 1e-6;
config.abs_tol = 1e-8;

solver::Solver solver(config);
solver.solve(state, forcing, params, model_config, dt, flux);
```

## Routing

Gamma distribution unit hydrograph following Clark et al. (2008) equation 13:

```cpp
#include <dfuse/routing.hpp>

// Generate unit hydrograph
Real shape = 2.5;      // Shape parameter
Real mean_delay = 1.5; // Mean delay (days)
Real dt = 1.0;         // Timestep (days)

std::vector<Real> uh;
int uh_length = routing::generate_unit_hydrograph(shape, mean_delay, dt, uh);

// Create routing buffer and apply
routing::RoutingBuffer router(uh.data(), uh_length);
Real routed_runoff = router.route(instant_runoff);
```

## Gradient Computation

### Numerical Gradients (always available)

```cpp
#include <dfuse/enzyme_ad.hpp>

std::vector<Real> grad_params(enzyme::NUM_PARAM_VARS);
enzyme::compute_loss_gradient_numerical(
    state_flat, forcing_data, params_flat, config_flat,
    observed, n_timesteps, grad_params.data()
);
```

### Enzyme AD (exact gradients, requires Enzyme)

```cpp
// Compile with -DDFUSE_USE_ENZYME=ON and Enzyme LLVM plugin
enzyme::compute_loss_gradient_enzyme(
    state_flat, forcing_data, params_flat, config_flat,
    observed, n_timesteps, grad_params.data()
);
```

### Adjoint Method (efficient for long time series)

```cpp
enzyme::AdjointSolver adjoint(n_timesteps);

// Forward pass (stores trajectory)
adjoint.forward(state_flat, forcing_data, params_flat, config_flat, dt);

// Backward pass (computes gradients)
adjoint.backward(observed, grad_params.data());
```

## Design Principles

### Differentiability

All physics computations use smooth approximations for discontinuities (following Kavetski & Kuczera 2007):

```cpp
// Logistic smoothing for bucket overflow
Real logistic_overflow(Real S, Real S_max, Real w) {
    Real x = (S - S_max - w * 5) / w;
    return 1.0 / (1.0 + exp(-x));
}
```

### GPU Efficiency

- **Struct-of-Arrays (SoA)** layout for coalesced memory access
- Template metaprogramming for compile-time physics selection
- Minimal branching in hot paths

### Enzyme Integration

For exact automatic differentiation:

```cpp
// Forward function
void fuse_step_enzyme(Real* state, const Real* forcing, 
                      const Real* params, Real dt, Real* flux);

// Enzyme generates gradient function automatically
__enzyme_autodiff(fuse_step_enzyme, 
    enzyme_dup, state, d_state,
    enzyme_const, forcing,
    enzyme_dup, params, d_params, ...);
```

## Performance

Preliminary benchmarks on NVIDIA A100:

| Configuration | HRUs | Timesteps | Time (ms) | Throughput |
|---------------|------|-----------|-----------|------------|
| VIC (GPU) | 10,000 | 365 | 45 | 81M steps/s |
| VIC (CPU) | 10,000 | 365 | 2,100 | 1.7M steps/s |
| Gradient (GPU) | 10,000 | 365 | 120 | 30M steps/s |

## Integration with SYMFLUENCE

dFUSE is designed to integrate with the [SYMFLUENCE](https://github.com/CH-Earth/CWARHM) hydrological modeling platform:

```python
# Chain with dMC-Route for end-to-end differentiable routing
from dfuse import FUSE
from dmc_route import DMCRoute

class DifferentiableHydrology(nn.Module):
    def __init__(self):
        self.rainfall_runoff = FUSE(config=VIC_CONFIG)
        self.routing = DMCRoute(method='muskingum_cunge')
    
    def forward(self, forcing, network):
        # Local runoff generation
        runoff = self.rainfall_runoff(forcing)
        
        # Route through river network
        streamflow = self.routing(runoff, network)
        
        return streamflow
```

## References

- Clark, M. P., et al. (2008). Framework for Understanding Structural Errors (FUSE): A modular framework to diagnose differences between hydrological models. Water Resources Research, 44(12). [doi:10.1029/2007WR006735](http://dx.doi.org/10.1029/2007WR006735)

- Henn, B., et al. (2015). An assessment of differences in gridded precipitation datasets in complex terrain. Journal of Hydrology, 530, 167-180.

- Kavetski, D., & Kuczera, G. (2007). Model smoothing strategies to remove microscale discontinuities and spurious secondary optima in objective functions in hydrological calibration. Water Resources Research, 43(3).

## License

GNU General Public License v3.0 (same as original FUSE)

## Citation

If you use dFUSE in your research, please cite:

```bibtex
@article{clark2008fuse,
  title={Framework for Understanding Structural Errors (FUSE): 
         A modular framework to diagnose differences between 
         hydrological models},
  author={Clark, Martyn P and Slater, Andrew G and Rupp, David E and 
          Woods, Ross A and Vrugt, Jasper A and Gupta, Hoshin V and 
          Wagener, Thorsten and Hay, Lauren E},
  journal={Water Resources Research},
  volume={44},
  number={12},
  year={2008},
  publisher={Wiley Online Library},
  doi={10.1029/2007WR006735}
}
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Original FUSE implementation by Martyn Clark and collaborators at NCAR
- Clark et al. (2008) for the modular modeling framework design
- Enzyme AD project for automatic differentiation support
