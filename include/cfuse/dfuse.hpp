/**
 * @file dfuse.hpp
 * @brief Main header for cFUSE - Differentiable FUSE hydrological model
 * 
 * Differentiable Framework for Understanding Structural Errors (cFUSE)
 * Based on Clark et al. (2008) WRR, doi:10.1029/2007WR006735
 * 
 * This is the main include file for cFUSE. Including this header provides
 * access to all model components.
 * 
 * Features:
 * - All 79 model configurations from Clark et al. (2008)
 * - GPU acceleration via CUDA
 * - Automatic differentiation via Enzyme
 * - SUNDIALS ODE solver integration
 * - Gamma distribution routing
 * 
 * @example Basic usage:
 * @code
 * #include <cfuse/dfuse.hpp>
 * 
 * using namespace dfuse;
 * 
 * // Create model configuration
 * ModelConfig config = VIC_CONFIG;
 * 
 * // Initialize state and parameters
 * State state;
 * state.S1 = 100.0;
 * state.S2 = 500.0;
 * 
 * Parameters params;
 * params.S1_max = 1000.0;
 * params.S2_max = 5000.0;
 * // ... set other parameters
 * params.compute_derived();
 * 
 * // Create forcing and solver
 * Forcing forcing{5.0, 3.0, 15.0};  // precip, pet, temp
 * solver::Solver solver;
 * 
 * // Run single timestep
 * Flux flux;
 * solver.solve(state, forcing, params, config, 1.0, flux);
 * 
 * printf("Runoff: %.2f mm/day\n", flux.total_runoff);
 * @endcode
 * 
 * @version 0.2.0
 */

#pragma once

#include <cstdio>

// Core configuration and enums
#include "config.hpp"

// State, flux, and parameter structures
#include "state.hpp"

// Physics computations
#include "physics.hpp"

// ODE Solvers (Euler, implicit, SUNDIALS)
#include "solver.hpp"

// Gamma distribution routing
#include "routing.hpp"

// Enzyme automatic differentiation
#include "enzyme_ad.hpp"

// GPU/CPU kernels (fuse_step is CPU-compatible)
#include "kernels.hpp"

namespace dfuse {

/**
 * @brief Library version
 */
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 2;
constexpr int VERSION_PATCH = 0;

/**
 * @brief Get version string
 */
inline const char* version() {
    static char version_str[32];
    snprintf(version_str, sizeof(version_str), "%d.%d.%d",
             VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    return version_str;
}

/**
 * @brief Check if CUDA is available
 */
inline bool cuda_available() {
#ifdef DFUSE_USE_CUDA
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if Enzyme AD is available
 */
inline bool enzyme_available() {
#ifdef DFUSE_USE_ENZYME
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if SUNDIALS is available
 */
inline bool sundials_available() {
#ifdef DFUSE_USE_SUNDIALS
    return true;
#else
    return false;
#endif
}

/**
 * @brief Print library info
 */
inline void print_info() {
    printf("cFUSE v%s - Differentiable FUSE Hydrological Model\n", version());
    printf("Based on Clark et al. (2008) WRR, doi:10.1029/2007WR006735\n");
    printf("\n");
    printf("Precision: %s\n", sizeof(Real) == 8 ? "double" : "float");
    printf("Max states: %d\n", MAX_TOTAL_STATES);
    printf("Num parameters: %d\n", NUM_PARAMETERS);
    printf("CUDA: %s\n", cuda_available() ? "enabled" : "disabled");
    printf("Enzyme AD: %s\n", enzyme_available() ? "enabled" : "disabled");
    printf("SUNDIALS: %s\n", sundials_available() ? "enabled" : "disabled");
}

} // namespace dfuse
