/**
 * @file kernels.hpp
 * @brief GPU/CPU kernels for dFUSE model execution
 * 
 * Differentiable Framework for Understanding Structural Errors (dFUSE)
 * Based on Clark et al. (2008) WRR, doi:10.1029/2007WR006735
 * 
 * This file contains:
 * - fuse_step: Single timestep device function (the model core)
 * - fuse_forward_kernel: Global kernel for batch forward execution
 * - fuse_backward_kernel: Global kernel for gradient computation (Enzyme)
 * 
 * Design for differentiability:
 * - All operations use smooth approximations for discontinuities
 * - State updates are performed in-place for Enzyme compatibility
 * - Flux outputs are computed as side-effects for adjoint computation
 */

#pragma once

#include "config.hpp"
#include "state.hpp"
#include "physics.hpp"

namespace dfuse {

// ============================================================================
// SINGLE TIMESTEP DEVICE FUNCTION
// ============================================================================

/**
 * @brief Execute single timestep of FUSE model
 * 
 * This is the core model function that updates state variables and computes
 * fluxes for a single HRU and single timestep. It is designed to be:
 * - Callable from both CPU and GPU
 * - Differentiable via Enzyme automatic differentiation
 * - Composable for different physics configurations
 * 
 * @param state Input/output model state (modified in place)
 * @param forcing Meteorological forcing for this timestep
 * @param params Model parameters
 * @param config Model configuration (physics options)
 * @param dt Timestep length (days)
 * @param[out] flux Computed fluxes for this timestep
 */
DFUSE_DEVICE inline void fuse_step(
    State& state,
    const Forcing& forcing,
    const Parameters& params,
    const ModelConfig& config,
    Real dt,
    Flux& flux
) {
    using namespace physics;
    
    // ========================================================================
    // 1. SNOW MODULE
    // ========================================================================
    Real rain, melt, SWE_new;
    if (config.enable_snow) {
        compute_snow(forcing.precip, forcing.temp, state.SWE, params,
                     rain, melt, SWE_new);
        flux.rain = rain;
        flux.melt = melt;
        flux.throughfall = rain + melt;
    } else {
        flux.rain = forcing.precip;
        flux.melt = Real(0);
        flux.throughfall = forcing.precip;
    }
    
    // ========================================================================
    // 2. SURFACE RUNOFF (Saturated Area)
    // ========================================================================
    compute_surface_runoff(flux.throughfall, state, params, config,
                           flux.Ac, flux.qsx);
    
    Real infiltration = flux.throughfall - flux.qsx;
    
    // ========================================================================
    // 3. BASEFLOW (needed for Sacramento percolation)
    // ========================================================================
    compute_baseflow(state, params, config, flux.qb, flux.qb_A, flux.qb_B);
    
    // ========================================================================
    // 4. FREE STORAGE COMPUTATION (needed for percolation and interflow)
    // ========================================================================
    // For SINGLE_STATE architecture, compute "free" storage as excess above tension capacity
    // This is needed for both percolation (FREE_STORAGE type) and interflow
    if (config.upper_arch == UpperLayerArch::SINGLE_STATE) {
        Real excess = state.S1 - params.S1_T_max;
        state.S1_F = (excess > Real(0)) ? excess : Real(0);
    }
    
    // ========================================================================
    // 5. PERCOLATION
    // ========================================================================
    // For Sacramento, need baseflow at saturation (q0) for percolation demand
    Real q0 = (config.baseflow == BaseflowType::NONLINEAR) ? 
              params.ks : flux.qb;  // Approximate
    flux.q12 = compute_percolation(state, params, config, q0);
    
    // ========================================================================
    // 6. INTERFLOW
    // ========================================================================
    flux.qif = compute_interflow(state, params, config, dt);
    
    // ========================================================================
    // 6. EVAPORATION
    // ========================================================================
    compute_evaporation(forcing.pet, state, params, config, flux.e1, flux.e2);
    
    // For PRMS architecture with two tension stores
    if (config.upper_arch == UpperLayerArch::TENSION2_FREE) {
        // Split e1 between primary and secondary tension stores
        Real total_tension = state.S1_TA + state.S1_TB;
        if (total_tension > Real(1e-10)) {
            flux.e1_A = flux.e1 * (state.S1_TA / total_tension);
            flux.e1_B = flux.e1 * (state.S1_TB / total_tension);
        } else {
            flux.e1_A = Real(0);
            flux.e1_B = Real(0);
        }
    } else {
        flux.e1_A = flux.e1;
        flux.e1_B = Real(0);
    }
    
    // ========================================================================
    // 7. OVERFLOW FLUXES
    // ========================================================================
    compute_overflow(infiltration, flux.q12, state, params, config,
                     flux.qurof, flux.qutof, flux.qufof,
                     flux.qstof, flux.qsfof, flux.qsfofa, flux.qsfofb);
    
    // ========================================================================
    // 8. STATE UPDATE (Explicit Euler for now)
    // ========================================================================
    // More sophisticated integration (implicit, adaptive) in production
    
    Real dSdt[MAX_TOTAL_STATES];
    compute_derivatives(state, flux, params, config, dSdt);
    
    int idx = 0;
    
    // Update upper layer
    switch (config.upper_arch) {
        case UpperLayerArch::SINGLE_STATE:
            state.S1 = smooth_max(state.S1 + dSdt[idx++] * dt, Real(0));
            break;
        case UpperLayerArch::TENSION_FREE:
            state.S1_T = smooth_max(state.S1_T + dSdt[idx++] * dt, Real(0));
            state.S1_F = smooth_max(state.S1_F + dSdt[idx++] * dt, Real(0));
            break;
        case UpperLayerArch::TENSION2_FREE:
            state.S1_TA = smooth_max(state.S1_TA + dSdt[idx++] * dt, Real(0));
            state.S1_TB = smooth_max(state.S1_TB + dSdt[idx++] * dt, Real(0));
            state.S1_F = smooth_max(state.S1_F + dSdt[idx++] * dt, Real(0));
            break;
    }
    
    // Update lower layer
    switch (config.lower_arch) {
        case LowerLayerArch::SINGLE_NOEVAP:
        case LowerLayerArch::SINGLE_EVAP:
            state.S2 = smooth_max(state.S2 + dSdt[idx++] * dt, Real(0));
            break;
        case LowerLayerArch::TENSION_2RESERV:
            state.S2_T = smooth_max(state.S2_T + dSdt[idx++] * dt, Real(0));
            state.S2_FA = smooth_max(state.S2_FA + dSdt[idx++] * dt, Real(0));
            state.S2_FB = smooth_max(state.S2_FB + dSdt[idx++] * dt, Real(0));
            break;
    }
    
    // Update snow
    if (config.enable_snow) {
        state.SWE = SWE_new;
    }
    
    // Sync derived state variables
    state.sync_derived(config);
    
    // ========================================================================
    // 9. COMPUTE TOTAL RUNOFF
    // ========================================================================
    flux.e_total = flux.e1 + flux.e2;
    flux.q_total = flux.qsx + flux.qif + flux.qb + flux.qufof + flux.qsfof;
}

/**
 * @brief Template version for compile-time physics selection
 */
template<typename Config>
DFUSE_DEVICE inline void fuse_step_static(
    State& state,
    const Forcing& forcing,
    const Parameters& params,
    Real dt,
    Flux& flux
) {
    // Convert static config to runtime and call main function
    // In optimized builds, the compiler should inline and eliminate branches
    constexpr ModelConfig config = Config::to_runtime();
    fuse_step(state, forcing, params, config, dt, flux);
}

// ============================================================================
// FORWARD KERNEL - Batch execution over HRUs/time
// ============================================================================

#ifdef __CUDACC__

/**
 * @brief CUDA kernel for batch forward model execution
 * 
 * Executes the model for many HRUs in parallel, optionally over multiple
 * timesteps. Each thread handles one HRU.
 * 
 * @param states_in Initial states [n_hru]
 * @param states_out Final states [n_hru]
 * @param forcing Forcing data [n_timesteps x n_hru]
 * @param params Parameters [n_hru] or [1] if shared
 * @param config Model configuration
 * @param n_hru Number of HRUs
 * @param n_timesteps Number of timesteps
 * @param dt Timestep length (days)
 * @param fluxes Output fluxes [n_timesteps x n_hru] (optional)
 * @param runoff Output total runoff [n_timesteps x n_hru]
 */
__global__ void fuse_forward_kernel(
    const StateBatch states_in,
    StateBatch states_out,
    const ForcingBatch forcing,
    const ParameterBatch params,
    const ModelConfig config,
    int n_hru,
    int n_timesteps,
    Real dt,
    FluxBatch* fluxes,       // Optional, can be nullptr
    Real* runoff             // [n_timesteps x n_hru]
) {
    int hru_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hru_idx >= n_hru) return;
    
    // Load initial state for this HRU
    State state = states_in.get(hru_idx);
    Parameters hru_params = params.get(hru_idx);
    
    Flux flux;
    
    // Time loop
    for (int t = 0; t < n_timesteps; ++t) {
        // Get forcing for this timestep
        Forcing f = forcing.get(hru_idx, t);
        
        // Execute single timestep
        fuse_step(state, f, hru_params, config, dt, flux);
        
        // Store outputs
        int out_idx = t * n_hru + hru_idx;
        runoff[out_idx] = flux.q_total;
        
        if (fluxes != nullptr) {
            fluxes->set(out_idx, flux);
        }
    }
    
    // Store final state
    states_out.set(hru_idx, state);
}

/**
 * @brief Template kernel for compile-time physics selection
 */
template<typename Config>
__global__ void fuse_forward_kernel_static(
    const StateBatch states_in,
    StateBatch states_out,
    const ForcingBatch forcing,
    const ParameterBatch params,
    int n_hru,
    int n_timesteps,
    Real dt,
    Real* runoff
) {
    int hru_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hru_idx >= n_hru) return;
    
    State state = states_in.get(hru_idx);
    Parameters hru_params = params.get(hru_idx);
    constexpr ModelConfig config = Config::to_runtime();
    
    Flux flux;
    
    for (int t = 0; t < n_timesteps; ++t) {
        Forcing f = forcing.get(hru_idx, t);
        fuse_step(state, f, hru_params, config, dt, flux);
        runoff[t * n_hru + hru_idx] = flux.q_total;
    }
    
    states_out.set(hru_idx, state);
}

#endif // __CUDACC__

// ============================================================================
// CPU FORWARD FUNCTION
// ============================================================================

/**
 * @brief CPU version of forward model execution
 * 
 * Executes the model for a single HRU over multiple timesteps.
 */
inline void fuse_forward_cpu(
    State& state,
    const Forcing* forcing,
    const Parameters& params,
    const ModelConfig& config,
    int n_timesteps,
    Real dt,
    Real* runoff,
    Flux* fluxes = nullptr
) {
    Flux flux;
    
    for (int t = 0; t < n_timesteps; ++t) {
        fuse_step(state, forcing[t], params, config, dt, flux);
        runoff[t] = flux.q_total;
        
        if (fluxes != nullptr) {
            fluxes[t] = flux;
        }
    }
}

// ============================================================================
// GRADIENT COMPUTATION (Enzyme Integration)
// ============================================================================

/**
 * @brief Enzyme-compatible single step for differentiation
 * 
 * Simplified interface suitable for Enzyme AD.
 * Takes flat arrays instead of structs for easier gradient accumulation.
 */
DFUSE_DEVICE inline void fuse_step_enzyme(
    Real* state_arr,           // [num_states] in/out
    const Real* forcing_arr,   // [3]: precip, pet, temp
    const Real* param_arr,     // [num_params]
    const int* config_arr,     // [7]: encoded configuration
    Real dt,
    Real* flux_arr             // [num_fluxes] out
) {
    // Decode configuration
    ModelConfig config;
    config.upper_arch = static_cast<UpperLayerArch>(config_arr[0]);
    config.lower_arch = static_cast<LowerLayerArch>(config_arr[1]);
    config.baseflow = static_cast<BaseflowType>(config_arr[2]);
    config.percolation = static_cast<PercolationType>(config_arr[3]);
    config.surface_runoff = static_cast<SurfaceRunoffType>(config_arr[4]);
    config.evaporation = static_cast<EvaporationType>(config_arr[5]);
    config.interflow = static_cast<InterflowType>(config_arr[6]);
    
    // Reconstruct structs
    State state;
    state.from_array(state_arr, config);
    
    Parameters params;
    params.from_array(param_arr);
    
    Forcing forcing(forcing_arr[0], forcing_arr[1], forcing_arr[2]);
    
    Flux flux;
    
    // Execute step
    fuse_step(state, forcing, params, config, dt, flux);
    
    // Write back state
    state.to_array(state_arr, config);
    
    // Write fluxes
    flux_arr[0] = flux.q_total;
    flux_arr[1] = flux.e_total;
    flux_arr[2] = flux.qsx;
    flux_arr[3] = flux.qb;
    flux_arr[4] = flux.q12;
}

#ifdef __ENZYME__
// Enzyme-generated gradient function prototype
extern "C" {
    void __enzyme_autodiff(
        void (*)(Real*, const Real*, const Real*, const int*, Real, Real*),
        int,      // enzyme_dup for state
        Real*,    // state
        Real*,    // d_state
        int,      // enzyme_const for forcing
        const Real*,
        int,      // enzyme_dup for params  
        const Real*,
        Real*,    // d_params
        int,      // enzyme_const for config
        const int*,
        int,      // enzyme_const for dt
        Real,
        int,      // enzyme_dup for flux
        Real*,
        Real*     // d_flux (input for backward pass)
    );
}

/**
 * @brief Compute gradients using Enzyme AD
 * 
 * @param state_arr Model state
 * @param d_state_arr Gradient of loss w.r.t. state (output)
 * @param forcing_arr Forcing data
 * @param param_arr Parameters
 * @param d_param_arr Gradient of loss w.r.t. parameters (output)
 * @param config_arr Configuration
 * @param dt Timestep
 * @param flux_arr Flux outputs
 * @param d_flux_arr Gradient of loss w.r.t. fluxes (input)
 */
DFUSE_DEVICE inline void fuse_step_gradient(
    Real* state_arr,
    Real* d_state_arr,
    const Real* forcing_arr,
    const Real* param_arr,
    Real* d_param_arr,
    const int* config_arr,
    Real dt,
    Real* flux_arr,
    Real* d_flux_arr
) {
    __enzyme_autodiff(
        fuse_step_enzyme,
        enzyme_dup, state_arr, d_state_arr,
        enzyme_const, forcing_arr,
        enzyme_dup, param_arr, d_param_arr,
        enzyme_const, config_arr,
        enzyme_const, dt,
        enzyme_dup, flux_arr, d_flux_arr
    );
}
#endif // __ENZYME__

// ============================================================================
// GRADIENT KERNEL
// ============================================================================

#ifdef __CUDACC__

/**
 * @brief CUDA kernel for backward pass / gradient computation
 * 
 * Computes gradients of loss w.r.t. parameters via backpropagation through time.
 * 
 * @param states State trajectory [n_timesteps x n_hru]
 * @param forcing Forcing data [n_timesteps x n_hru]
 * @param params Parameters [n_hru]
 * @param config Model configuration
 * @param n_hru Number of HRUs
 * @param n_timesteps Number of timesteps
 * @param dt Timestep
 * @param d_loss_d_runoff Gradient of loss w.r.t runoff [n_timesteps x n_hru]
 * @param d_params Output parameter gradients [n_hru x n_params]
 */
__global__ void fuse_grad_kernel(
    const Real* states,           // Saved forward states
    const ForcingBatch forcing,
    const ParameterBatch params,
    const ModelConfig config,
    int n_hru,
    int n_timesteps,
    Real dt,
    const Real* d_loss_d_runoff,
    Real* d_params
) {
    int hru_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hru_idx >= n_hru) return;
    
    // Initialize parameter gradients to zero
    Real local_d_params[NUM_PARAMETERS] = {0};
    
    // Backward through time
    Real d_state[MAX_TOTAL_STATES] = {0};
    
    for (int t = n_timesteps - 1; t >= 0; --t) {
        int idx = t * n_hru + hru_idx;
        
        // Gradient from loss
        Real d_flux[NUM_FLUXES] = {0};
        d_flux[0] = d_loss_d_runoff[idx];  // d_loss/d_q_total
        
        // TODO: Call enzyme gradient function here
        // This requires careful state management for BPTT
        
        // Accumulate parameter gradients
        // local_d_params += current step gradients
    }
    
    // Write accumulated gradients
    for (int p = 0; p < NUM_PARAMETERS; ++p) {
        d_params[hru_idx * NUM_PARAMETERS + p] = local_d_params[p];
    }
}

#endif // __CUDACC__

} // namespace dfuse
