/**
 * @file enzyme_ad.hpp
 * @brief Enzyme Automatic Differentiation integration for dFUSE
 * 
 * Provides exact gradients through Enzyme AD compiler plugin.
 * Enzyme transforms LLVM IR to compute derivatives automatically.
 * 
 * Build requirements:
 *   - Clang with Enzyme plugin: clang++ -fplugin=/path/to/ClangEnzyme-XX.so
 *   - Or use enzyme-wrapped clang from Enzyme releases
 * 
 * References:
 *   - Moses & Churavy (2020) "Instead of Rewriting Foreign Code for ML"
 *   - https://enzyme.mit.edu/
 */

#pragma once

#include "config.hpp"
#include "state.hpp"
#include "physics.hpp"
#include <cstring>
#include <array>
#include <vector>

namespace dfuse {
namespace enzyme {

// ============================================================================
// ENZYME DECLARATIONS
// ============================================================================

#ifdef DFUSE_USE_ENZYME

// Enzyme external function declarations
extern "C" {
    // Core Enzyme differentiation function
    void __enzyme_autodiff(void*, ...);
    
    // Enzyme activity annotations
    int enzyme_dup;      // Duplicated (active) argument
    int enzyme_dupnoneed; // Active but don't need shadow
    int enzyme_const;    // Constant (inactive) argument
    int enzyme_out;      // Output-only argument
}

// Activity annotation helpers
#define ENZYME_DUP enzyme_dup,
#define ENZYME_CONST enzyme_const,
#define ENZYME_OUT enzyme_out,

#else

// Stub definitions when Enzyme not available
#define ENZYME_DUP
#define ENZYME_CONST
#define ENZYME_OUT

#endif // DFUSE_USE_ENZYME

// ============================================================================
// FLAT ARRAY INTERFACE FOR ENZYME
// ============================================================================

/**
 * @brief Array sizes for flat interface
 * 
 * Enzyme works best with simple arrays, so we flatten all structures.
 */
constexpr int NUM_STATE_VARS = 10;    // S1, S1_T, S1_TA, S1_TB, S1_F, S2, S2_T, S2_FA, S2_FB, SWE
constexpr int NUM_FLUX_VARS = 18;     // All flux components
constexpr int NUM_PARAM_VARS = 24;    // All parameters
constexpr int NUM_FORCING_VARS = 3;   // precip, pet, temp
constexpr int NUM_CONFIG_VARS = 8;    // Model configuration flags

/**
 * @brief Pack State into flat array
 */
inline void pack_state(const State& state, Real* arr) {
    arr[0] = state.S1;
    arr[1] = state.S1_T;
    arr[2] = state.S1_TA;
    arr[3] = state.S1_TB;
    arr[4] = state.S1_F;
    arr[5] = state.S2;
    arr[6] = state.S2_T;
    arr[7] = state.S2_FA;
    arr[8] = state.S2_FB;
    arr[9] = state.SWE;
}

/**
 * @brief Unpack flat array to State
 */
inline void unpack_state(const Real* arr, State& state) {
    state.S1 = arr[0];
    state.S1_T = arr[1];
    state.S1_TA = arr[2];
    state.S1_TB = arr[3];
    state.S1_F = arr[4];
    state.S2 = arr[5];
    state.S2_T = arr[6];
    state.S2_FA = arr[7];
    state.S2_FB = arr[8];
    state.SWE = arr[9];
}

/**
 * @brief Pack Parameters into flat array
 */
inline void pack_params(const Parameters& params, Real* arr) {
    arr[0] = params.S1_max;
    arr[1] = params.S2_max;
    arr[2] = params.f_tens;
    arr[3] = params.f_rchr;
    arr[4] = params.f_base;
    arr[5] = params.r1;
    arr[6] = params.ku;
    arr[7] = params.c;
    arr[8] = params.alpha;
    arr[9] = params.psi;
    arr[10] = params.kappa;
    arr[11] = params.ki;
    arr[12] = params.ks;
    arr[13] = params.n;
    arr[14] = params.v;
    arr[15] = params.v_A;
    arr[16] = params.v_B;
    arr[17] = params.Ac_max;
    arr[18] = params.b;
    arr[19] = params.lambda;
    arr[20] = params.chi;
    arr[21] = params.mu_t;
    arr[22] = params.T_rain;
    arr[23] = params.melt_rate;
}

/**
 * @brief Unpack flat array to Parameters
 */
inline void unpack_params(const Real* arr, Parameters& params) {
    params.S1_max = arr[0];
    params.S2_max = arr[1];
    params.f_tens = arr[2];
    params.f_rchr = arr[3];
    params.f_base = arr[4];
    params.r1 = arr[5];
    params.ku = arr[6];
    params.c = arr[7];
    params.alpha = arr[8];
    params.psi = arr[9];
    params.kappa = arr[10];
    params.ki = arr[11];
    params.ks = arr[12];
    params.n = arr[13];
    params.v = arr[14];
    params.v_A = arr[15];
    params.v_B = arr[16];
    params.Ac_max = arr[17];
    params.b = arr[18];
    params.lambda = arr[19];
    params.chi = arr[20];
    params.mu_t = arr[21];
    params.T_rain = arr[22];
    params.melt_rate = arr[23];
}

/**
 * @brief Pack ModelConfig into integer array
 */
inline void pack_config(const ModelConfig& config, int* arr) {
    arr[0] = static_cast<int>(config.upper_arch);
    arr[1] = static_cast<int>(config.lower_arch);
    arr[2] = static_cast<int>(config.evaporation);
    arr[3] = static_cast<int>(config.percolation);
    arr[4] = static_cast<int>(config.interflow);
    arr[5] = static_cast<int>(config.baseflow);
    arr[6] = static_cast<int>(config.surface_runoff);
    arr[7] = config.enable_snow ? 1 : 0;
}

/**
 * @brief Unpack integer array to ModelConfig
 */
inline void unpack_config(const int* arr, ModelConfig& config) {
    config.upper_arch = static_cast<UpperLayerArch>(arr[0]);
    config.lower_arch = static_cast<LowerLayerArch>(arr[1]);
    config.evaporation = static_cast<EvaporationType>(arr[2]);
    config.percolation = static_cast<PercolationType>(arr[3]);
    config.interflow = static_cast<InterflowType>(arr[4]);
    config.baseflow = static_cast<BaseflowType>(arr[5]);
    config.surface_runoff = static_cast<SurfaceRunoffType>(arr[6]);
    config.enable_snow = arr[7] != 0;
}

// ============================================================================
// ENZYME-COMPATIBLE FORWARD FUNCTION
// ============================================================================

/**
 * @brief Single timestep forward function with flat arrays
 * 
 * This is the core function that Enzyme will differentiate.
 * All inputs/outputs are flat arrays for maximum compatibility.
 * 
 * @param state_in Input state [NUM_STATE_VARS]
 * @param forcing Forcing data [NUM_FORCING_VARS]: {precip, pet, temp}
 * @param params Parameters [NUM_PARAM_VARS]
 * @param config_arr Config flags [NUM_CONFIG_VARS]
 * @param dt Time step
 * @param state_out Output state [NUM_STATE_VARS]
 * @param runoff Output total runoff (scalar)
 */
inline void fuse_step_flat(
    const Real* state_in,
    const Real* forcing,
    const Real* params,
    const int* config_arr,
    Real dt,
    Real* state_out,
    Real* runoff
) {
    // Unpack inputs
    State state;
    unpack_state(state_in, state);
    
    Parameters p;
    unpack_params(params, p);
    p.compute_derived();
    
    ModelConfig config;
    unpack_config(config_arr, config);
    
    Real precip = forcing[0];
    Real pet = forcing[1];
    Real temp = forcing[2];
    
    // ========== SNOW MODULE ==========
    Real rain, melt, throughfall;
    if (config.enable_snow) {
        Real snow_frac = physics::smooth_sigmoid(p.T_rain - temp, Real(1));
        Real snow = precip * snow_frac;
        rain = precip * (Real(1) - snow_frac);
        
        state.SWE += snow * dt;
        melt = p.melt_rate * physics::smooth_max(temp - p.T_rain, Real(0), Real(0.5));
        melt = physics::smooth_min(melt, state.SWE / dt, Real(0.1));
        state.SWE -= melt * dt;
        throughfall = rain + melt;
    } else {
        rain = precip;
        melt = Real(0);
        throughfall = precip;
        state.SWE = Real(0);
    }
    
    // ========== SURFACE RUNOFF ==========
    Real Ac, qsx;
    switch (config.surface_runoff) {
        case SurfaceRunoffType::UZ_LINEAR: {
            Real S1_T_frac = state.S1_T / (p.S1_T_max + Real(1e-10));
            Ac = S1_T_frac * p.Ac_max;
            break;
        }
        case SurfaceRunoffType::UZ_PARETO: {
            Real S1_frac = state.S1 / (p.S1_max + Real(1e-10));
            S1_frac = physics::smooth_clamp(S1_frac, Real(0), Real(1), Real(0.01));
            Ac = Real(1) - std::pow(Real(1) - S1_frac, p.b);
            break;
        }
        case SurfaceRunoffType::LZ_GAMMA:
        default: {
            Real S2_frac = state.S2 / (p.S2_max + Real(1e-10));
            Ac = physics::smooth_clamp(S2_frac, Real(0), Real(1), Real(0.01));
            break;
        }
    }
    Ac = physics::smooth_clamp(Ac, Real(0), Real(1), Real(0.01));
    qsx = Ac * throughfall;
    Real infiltration = throughfall - qsx;
    
    // ========== BASEFLOW ==========
    Real qb = Real(0), qb_A = Real(0), qb_B = Real(0);
    switch (config.baseflow) {
        case BaseflowType::LINEAR:
            qb = p.v * state.S2;
            break;
        case BaseflowType::PARALLEL_LINEAR:
            qb_A = p.v_A * state.S2_FA;
            qb_B = p.v_B * state.S2_FB;
            qb = qb_A + qb_B;
            break;
        case BaseflowType::NONLINEAR: {
            Real S2_frac = state.S2 / (p.S2_max + Real(1e-10));
            S2_frac = physics::smooth_clamp(S2_frac, Real(0), Real(1), Real(0.01));
            qb = p.ks * std::pow(S2_frac + Real(1e-10), p.n);
            break;
        }
        case BaseflowType::TOPMODEL: {
            Real S2_frac = state.S2 / (p.S2_max + Real(1e-10));
            S2_frac = physics::smooth_clamp(S2_frac, Real(0), Real(1), Real(0.01));
            qb = p.ks * p.m / std::pow(p.lambda_n + Real(1e-10), p.n) * 
                 std::pow(S2_frac * p.S2_max + Real(1e-10), p.n);
            break;
        }
    }
    
    // ========== PERCOLATION ==========
    Real q12 = Real(0);
    switch (config.percolation) {
        case PercolationType::TOTAL_STORAGE: {
            Real S1_frac = state.S1 / (p.S1_max + Real(1e-10));
            S1_frac = physics::smooth_clamp(S1_frac, Real(0), Real(1), Real(0.01));
            q12 = p.ku * std::pow(S1_frac + Real(1e-10), p.c);
            break;
        }
        case PercolationType::FREE_STORAGE: {
            Real S1_F_frac = state.S1_F / (p.S1_F_max + Real(1e-10));
            S1_F_frac = physics::smooth_clamp(S1_F_frac, Real(0), Real(1), Real(0.01));
            q12 = p.ku * std::pow(S1_F_frac + Real(1e-10), p.c);
            break;
        }
        case PercolationType::LOWER_DEMAND: {
            Real S1_F_frac = state.S1_F / (p.S1_F_max + Real(1e-10));
            Real S2_frac = state.S2 / (p.S2_max + Real(1e-10));
            S1_F_frac = physics::smooth_clamp(S1_F_frac, Real(0), Real(1), Real(0.01));
            S2_frac = physics::smooth_clamp(S2_frac, Real(0), Real(1), Real(0.01));
            Real dlz = Real(1) + p.alpha * std::pow(S2_frac + Real(1e-10), p.psi);
            q12 = qb * dlz * S1_F_frac;
            break;
        }
    }
    
    // ========== INTERFLOW ==========
    Real qif = Real(0);
    if (config.interflow == InterflowType::LINEAR) {
        Real S1_F_frac = state.S1_F / (p.S1_F_max + Real(1e-10));
        S1_F_frac = physics::smooth_clamp(S1_F_frac, Real(0), Real(1), Real(0.01));
        qif = p.ki * S1_F_frac;
    }
    
    // ========== EVAPORATION ==========
    Real e1 = Real(0), e2 = Real(0);
    Real S1_T_frac = state.S1_T / (p.S1_T_max + Real(1e-10));
    Real S2_T_frac = state.S2_T / (p.S2_T_max + Real(1e-10));
    S1_T_frac = physics::smooth_clamp(S1_T_frac, Real(0), Real(1), Real(0.01));
    S2_T_frac = physics::smooth_clamp(S2_T_frac, Real(0), Real(1), Real(0.01));
    
    switch (config.evaporation) {
        case EvaporationType::SEQUENTIAL:
            e1 = pet * S1_T_frac;
            e2 = (pet - e1) * S2_T_frac;
            break;
        case EvaporationType::ROOT_WEIGHT:
            e1 = pet * p.r1 * S1_T_frac;
            e2 = pet * (Real(1) - p.r1) * S2_T_frac;
            break;
    }
    
    // ========== OVERFLOW (Smooth) ==========
    Real w = Real(0.1);  // Smoothing width
    
    // Upper layer overflow
    Real qufof = Real(0), qutof = Real(0), qurof = Real(0);
    switch (config.upper_arch) {
        case UpperLayerArch::SINGLE_STATE:
            qufof = physics::smooth_overflow(state.S1, p.S1_max, infiltration - e1 - q12 - qif, w);
            break;
        case UpperLayerArch::TENSION_FREE:
            qutof = physics::smooth_overflow(state.S1_T, p.S1_T_max, infiltration - e1, w);
            qufof = physics::smooth_overflow(state.S1_F, p.S1_F_max, qutof - q12 - qif, w);
            break;
        case UpperLayerArch::TENSION2_FREE:
            qurof = physics::smooth_overflow(state.S1_TA, p.S1_TA_max, infiltration - e1 * p.r1, w);
            qutof = physics::smooth_overflow(state.S1_TB, p.S1_TB_max, qurof - e1 * (Real(1) - p.r1), w);
            qufof = physics::smooth_overflow(state.S1_F, p.S1_F_max, qutof - q12 - qif, w);
            break;
    }
    
    // Lower layer overflow
    Real qsfof = Real(0), qstof = Real(0), qsfofa = Real(0), qsfofb = Real(0);
    switch (config.lower_arch) {
        case LowerLayerArch::SINGLE_NOEVAP:
            qsfof = physics::smooth_overflow(state.S2, p.S2_max, q12 - qb, w);
            break;
        case LowerLayerArch::SINGLE_EVAP:
            qsfof = physics::smooth_overflow(state.S2, p.S2_max, q12 - qb - e2, w);
            break;
        case LowerLayerArch::TENSION_2RESERV:
            qstof = physics::smooth_overflow(state.S2_T, p.S2_T_max, p.kappa * q12 - e2, w);
            Real to_free = (Real(1) - p.kappa) * q12 / Real(2) + qstof / Real(2);
            qsfofa = physics::smooth_overflow(state.S2_FA, p.S2_FA_max, to_free - qb_A, w);
            qsfofb = physics::smooth_overflow(state.S2_FB, p.S2_FB_max, to_free - qb_B, w);
            qsfof = qsfofa + qsfofb;
            break;
    }
    
    // ========== STATE UPDATE ==========
    switch (config.upper_arch) {
        case UpperLayerArch::SINGLE_STATE:
            state.S1 += (infiltration - e1 - q12 - qif - qufof) * dt;
            state.S1_T = state.S1 * p.f_tens;
            state.S1_F = state.S1 * (Real(1) - p.f_tens);
            break;
        case UpperLayerArch::TENSION_FREE:
            state.S1_T += (infiltration - e1 - qutof) * dt;
            state.S1_F += (qutof - q12 - qif - qufof) * dt;
            state.S1 = state.S1_T + state.S1_F;
            break;
        case UpperLayerArch::TENSION2_FREE:
            state.S1_TA += (infiltration - e1 * p.r1 - qurof) * dt;
            state.S1_TB += (qurof - e1 * (Real(1) - p.r1) - qutof) * dt;
            state.S1_F += (qutof - q12 - qif - qufof) * dt;
            state.S1_T = state.S1_TA + state.S1_TB;
            state.S1 = state.S1_T + state.S1_F;
            break;
    }
    
    switch (config.lower_arch) {
        case LowerLayerArch::SINGLE_NOEVAP:
            state.S2 += (q12 - qb - qsfof) * dt;
            state.S2_T = state.S2 * p.f_rchr;
            break;
        case LowerLayerArch::SINGLE_EVAP:
            state.S2 += (q12 - qb - e2 - qsfof) * dt;
            state.S2_T = state.S2 * p.f_rchr;
            break;
        case LowerLayerArch::TENSION_2RESERV: {
            state.S2_T += (p.kappa * q12 - e2 - qstof) * dt;
            Real to_free = (Real(1) - p.kappa) * q12 / Real(2) + qstof / Real(2);
            state.S2_FA += (to_free - qb_A - qsfofa) * dt;
            state.S2_FB += (to_free - qb_B - qsfofb) * dt;
            state.S2 = state.S2_T + state.S2_FA + state.S2_FB;
            break;
        }
    }
    
    // Enforce non-negativity (smooth)
    state.S1 = physics::smooth_max(state.S1, Real(0), Real(0.01));
    state.S1_T = physics::smooth_max(state.S1_T, Real(0), Real(0.01));
    state.S1_TA = physics::smooth_max(state.S1_TA, Real(0), Real(0.01));
    state.S1_TB = physics::smooth_max(state.S1_TB, Real(0), Real(0.01));
    state.S1_F = physics::smooth_max(state.S1_F, Real(0), Real(0.01));
    state.S2 = physics::smooth_max(state.S2, Real(0), Real(0.01));
    state.S2_T = physics::smooth_max(state.S2_T, Real(0), Real(0.01));
    state.S2_FA = physics::smooth_max(state.S2_FA, Real(0), Real(0.01));
    state.S2_FB = physics::smooth_max(state.S2_FB, Real(0), Real(0.01));
    state.SWE = physics::smooth_max(state.SWE, Real(0), Real(0.01));
    
    // Pack outputs
    pack_state(state, state_out);
    *runoff = qsx + qb + qif + qufof + qsfof;
}

// ============================================================================
// LOSS FUNCTION FOR ENZYME
// ============================================================================

/**
 * @brief Compute MSE loss for a time series
 * 
 * This is the function we'll differentiate to get parameter gradients.
 */
inline Real compute_mse_loss(
    const Real* initial_state,
    const Real* forcing_series,  // [n_timesteps * 3]
    const Real* params,
    const int* config_arr,
    Real dt,
    const Real* observed_runoff,
    int n_timesteps
) {
    Real state[NUM_STATE_VARS];
    Real state_new[NUM_STATE_VARS];
    std::memcpy(state, initial_state, NUM_STATE_VARS * sizeof(Real));
    
    Real loss = Real(0);
    
    for (int t = 0; t < n_timesteps; ++t) {
        const Real* forcing = &forcing_series[t * NUM_FORCING_VARS];
        Real runoff;
        
        fuse_step_flat(state, forcing, params, config_arr, dt, state_new, &runoff);
        
        Real diff = runoff - observed_runoff[t];
        loss += diff * diff;
        
        std::memcpy(state, state_new, NUM_STATE_VARS * sizeof(Real));
    }
    
    return loss / n_timesteps;
}

// ============================================================================
// ENZYME GRADIENT COMPUTATION
// ============================================================================

#ifdef DFUSE_USE_ENZYME

/**
 * @brief Compute gradient of loss w.r.t. parameters using Enzyme
 * 
 * @param initial_state Initial state [NUM_STATE_VARS]
 * @param forcing_series Forcing [n_timesteps * 3]
 * @param params Parameters [NUM_PARAM_VARS]
 * @param config_arr Config [NUM_CONFIG_VARS]
 * @param dt Time step
 * @param observed_runoff Observations [n_timesteps]
 * @param n_timesteps Number of timesteps
 * @param grad_params Output gradients [NUM_PARAM_VARS]
 */
inline void compute_loss_gradient_enzyme(
    const Real* initial_state,
    const Real* forcing_series,
    const Real* params,
    const int* config_arr,
    Real dt,
    const Real* observed_runoff,
    int n_timesteps,
    Real* grad_params
) {
    // Zero gradient output
    std::memset(grad_params, 0, NUM_PARAM_VARS * sizeof(Real));
    
    // Call Enzyme autodiff
    // Arguments: function pointer, then pairs of (activity, value, shadow) for each arg
    __enzyme_autodiff(
        (void*)compute_mse_loss,
        enzyme_const, initial_state,           // Don't diff w.r.t. initial state
        enzyme_const, forcing_series,          // Don't diff w.r.t. forcing
        enzyme_dup, params, grad_params,       // Differentiate w.r.t. params
        enzyme_const, config_arr,              // Config is integer, always const
        enzyme_const, dt,                      // dt is constant
        enzyme_const, observed_runoff,         // Don't diff w.r.t. observations
        enzyme_const, n_timesteps              // Constant
    );
}

/**
 * @brief Compute gradient of single step output w.r.t. inputs
 */
inline void compute_step_gradient_enzyme(
    const Real* state_in,
    const Real* forcing,
    const Real* params,
    const int* config_arr,
    Real dt,
    Real* grad_state_in,    // [NUM_STATE_VARS]
    Real* grad_params       // [NUM_PARAM_VARS]
) {
    Real state_out[NUM_STATE_VARS];
    Real grad_state_out[NUM_STATE_VARS];
    Real runoff;
    Real grad_runoff = Real(1);  // Seed gradient
    
    // Zero gradients
    std::memset(grad_state_in, 0, NUM_STATE_VARS * sizeof(Real));
    std::memset(grad_state_out, 0, NUM_STATE_VARS * sizeof(Real));
    std::memset(grad_params, 0, NUM_PARAM_VARS * sizeof(Real));
    
    __enzyme_autodiff(
        (void*)fuse_step_flat,
        enzyme_dup, state_in, grad_state_in,
        enzyme_const, forcing,
        enzyme_dup, params, grad_params,
        enzyme_const, config_arr,
        enzyme_const, dt,
        enzyme_dup, state_out, grad_state_out,
        enzyme_dup, &runoff, &grad_runoff
    );
}

#endif // DFUSE_USE_ENZYME

// ============================================================================
// NUMERICAL GRADIENT (Fallback when Enzyme not available)
// ============================================================================

/**
 * @brief Compute gradient numerically via finite differences
 * 
 * Used as fallback or for validation of Enzyme gradients.
 */
inline void compute_loss_gradient_numerical(
    const Real* initial_state,
    const Real* forcing_series,
    const Real* params,
    const int* config_arr,
    Real dt,
    const Real* observed_runoff,
    int n_timesteps,
    Real* grad_params,
    Real eps = Real(1e-5)
) {
    Real params_pert[NUM_PARAM_VARS];
    std::memcpy(params_pert, params, NUM_PARAM_VARS * sizeof(Real));
    
    Real loss_base = compute_mse_loss(initial_state, forcing_series, params,
                                       config_arr, dt, observed_runoff, n_timesteps);
    
    for (int i = 0; i < NUM_PARAM_VARS; ++i) {
        Real orig = params_pert[i];
        Real h = eps * std::max(std::abs(orig), Real(1));
        
        // Forward difference
        params_pert[i] = orig + h;
        Real loss_plus = compute_mse_loss(initial_state, forcing_series, params_pert,
                                           config_arr, dt, observed_runoff, n_timesteps);
        
        // Central difference for better accuracy
        params_pert[i] = orig - h;
        Real loss_minus = compute_mse_loss(initial_state, forcing_series, params_pert,
                                            config_arr, dt, observed_runoff, n_timesteps);
        
        grad_params[i] = (loss_plus - loss_minus) / (Real(2) * h);
        params_pert[i] = orig;
    }
}

/**
 * @brief Unified gradient computation interface
 * 
 * Uses Enzyme if available, falls back to numerical.
 */
inline void compute_loss_gradient(
    const Real* initial_state,
    const Real* forcing_series,
    const Real* params,
    const int* config_arr,
    Real dt,
    const Real* observed_runoff,
    int n_timesteps,
    Real* grad_params
) {
#ifdef DFUSE_USE_ENZYME
    compute_loss_gradient_enzyme(initial_state, forcing_series, params,
                                  config_arr, dt, observed_runoff, n_timesteps,
                                  grad_params);
#else
    compute_loss_gradient_numerical(initial_state, forcing_series, params,
                                     config_arr, dt, observed_runoff, n_timesteps,
                                     grad_params);
#endif
}

// ============================================================================
// JACOBIAN COMPUTATION
// ============================================================================

/**
 * @brief Compute Jacobian of state transition
 * 
 * J[i,j] = d(state_out[i]) / d(state_in[j])
 */
inline void compute_state_jacobian(
    const Real* state_in,
    const Real* forcing,
    const Real* params,
    const int* config_arr,
    Real dt,
    Real* jacobian,  // [NUM_STATE_VARS * NUM_STATE_VARS]
    Real eps = Real(1e-6)
) {
    Real state_base[NUM_STATE_VARS];
    Real state_pert[NUM_STATE_VARS];
    Real state_out_base[NUM_STATE_VARS];
    Real state_out_pert[NUM_STATE_VARS];
    Real runoff;
    
    std::memcpy(state_base, state_in, NUM_STATE_VARS * sizeof(Real));
    fuse_step_flat(state_base, forcing, params, config_arr, dt, state_out_base, &runoff);
    
    for (int j = 0; j < NUM_STATE_VARS; ++j) {
        std::memcpy(state_pert, state_in, NUM_STATE_VARS * sizeof(Real));
        Real h = eps * std::max(std::abs(state_pert[j]), Real(1));
        state_pert[j] += h;
        
        fuse_step_flat(state_pert, forcing, params, config_arr, dt, state_out_pert, &runoff);
        
        for (int i = 0; i < NUM_STATE_VARS; ++i) {
            jacobian[i * NUM_STATE_VARS + j] = (state_out_pert[i] - state_out_base[i]) / h;
        }
    }
}

/**
 * @brief Compute sensitivity of runoff to parameters
 */
inline void compute_runoff_sensitivity(
    const Real* state_in,
    const Real* forcing,
    const Real* params,
    const int* config_arr,
    Real dt,
    Real* sensitivity,  // [NUM_PARAM_VARS]
    Real eps = Real(1e-6)
) {
    Real params_pert[NUM_PARAM_VARS];
    Real state_out[NUM_STATE_VARS];
    Real runoff_base, runoff_pert;
    
    std::memcpy(params_pert, params, NUM_PARAM_VARS * sizeof(Real));
    fuse_step_flat(state_in, forcing, params, config_arr, dt, state_out, &runoff_base);
    
    for (int i = 0; i < NUM_PARAM_VARS; ++i) {
        Real orig = params_pert[i];
        Real h = eps * std::max(std::abs(orig), Real(1));
        params_pert[i] = orig + h;
        
        fuse_step_flat(state_in, forcing, params_pert, config_arr, dt, state_out, &runoff_pert);
        sensitivity[i] = (runoff_pert - runoff_base) / h;
        
        params_pert[i] = orig;
    }
}

// ============================================================================
// ADJOINT METHOD FOR LONG TIME SERIES
// ============================================================================

/**
 * @brief Adjoint state method for efficient gradient computation
 * 
 * For long time series, adjoint method is more efficient than
 * forward-mode AD: O(n_params) vs O(n_timesteps * n_params).
 */
class AdjointSolver {
public:
    AdjointSolver(int n_timesteps, const int* config_arr)
        : n_timesteps_(n_timesteps) {
        std::memcpy(config_arr_, config_arr, NUM_CONFIG_VARS * sizeof(int));
        
        // Allocate storage for forward trajectory
        states_.resize(n_timesteps_ + 1);
        for (auto& s : states_) s.resize(NUM_STATE_VARS);
    }
    
    /**
     * @brief Run forward pass and store trajectory
     */
    Real forward(
        const Real* initial_state,
        const Real* forcing_series,
        const Real* params,
        Real dt,
        const Real* observed_runoff,
        Real* predicted_runoff = nullptr
    ) {
        std::memcpy(states_[0].data(), initial_state, NUM_STATE_VARS * sizeof(Real));
        
        Real loss = Real(0);
        runoffs_.resize(n_timesteps_);
        
        for (int t = 0; t < n_timesteps_; ++t) {
            const Real* forcing = &forcing_series[t * NUM_FORCING_VARS];
            fuse_step_flat(states_[t].data(), forcing, params, config_arr_, dt,
                          states_[t + 1].data(), &runoffs_[t]);
            
            Real diff = runoffs_[t] - observed_runoff[t];
            loss += diff * diff;
            
            if (predicted_runoff) {
                predicted_runoff[t] = runoffs_[t];
            }
        }
        
        return loss / n_timesteps_;
    }
    
    /**
     * @brief Run backward pass to compute gradients
     */
    void backward(
        const Real* forcing_series,
        const Real* params,
        Real dt,
        const Real* observed_runoff,
        Real* grad_params
    ) {
        std::memset(grad_params, 0, NUM_PARAM_VARS * sizeof(Real));
        
        // Initialize adjoint state
        std::vector<Real> adjoint(NUM_STATE_VARS, Real(0));
        std::vector<Real> adjoint_new(NUM_STATE_VARS);
        
        // Backward through time
        for (int t = n_timesteps_ - 1; t >= 0; --t) {
            const Real* forcing = &forcing_series[t * NUM_FORCING_VARS];
            
            // Loss gradient: d(loss)/d(runoff[t]) = 2*(runoff[t] - obs[t]) / n
            Real dloss_drunoff = Real(2) * (runoffs_[t] - observed_runoff[t]) / n_timesteps_;
            
            // Compute local gradients
            Real jacobian[NUM_STATE_VARS * NUM_STATE_VARS];
            compute_state_jacobian(states_[t].data(), forcing, params, config_arr_, dt, jacobian);
            
            Real sensitivity[NUM_PARAM_VARS];
            compute_runoff_sensitivity(states_[t].data(), forcing, params, config_arr_, dt, sensitivity);
            
            // Accumulate parameter gradients
            for (int i = 0; i < NUM_PARAM_VARS; ++i) {
                grad_params[i] += dloss_drunoff * sensitivity[i];
            }
            
            // Propagate adjoint: adjoint_prev = J^T * adjoint + d(runoff)/d(state) * dloss_drunoff
            std::fill(adjoint_new.begin(), adjoint_new.end(), Real(0));
            for (int i = 0; i < NUM_STATE_VARS; ++i) {
                for (int j = 0; j < NUM_STATE_VARS; ++j) {
                    adjoint_new[j] += jacobian[i * NUM_STATE_VARS + j] * adjoint[i];
                }
            }
            
            // Add contribution from runoff sensitivity to state
            Real runoff_state_sens[NUM_STATE_VARS];
            compute_runoff_state_sensitivity(states_[t].data(), forcing, params, 
                                             config_arr_, dt, runoff_state_sens);
            for (int j = 0; j < NUM_STATE_VARS; ++j) {
                adjoint_new[j] += runoff_state_sens[j] * dloss_drunoff;
            }
            
            adjoint = adjoint_new;
        }
    }
    
private:
    int n_timesteps_;
    int config_arr_[NUM_CONFIG_VARS];
    std::vector<std::vector<Real>> states_;
    std::vector<Real> runoffs_;
    
    void compute_runoff_state_sensitivity(
        const Real* state_in,
        const Real* forcing,
        const Real* params,
        const int* config_arr,
        Real dt,
        Real* sensitivity
    ) {
        Real state_pert[NUM_STATE_VARS];
        Real state_out[NUM_STATE_VARS];
        Real runoff_base, runoff_pert;
        const Real eps = Real(1e-6);
        
        fuse_step_flat(state_in, forcing, params, config_arr, dt, state_out, &runoff_base);
        
        for (int i = 0; i < NUM_STATE_VARS; ++i) {
            std::memcpy(state_pert, state_in, NUM_STATE_VARS * sizeof(Real));
            Real h = eps * std::max(std::abs(state_pert[i]), Real(1));
            state_pert[i] += h;
            
            fuse_step_flat(state_pert, forcing, params, config_arr, dt, state_out, &runoff_pert);
            sensitivity[i] = (runoff_pert - runoff_base) / h;
        }
    }
};

} // namespace enzyme
} // namespace dfuse
