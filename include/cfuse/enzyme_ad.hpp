/**
 * @file enzyme_ad.hpp
 * @brief Enzyme Automatic Differentiation integration for cFUSE
 */

 #pragma once

 #include "config.hpp"
 #include "state.hpp"
 #include "physics.hpp"
 #include "routing.hpp" 
 #include <cstring>
 #include <array>
 #include <vector>
 #include <algorithm>
 
 namespace dfuse {
 
 // Forward declaration of the core kernel (defined in kernels.hpp)
 DFUSE_HOST_DEVICE void fuse_step(
     State& state,
     const Forcing& forcing,
     const Parameters& params,
     const ModelConfig& config,
     Real dt,
     Flux& flux
 );
 
 namespace enzyme {
 
 // ============================================================================
 // ENZYME DECLARATIONS
 // ============================================================================
 
 #ifdef DFUSE_USE_ENZYME
 extern "C" {
     void __enzyme_autodiff(void*, ...);
     int enzyme_dup;
     int enzyme_dupnoneed;
     int enzyme_const;
     int enzyme_out;
 }
 #define ENZYME_DUP enzyme_dup,
 #define ENZYME_CONST enzyme_const,
 #define ENZYME_OUT enzyme_out,
 #else
 #define ENZYME_DUP
 #define ENZYME_CONST
 #define ENZYME_OUT
 #endif
 
 // ============================================================================
 // FLAT ARRAY INTERFACE
 // ============================================================================
 
 constexpr int NUM_STATE_VARS = 10;    // S1, S1_T, S1_TA, S1_TB, S1_F, S2, S2_T, S2_FA, S2_FB, SWE
 constexpr int NUM_FLUX_VARS = 18;     
 constexpr int NUM_PARAM_VARS = 29;    
 constexpr int NUM_FORCING_VARS = 3;   
 constexpr int NUM_CONFIG_VARS = 8;    
 
 // New constants for Banded model
 constexpr int NUM_SOIL_VARS = 9;      // Soil states only (No SWE)
 constexpr int MAX_BANDS = 30;         // Max supported elevation bands
 constexpr int TOTAL_VARS_BANDS = NUM_SOIL_VARS + MAX_BANDS;
 
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
     arr[23] = params.T_melt;
     arr[24] = params.melt_rate; 
     arr[25] = params.lapse_rate;
     arr[26] = params.opg;
     arr[27] = params.MFMAX;
     arr[28] = params.MFMIN;
 }
 
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
     params.T_melt = arr[23];
     params.melt_rate = arr[24];
     params.lapse_rate = arr[25];
     params.opg = arr[26];
     params.MFMAX = arr[27];
     params.MFMIN = arr[28];
 }
 
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
 // ENZYME-COMPATIBLE FORWARD FUNCTIONS
 // ============================================================================
 
 // 1. Single Cell Kernel (Legacy/Simple)
 inline void fuse_step_flat(
     const Real* state_in,
     const Real* forcing,
     const Real* params,
     const int* config_arr,
     Real dt,
     Real* state_out,
     Real* runoff
 ) {
     State state;
     unpack_state(state_in, state);
     
     Parameters p;
     unpack_params(params, p);
     p.compute_derived();
     
     ModelConfig config;
     unpack_config(config_arr, config);
     
     Forcing f(forcing[0], forcing[1], forcing[2]);
     Flux flux;
     
     // Call Core Kernel
     fuse_step(state, f, p, config, dt, flux);
     
     pack_state(state, state_out);
     *runoff = flux.q_total;
 }
 
 // 2. Banded Distributed Kernel (For Global/Regional Optimization)
 inline void fuse_step_bands_flat(
     const Real* state_in,      // [NUM_SOIL_VARS + MAX_BANDS]
     const Real* forcing,       // [precip, pet, temp] at ref elev
     const Real* params_arr,
     const int* config_arr,
     Real dt,
     const Real* band_props,    // [MAX_BANDS * 2] -> area_frac, mean_elev
     int n_bands,
     Real ref_elev,
     Real* state_out,
     Real* runoff
 ) {
     // 1. Unpack Soil State (Manually to avoid SWE confusion)
     // S1...S2_FB are first 9 vars
     State state;
     state.S1 = state_in[0]; state.S1_T = state_in[1]; state.S1_TA = state_in[2];
     state.S1_TB = state_in[3]; state.S1_F = state_in[4];
     state.S2 = state_in[5]; state.S2_T = state_in[6];
     state.S2_FA = state_in[7]; state.S2_FB = state_in[8];
     state.SWE = 0; // Handled by bands
 
     // 2. Unpack Parameters & Config
     Parameters p; unpack_params(params_arr, p); p.compute_derived();
     ModelConfig config; unpack_config(config_arr, config);
     
     // 3. Snow Bands Logic
     Real rain_eff, melt_eff;
     Real swe_new[MAX_BANDS];
     
     const Real* area_frac = band_props;
     const Real* mean_elev = band_props + MAX_BANDS;
     const Real* swe_current = state_in + NUM_SOIL_VARS; // SWE starts after soil
     
     // Compute distributed snow
     physics::compute_snow_elevation_bands<MAX_BANDS>(
         forcing[0], forcing[2], swe_current, n_bands, 
         area_frac, mean_elev, ref_elev, p, 
         rain_eff, melt_eff, swe_new, 0 // day 0 for constant melt rate assumption in AD
     );
     
     // 4. Soil Step (Using Aggregated Inputs)
     ModelConfig soil_config = config;
     soil_config.enable_snow = false; // Disable internal snow
     
     Forcing f_soil(rain_eff + melt_eff, forcing[1], forcing[2]);
     Flux flux;
     
     fuse_step(state, f_soil, p, soil_config, dt, flux);
     
     // 5. Pack Output
     state_out[0] = state.S1; state_out[1] = state.S1_T; state_out[2] = state.S1_TA;
     state_out[3] = state.S1_TB; state_out[4] = state.S1_F;
     state_out[5] = state.S2; state_out[6] = state.S2_T;
     state_out[7] = state.S2_FA; state_out[8] = state.S2_FB;
     
     // Copy new SWEs back
     for(int b=0; b<n_bands; ++b) state_out[NUM_SOIL_VARS + b] = swe_new[b];
     // Zero out unused bands
     for(int b=n_bands; b<MAX_BANDS; ++b) state_out[NUM_SOIL_VARS + b] = 0;
     
     *runoff = flux.q_total;
 }
 
 // ============================================================================
 // LEGACY GRADIENT HELPERS (Required for C++ Tests)
 // ============================================================================
 
 inline Real compute_mse_loss(
     const Real* initial_state,
     const Real* forcing_series,
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
     
     // Base loss
     Real loss_base = compute_mse_loss(initial_state, forcing_series, params,
                                        config_arr, dt, observed_runoff, n_timesteps);
     
     for (int i = 0; i < NUM_PARAM_VARS; ++i) {
         Real orig = params_pert[i];
         Real h = eps * std::max(std::abs(orig), Real(1));
         
         // Forward difference
         params_pert[i] = orig + h;
         Real loss_plus = compute_mse_loss(initial_state, forcing_series, params_pert,
                                            config_arr, dt, observed_runoff, n_timesteps);
         
         grad_params[i] = (loss_plus - loss_base) / h;
         params_pert[i] = orig;
     }
 }
 
 #ifdef DFUSE_USE_ENZYME
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
     std::memset(grad_params, 0, NUM_PARAM_VARS * sizeof(Real));
     
     __enzyme_autodiff(
         (void*)compute_mse_loss,
         enzyme_const, initial_state,
         enzyme_const, forcing_series,
         enzyme_dup,   params, grad_params,
         enzyme_const, config_arr,
         enzyme_const, dt,
         enzyme_const, observed_runoff,
         enzyme_const, n_timesteps
     );
 }
 #endif
 
 // Helper for Jacobian tests
 inline void compute_state_jacobian(
     const Real* state_in,
     const Real* forcing,
     const Real* params,
     const int* config_arr,
     Real dt,
     Real* jacobian,
     Real eps = Real(1e-6)
 ) {
     Real state_pert[NUM_STATE_VARS];
     Real state_out_base[NUM_STATE_VARS];
     Real state_out_pert[NUM_STATE_VARS];
     Real runoff;
     
     fuse_step_flat(state_in, forcing, params, config_arr, dt, state_out_base, &runoff);
     
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
 
 // Legacy Adjoint Solver class (Stub for tests)
 class AdjointSolver {
 public:
     AdjointSolver(int n_timesteps, const int* config_arr) {}
 };
 
 } // namespace enzyme
 } // namespace dfuse