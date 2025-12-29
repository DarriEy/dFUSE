/**
 * @file kernels.hpp
 * @brief GPU/CPU kernels for cFUSE model execution
 * 
 * Updates:
 * - Added robust checks in state updates to prevent NaNs.
 * - Ensured flux limiting logic is numerically stable.
 * - Exposed ALL fluxes in fuse_step_enzyme for full gradient coverage.
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
     
     // Prevent division by zero in dt
     if (dt < Real(1e-6)) dt = Real(1e-6);
 
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
     // 3. BASEFLOW
     // ========================================================================
     compute_baseflow(state, params, config, flux.qb, flux.qb_A, flux.qb_B);
     
     // ========================================================================
     // 4. FREE STORAGE COMPUTATION (needed for percolation and interflow)
     // ========================================================================
     if (config.upper_arch == UpperLayerArch::SINGLE_STATE) {
         Real excess = state.S1 - params.S1_T_max;
         state.S1_F = smooth_max(excess, Real(0));
     }
     
     // ========================================================================
     // 5. PERCOLATION
     // ========================================================================
     Real q0 = (config.baseflow == BaseflowType::NONLINEAR) ? 
               params.ks : flux.qb;
     flux.q12 = compute_percolation(state, params, config, q0);
     
     // ========================================================================
     // 6. INTERFLOW
     // ========================================================================
     flux.qif = compute_interflow(state, params, config, dt);
     
     // ========================================================================
     // 7. EVAPORATION
     // ========================================================================
     compute_evaporation(forcing.pet, state, params, config, flux.e1, flux.e2);
     
     if (config.upper_arch == UpperLayerArch::TENSION2_FREE) {
         Real total_tension = state.S1_TA + state.S1_TB;
         // Safe division for AD
         if (total_tension > Real(1e-6)) {
             flux.e1_A = flux.e1 * (state.S1_TA / total_tension);
             flux.e1_B = flux.e1 * (state.S1_TB / total_tension);
         } else {
             flux.e1_A = flux.e1 * Real(0.5); // Fallback split
             flux.e1_B = flux.e1 * Real(0.5);
         }
     } else {
         flux.e1_A = flux.e1;
         flux.e1_B = Real(0);
     }
     
     // ========================================================================
     // 8. OVERFLOW FLUXES
     // ========================================================================
     compute_overflow(infiltration, flux.q12, state, params, config,
                      flux.qurof, flux.qutof, flux.qufof,
                      flux.qstof, flux.qsfof, flux.qsfofa, flux.qsfofb);
     
     // ========================================================================
     // 9. STATE UPDATE (Explicit Euler)
     // ========================================================================
     Real dSdt[MAX_TOTAL_STATES];
     compute_derivatives(state, flux, params, config, dSdt);
     
     int idx = 0;
     
     auto update_state = [&](Real& s, Real rate) {
         // Apply rate
         s += rate * dt;
         // Soft clamp to ensure non-negativity without hard discontinuity
         s = smooth_max(s, Real(0)); 
     };
     
     // Update upper layer
     switch (config.upper_arch) {
         case UpperLayerArch::SINGLE_STATE:
             update_state(state.S1, dSdt[idx++]);
             break;
         case UpperLayerArch::TENSION_FREE:
             update_state(state.S1_T, dSdt[idx++]);
             update_state(state.S1_F, dSdt[idx++]);
             break;
         case UpperLayerArch::TENSION2_FREE:
             update_state(state.S1_TA, dSdt[idx++]);
             update_state(state.S1_TB, dSdt[idx++]);
             update_state(state.S1_F, dSdt[idx++]);
             break;
     }
     
     // Update lower layer
     switch (config.lower_arch) {
         case LowerLayerArch::SINGLE_NOEVAP:
         case LowerLayerArch::SINGLE_EVAP:
             update_state(state.S2, dSdt[idx++]);
             break;
         case LowerLayerArch::TENSION_2RESERV:
             update_state(state.S2_T, dSdt[idx++]);
             update_state(state.S2_FA, dSdt[idx++]);
             update_state(state.S2_FB, dSdt[idx++]);
             break;
     }
     
     // Update snow
     if (config.enable_snow) {
         state.SWE = SWE_new;
     }
     
     // Sync derived state variables
     state.sync_derived(config);
     
     // ========================================================================
     // 10. COMPUTE TOTAL RUNOFF
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
     constexpr ModelConfig config = Config::to_runtime();
     fuse_step(state, forcing, params, config, dt, flux);
 }
 
 // ============================================================================
 // FORWARD KERNEL - Batch execution over HRUs/time
 // ============================================================================
 
 #ifdef __CUDACC__
 
 __global__ void fuse_forward_kernel(
     const StateBatch states_in,
     StateBatch states_out,
     const ForcingBatch forcing,
     const ParameterBatch params,
     const ModelConfig config,
     int n_hru,
     int n_timesteps,
     Real dt,
     FluxBatch* fluxes,
     Real* runoff
 ) {
     int hru_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (hru_idx >= n_hru) return;
     
     State state = states_in.get(hru_idx);
     Parameters hru_params = params.get(hru_idx);
     
     Flux flux;
     
     for (int t = 0; t < n_timesteps; ++t) {
         Forcing f = forcing.get(hru_idx, t);
         fuse_step(state, f, hru_params, config, dt, flux);
         
         int out_idx = t * n_hru + hru_idx;
         runoff[out_idx] = flux.q_total;
         
         if (fluxes != nullptr) {
             fluxes->set(out_idx, flux);
         }
     }
     
     states_out.set(hru_idx, state);
 }
 
 #endif // __CUDACC__
 
 // ============================================================================
 // CPU FORWARD FUNCTION
 // ============================================================================
 
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
 
 DFUSE_DEVICE inline void fuse_step_enzyme(
     Real* state_arr,           // [num_states] in/out
     const Real* forcing_arr,   // [3]: precip, pet, temp
     const Real* param_arr,     // [num_params]
     const int* config_arr,     // [8]: encoded configuration
     Real dt,
     Real* flux_arr             // [num_fluxes] out
 ) {
     ModelConfig config;
     config.upper_arch = static_cast<UpperLayerArch>(config_arr[0]);
     config.lower_arch = static_cast<LowerLayerArch>(config_arr[1]);
     config.evaporation = static_cast<EvaporationType>(config_arr[2]);
     config.percolation = static_cast<PercolationType>(config_arr[3]);
     config.interflow = static_cast<InterflowType>(config_arr[4]);
     config.baseflow = static_cast<BaseflowType>(config_arr[5]);
     config.surface_runoff = static_cast<SurfaceRunoffType>(config_arr[6]);
     config.enable_snow = (config_arr[7] != 0);
     
     State state;
     state.from_array(state_arr, config);
     
     Parameters params;
     params.from_array(param_arr);
     
     Forcing forcing(forcing_arr[0], forcing_arr[1], forcing_arr[2]);
     
     Flux flux;
     fuse_step(state, forcing, params, config, dt, flux);
     
     state.to_array(state_arr, config);
     
     // Store ALL fluxes for comprehensive gradient support
     // Maps to Flux struct in state.hpp
     flux_arr[0] = flux.q_total;
     flux_arr[1] = flux.e_total;
     flux_arr[2] = flux.qsx;
     flux_arr[3] = flux.qb;
     flux_arr[4] = flux.q12;
     flux_arr[5] = flux.e1;
     flux_arr[6] = flux.e2;
     flux_arr[7] = flux.qif;
     flux_arr[8] = flux.rain;
     flux_arr[9] = flux.melt;
     flux_arr[10] = flux.Ac;
     flux_arr[11] = flux.qufof;
     flux_arr[12] = flux.qsfof;
     flux_arr[13] = flux.throughfall;
     // Additional internal fluxes if needed
     // flux_arr[14] = flux.qb_A;
     // flux_arr[15] = flux.qb_B;
 }
 
 #ifdef __ENZYME__
 extern "C" {
     void __enzyme_autodiff(void*, ...);
 }
 #endif
 
 } // namespace dfuse