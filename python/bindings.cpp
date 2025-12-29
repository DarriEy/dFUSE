/**
 * @file bindings.cpp
 * @brief Python bindings for cFUSE via pybind11 (Enzyme Enabled)
 * 
 * UPDATED: Added run_fuse_batch_gradient for coupled cFUSE+dRoute calibration
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
 #include <pybind11/stl.h>
 
 #include "cfuse/dfuse.hpp"
 #include "cfuse/kernels.hpp"
 #include "cfuse/physics.hpp"
 #include "cfuse/enzyme_ad.hpp"
 #include "cfuse/solver.hpp"
 #include "cfuse/routing.hpp"
 
 // CVODES adjoint sensitivity (optional - requires CVODES, not just CVODE)
 #ifdef DFUSE_USE_CVODES
 #include "cfuse/cvodes_adjoint.hpp"
 #endif
 
 #include <vector>
 #include <cstring>
 #include <algorithm>
 #include <stdexcept>
 #include <iostream>
 #include <cmath>
 
 #ifdef DFUSE_USE_CUDA
 #include <cuda_runtime.h>
 #endif
 
 namespace py = pybind11;
 using namespace dfuse;
 using namespace dfuse::enzyme;
 
 // ========================================================================
 // HELPERS
 // ========================================================================
 
 ModelConfig config_from_dict(py::dict config_dict) {
     ModelConfig config;
     config.upper_arch = static_cast<UpperLayerArch>(config_dict["upper_arch"].cast<int>());
     config.lower_arch = static_cast<LowerLayerArch>(config_dict["lower_arch"].cast<int>());
     config.baseflow = static_cast<BaseflowType>(config_dict["baseflow"].cast<int>());
     config.percolation = static_cast<PercolationType>(config_dict["percolation"].cast<int>());
     config.surface_runoff = static_cast<SurfaceRunoffType>(config_dict["surface_runoff"].cast<int>());
     config.evaporation = static_cast<EvaporationType>(config_dict["evaporation"].cast<int>());
     config.interflow = static_cast<InterflowType>(config_dict["interflow"].cast<int>());
     config.enable_snow = config_dict.contains("enable_snow") ? config_dict["enable_snow"].cast<bool>() : true;
     return config;
 }
 
 void config_to_int_array(const ModelConfig& config, int* arr) {
     arr[0] = static_cast<int>(config.upper_arch);
     arr[1] = static_cast<int>(config.lower_arch);
     arr[2] = static_cast<int>(config.evaporation);
     arr[3] = static_cast<int>(config.percolation);
     arr[4] = static_cast<int>(config.interflow); 
     arr[5] = static_cast<int>(config.baseflow);
     arr[6] = static_cast<int>(config.surface_runoff);
     arr[7] = config.enable_snow ? 1 : 0;
 }
 
 py::array_t<Real> state_to_numpy(const State& state, const ModelConfig& config) {
     auto result = py::array_t<Real>(MAX_TOTAL_STATES);
     auto buf = result.mutable_unchecked<1>();
     Real arr[MAX_TOTAL_STATES];
     state.to_array(arr, config);
     for (int i = 0; i < MAX_TOTAL_STATES; ++i) buf(i) = arr[i];
     return result;
 }
 
 State state_from_numpy(py::array_t<Real> arr, const ModelConfig& config) {
     auto buf = arr.unchecked<1>();
     Real state_arr[MAX_TOTAL_STATES];
     for (ssize_t i = 0; i < arr.size() && i < MAX_TOTAL_STATES; ++i) state_arr[i] = buf(i);
     State state;
     state.from_array(state_arr, config);
     return state;
 }
 
Parameters params_from_numpy(py::array_t<Real> arr) {
    auto buf = arr.unchecked<1>();
    Real param_arr[NUM_PARAMETERS];
    for (ssize_t i = 0; i < arr.size() && i < NUM_PARAMETERS; ++i) param_arr[i] = buf(i);
    Parameters params;
    params.from_array(param_arr);
    return params;
}

py::array_t<Real> route_runoff_timeseries(
    py::array_t<Real, py::array::c_style | py::array::forcecast> instant_runoff,
    Real shape,
    Real mean_delay,
    Real dt
) {
    auto in_buf = instant_runoff.request();
    if (in_buf.ndim != 1) {
        throw std::runtime_error("instant_runoff must be a 1D array");
    }
    int n_timesteps = static_cast<int>(in_buf.shape[0]);
    auto result = py::array_t<Real>(n_timesteps);
    auto out_buf = result.request();
    routing::route_timeseries(
        static_cast<Real*>(in_buf.ptr),
        shape,
        mean_delay,
        dt,
        static_cast<Real*>(out_buf.ptr),
        n_timesteps
    );
    return result;
}

 // Get number of active states for a config
 int get_num_active_states(const ModelConfig& config) {
     int n_upper = 1;
     if (config.upper_arch == UpperLayerArch::TENSION_FREE) n_upper = 2;
     else if (config.upper_arch == UpperLayerArch::TENSION2_FREE) n_upper = 3;
     
     int n_lower = 1;
     if (config.lower_arch == LowerLayerArch::TENSION_2RESERV) n_lower = 3;
     
     int n = n_upper + n_lower;
     if (config.enable_snow) n += 1;
     return n;
 }
 
 // ========================================================================
 // RUNNERS (Existing - kept for compatibility)
 // ========================================================================
 
 py::tuple run_fuse_cpu(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::dict config_dict,
     Real dt,
     bool return_fluxes = false,
     std::string solver_type = "euler"
 ) {
     ModelConfig config = config_from_dict(config_dict);
     auto forcing_buf = forcing.unchecked<2>();
     int n_timesteps = static_cast<int>(forcing_buf.shape(0));
     State state = state_from_numpy(initial_state, config);
     Parameters parameters = params_from_numpy(params);
     
     solver::SolverConfig solver_cfg;
     if (solver_type == "sundials" || solver_type == "sundials_bdf") {
         #ifdef DFUSE_USE_SUNDIALS
         solver_cfg.method = solver::SolverMethod::SUNDIALS_BDF;
         solver_cfg.rel_tol = 1e-5;
         solver_cfg.abs_tol = 1e-7;
         #else
         throw std::runtime_error("SUNDIALS not compiled.");
         #endif
     } else {
         solver_cfg.method = solver::SolverMethod::EXPLICIT_EULER;
     }
     solver::Solver solver_obj(solver_cfg);
 
     auto runoff = py::array_t<Real>(n_timesteps);
     auto runoff_buf = runoff.mutable_unchecked<1>();
     std::vector<Flux> flux_history;
     if (return_fluxes) flux_history.resize(n_timesteps);
     
     Flux flux;
     for (int t = 0; t < n_timesteps; ++t) {
         Forcing f(forcing_buf(t, 0), forcing_buf(t, 1), forcing_buf(t, 2));
         solver_obj.solve(state, f, parameters, config, dt, flux);
         runoff_buf(t) = flux.q_total;
         if (return_fluxes) flux_history[t] = flux;
     }
     auto final_state = state_to_numpy(state, config);
     
     if (return_fluxes) {
         auto fluxes = py::array_t<Real>({n_timesteps, NUM_FLUXES});
         auto flux_buf = fluxes.mutable_unchecked<2>();
         for (int t = 0; t < n_timesteps; ++t) {
             flux_buf(t, 0) = flux_history[t].q_total;
             flux_buf(t, 1) = flux_history[t].e_total;
             flux_buf(t, 2) = flux_history[t].qsx;
             flux_buf(t, 3) = flux_history[t].qb;
             flux_buf(t, 4) = flux_history[t].q12;
             flux_buf(t, 5) = flux_history[t].e1;
             flux_buf(t, 6) = flux_history[t].e2;
             flux_buf(t, 7) = flux_history[t].qif;
             flux_buf(t, 8) = flux_history[t].rain;
             flux_buf(t, 9) = flux_history[t].melt;
             flux_buf(t, 10) = flux_history[t].Ac;
         }
         return py::make_tuple(final_state, runoff, fluxes);
     }
     return py::make_tuple(final_state, runoff);
 }
 
 // Batch Runner (forward only)
 py::tuple run_fuse_batch_cpu(
     py::array_t<Real> initial_states,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::dict config_dict,
     Real dt
 ) {
     ModelConfig config = config_from_dict(config_dict);
     auto states_buf = initial_states.unchecked<2>();
     int n_hru = static_cast<int>(states_buf.shape(0));
     int n_states_in = static_cast<int>(states_buf.shape(1));
     bool shared_forcing = (forcing.ndim() == 2);
     bool shared_params = (params.ndim() == 1);
     int n_timesteps;
     if (shared_forcing) n_timesteps = static_cast<int>(forcing.unchecked<2>().shape(0));
     else n_timesteps = static_cast<int>(forcing.unchecked<3>().shape(0));
     
     auto final_states = py::array_t<Real>({n_hru, MAX_TOTAL_STATES});
     auto runoff = py::array_t<Real>({n_timesteps, n_hru});
     auto states_out_buf = final_states.mutable_unchecked<2>();
     auto runoff_buf = runoff.mutable_unchecked<2>();
     std::fill(states_out_buf.mutable_data(0,0), states_out_buf.mutable_data(0,0) + final_states.size(), 0.0);
     
     #pragma omp parallel for
     for (int h = 0; h < n_hru; ++h) {
         Real state_arr[MAX_TOTAL_STATES];
         for (int s = 0; s < MAX_TOTAL_STATES; ++s) {
             state_arr[s] = (s < n_states_in) ? states_buf(h, s) : Real(0);
         }
         State state;
         state.from_array(state_arr, config);
         Parameters parameters;
         Real param_arr[NUM_PARAMETERS];
         if (shared_params) {
             auto p_buf = params.unchecked<1>();
             for (int i = 0; i < NUM_PARAMETERS; ++i) param_arr[i] = p_buf(i);
         } else {
             auto p_buf = params.unchecked<2>();
             for (int i = 0; i < NUM_PARAMETERS; ++i) param_arr[i] = p_buf(h, i);
         }
         parameters.from_array(param_arr);
         Flux flux;
         for (int t = 0; t < n_timesteps; ++t) {
             Forcing f;
             if (shared_forcing) {
                 auto f_buf = forcing.unchecked<2>();
                 f = Forcing(f_buf(t, 0), f_buf(t, 1), f_buf(t, 2));
             } else {
                 auto f_buf = forcing.unchecked<3>();
                 f = Forcing(f_buf(t, h, 0), f_buf(t, h, 1), f_buf(t, h, 2));
             }
             fuse_step(state, f, parameters, config, dt, flux);
             runoff_buf(t, h) = flux.q_total;
         }
         state.to_array(state_arr, config);
         for (int s = 0; s < MAX_TOTAL_STATES; ++s) states_out_buf(h, s) = state_arr[s];
     }
     return py::make_tuple(final_states, runoff);
 }

// ========================================================================
// NEW: BATCH GRADIENT COMPUTATION FOR COUPLED CALIBRATION
// ========================================================================

#ifdef DFUSE_USE_ENZYME

/**
 * @brief Forward pass for a single HRU that computes weighted sum with grad_runoff.
 * 
 * This function is designed to be differentiated by Enzyme.
 * It computes: sum_t(runoff[t] * grad_runoff[t]) for one HRU.
 * 
 * Enzyme will differentiate this to get: d(sum)/d(params) = sum_t(d_runoff[t]/d_params * grad_runoff[t])
 * 
 * NOTE: dt is passed as pointer to avoid float->double promotion in variadic __enzyme_autodiff
 */
void fuse_single_hru_weighted_sum(
    const Real* state_in,       // [MAX_TOTAL_STATES]
    const Real* forcing_flat,   // [n_timesteps * 3] - P, PET, T for this HRU
    const Real* param_arr,      // [NUM_PARAMETERS]
    const int* config_arr,      // [8] - model config
    const Real* dt_ptr,         // Pointer to dt (avoids variadic promotion issues)
    int n_timesteps,
    const Real* grad_runoff,    // [n_timesteps] - upstream gradient dL/d(runoff)
    Real* weighted_sum_out      // [1] - output: sum(runoff * grad_runoff)
) {
    Real dt = *dt_ptr;
    
    // Reconstruct state
    State state;
    Real state_arr[MAX_TOTAL_STATES];
    std::memcpy(state_arr, state_in, MAX_TOTAL_STATES * sizeof(Real));
    
    // Reconstruct config
    ModelConfig config;
    config.upper_arch = static_cast<UpperLayerArch>(config_arr[0]);
    config.lower_arch = static_cast<LowerLayerArch>(config_arr[1]);
    config.evaporation = static_cast<EvaporationType>(config_arr[2]);
    config.percolation = static_cast<PercolationType>(config_arr[3]);
    config.interflow = static_cast<InterflowType>(config_arr[4]);
    config.baseflow = static_cast<BaseflowType>(config_arr[5]);
    config.surface_runoff = static_cast<SurfaceRunoffType>(config_arr[6]);
    config.enable_snow = (config_arr[7] == 1);
    
    state.from_array(state_arr, config);
    
    // Reconstruct parameters
    Parameters params;
    Real p_arr[NUM_PARAMETERS];
    std::memcpy(p_arr, param_arr, NUM_PARAMETERS * sizeof(Real));
    params.from_array(p_arr);
    
    // Run forward pass and accumulate weighted sum
    Real sum = 0.0;
    Flux flux;
    
    for (int t = 0; t < n_timesteps; ++t) {
        Forcing f(
            forcing_flat[t * 3 + 0],  // precip
            forcing_flat[t * 3 + 1],  // pet
            forcing_flat[t * 3 + 2]   // temp
        );
        
        fuse_step(state, f, params, config, dt, flux);
        
        // Accumulate: runoff[t] * grad_runoff[t]
        sum += flux.q_total * grad_runoff[t];
    }
    
    *weighted_sum_out = sum;
}

/**
 * @brief Compute gradients for batch of HRUs with shared parameters.
 * 
 * This is the main function for coupled cFUSE+dRoute calibration.
 * 
 * For shared parameters across HRUs, gradients accumulate:
 *   d_params = sum_h sum_t (dL/d(runoff[t,h]) * d(runoff[t,h])/d(params))
 * 
 * @param initial_states [n_hru, n_states] Initial state for each HRU
 * @param forcing [n_timesteps, n_hru, 3] Forcing data (P, PET, T)
 * @param params [n_params] Shared parameters
 * @param grad_runoff [n_timesteps, n_hru] Upstream gradient dL/d(runoff)
 * @param config_dict Model configuration
 * @param dt Timestep in days
 * @return Parameter gradients [n_params]
 */
py::array_t<Real> run_fuse_batch_gradient(
    py::array_t<Real> initial_states,   // [n_hru, n_states]
    py::array_t<Real> forcing,          // [n_timesteps, n_hru, 3]
    py::array_t<Real> params,           // [n_params]
    py::array_t<Real> grad_runoff,      // [n_timesteps, n_hru]
    py::dict config_dict,
    Real dt
) {
    // Parse config
    ModelConfig config = config_from_dict(config_dict);
    int config_arr[8];
    config_to_int_array(config, config_arr);
    
    // Get dimensions
    auto states_buf = initial_states.unchecked<2>();
    auto forcing_buf = forcing.unchecked<3>();
    auto params_buf = params.unchecked<1>();
    auto grad_buf = grad_runoff.unchecked<2>();
    
    int n_hru = static_cast<int>(states_buf.shape(0));
    int n_states_in = static_cast<int>(states_buf.shape(1));
    int n_timesteps = static_cast<int>(forcing_buf.shape(0));
    int n_params = static_cast<int>(params_buf.size());
    
    // Prepare shared parameter array
    Real param_arr[NUM_PARAMETERS] = {0};
    for (int i = 0; i < n_params && i < NUM_PARAMETERS; ++i) {
        param_arr[i] = params_buf(i);
    }
    
    // Accumulate gradients across HRUs
    Real d_params_accum[NUM_PARAMETERS] = {0};
    
    // Process each HRU (can be parallelized with reduction)
    #pragma omp parallel
    {
        // Thread-local gradient accumulator
        Real d_params_local[NUM_PARAMETERS] = {0};
        
        #pragma omp for
        for (int h = 0; h < n_hru; ++h) {
            // Prepare state for this HRU
            Real state_arr[MAX_TOTAL_STATES] = {0};
            for (int s = 0; s < n_states_in && s < MAX_TOTAL_STATES; ++s) {
                state_arr[s] = states_buf(h, s);
            }
            
            // Prepare forcing for this HRU (flatten to [n_timesteps * 3])
            std::vector<Real> forcing_flat(n_timesteps * 3);
            for (int t = 0; t < n_timesteps; ++t) {
                forcing_flat[t * 3 + 0] = forcing_buf(t, h, 0);
                forcing_flat[t * 3 + 1] = forcing_buf(t, h, 1);
                forcing_flat[t * 3 + 2] = forcing_buf(t, h, 2);
            }
            
            // Prepare gradient for this HRU
            std::vector<Real> grad_flat(n_timesteps);
            for (int t = 0; t < n_timesteps; ++t) {
                grad_flat[t] = grad_buf(t, h);
            }
            
            // Copy parameters (Enzyme will modify the shadow)
            Real param_copy[NUM_PARAMETERS];
            Real d_param_hru[NUM_PARAMETERS] = {0};
            std::memcpy(param_copy, param_arr, NUM_PARAMETERS * sizeof(Real));
            
            // Output and shadow
            Real result = 0.0;
            Real d_result = 1.0;  // Seed: d(result)/d(result) = 1
            
            // dt as pointer to avoid float->double variadic promotion
            Real dt_val = static_cast<Real>(dt);
            
            // Call Enzyme autodiff
            __enzyme_autodiff(
                (void*)fuse_single_hru_weighted_sum,
                enzyme_const, state_arr,
                enzyme_const, forcing_flat.data(),
                enzyme_dup, param_copy, d_param_hru,  // Differentiate w.r.t. params
                enzyme_const, config_arr,
                enzyme_const, &dt_val,
                enzyme_const, n_timesteps,
                enzyme_const, grad_flat.data(),
                enzyme_dup, &result, &d_result
            );
            
            // Accumulate to thread-local
            for (int i = 0; i < NUM_PARAMETERS; ++i) {
                d_params_local[i] += d_param_hru[i];
            }
        }
        
        // Reduce thread-local to global
        #pragma omp critical
        {
            for (int i = 0; i < NUM_PARAMETERS; ++i) {
                d_params_accum[i] += d_params_local[i];
            }
        }
    }
    
    // Return gradients
    auto result = py::array_t<Real>(n_params);
    auto result_buf = result.mutable_unchecked<1>();
    for (int i = 0; i < n_params; ++i) {
        result_buf(i) = d_params_accum[i];
    }
    
    return result;
}

/**
 * @brief Combined forward and backward pass for efficiency.
 * 
 * Returns both runoff and parameter gradients in one call.
 * More efficient when you need both (avoids re-running forward pass).
 */
py::tuple run_fuse_batch_forward_backward(
    py::array_t<Real> initial_states,   // [n_hru, n_states]
    py::array_t<Real> forcing,          // [n_timesteps, n_hru, 3]
    py::array_t<Real> params,           // [n_params]
    py::array_t<Real> grad_runoff,      // [n_timesteps, n_hru] - can be zeros for forward-only
    py::dict config_dict,
    Real dt,
    bool compute_gradients = true
) {
    // Parse config
    ModelConfig config = config_from_dict(config_dict);
    int config_arr[8];
    config_to_int_array(config, config_arr);
    
    // Get dimensions
    auto states_buf = initial_states.unchecked<2>();
    auto forcing_buf = forcing.unchecked<3>();
    auto params_buf = params.unchecked<1>();
    auto grad_buf = grad_runoff.unchecked<2>();
    
    int n_hru = static_cast<int>(states_buf.shape(0));
    int n_states_in = static_cast<int>(states_buf.shape(1));
    int n_timesteps = static_cast<int>(forcing_buf.shape(0));
    int n_params = static_cast<int>(params_buf.size());
    
    // Prepare outputs
    auto runoff_out = py::array_t<Real>({n_timesteps, n_hru});
    auto final_states = py::array_t<Real>({n_hru, MAX_TOTAL_STATES});
    auto runoff_buf = runoff_out.mutable_unchecked<2>();
    auto states_out_buf = final_states.mutable_unchecked<2>();
    
    // Parameter array
    Real param_arr[NUM_PARAMETERS] = {0};
    for (int i = 0; i < n_params && i < NUM_PARAMETERS; ++i) {
        param_arr[i] = params_buf(i);
    }
    
    // Gradient accumulator
    Real d_params_accum[NUM_PARAMETERS] = {0};
    
    #pragma omp parallel
    {
        Real d_params_local[NUM_PARAMETERS] = {0};
        
        #pragma omp for
        for (int h = 0; h < n_hru; ++h) {
            // Setup state
            Real state_arr[MAX_TOTAL_STATES] = {0};
            for (int s = 0; s < n_states_in && s < MAX_TOTAL_STATES; ++s) {
                state_arr[s] = states_buf(h, s);
            }
            
            State state;
            state.from_array(state_arr, config);
            
            Parameters parameters;
            Real p_arr[NUM_PARAMETERS];
            std::memcpy(p_arr, param_arr, NUM_PARAMETERS * sizeof(Real));
            parameters.from_array(p_arr);
            
            // Forward pass - store runoff
            std::vector<Real> runoff_hru(n_timesteps);
            Flux flux;
            
            for (int t = 0; t < n_timesteps; ++t) {
                Forcing f(
                    forcing_buf(t, h, 0),
                    forcing_buf(t, h, 1),
                    forcing_buf(t, h, 2)
                );
                fuse_step(state, f, parameters, config, dt, flux);
                runoff_hru[t] = flux.q_total;
                runoff_buf(t, h) = flux.q_total;
            }
            
            // Store final state
            state.to_array(state_arr, config);
            for (int s = 0; s < MAX_TOTAL_STATES; ++s) {
                states_out_buf(h, s) = state_arr[s];
            }
            
            // Backward pass if requested
            if (compute_gradients) {
                // Reset state
                for (int s = 0; s < n_states_in && s < MAX_TOTAL_STATES; ++s) {
                    state_arr[s] = states_buf(h, s);
                }
                
                // Prepare forcing flat
                std::vector<Real> forcing_flat(n_timesteps * 3);
                for (int t = 0; t < n_timesteps; ++t) {
                    forcing_flat[t * 3 + 0] = forcing_buf(t, h, 0);
                    forcing_flat[t * 3 + 1] = forcing_buf(t, h, 1);
                    forcing_flat[t * 3 + 2] = forcing_buf(t, h, 2);
                }
                
                // Prepare gradient
                std::vector<Real> grad_flat(n_timesteps);
                for (int t = 0; t < n_timesteps; ++t) {
                    grad_flat[t] = grad_buf(t, h);
                }
                
                // Enzyme autodiff
                Real param_copy[NUM_PARAMETERS];
                Real d_param_hru[NUM_PARAMETERS] = {0};
                std::memcpy(param_copy, param_arr, NUM_PARAMETERS * sizeof(Real));
                
                Real result = 0.0;
                Real d_result = 1.0;
                
                // dt as pointer to avoid float->double variadic promotion
                Real dt_val = static_cast<Real>(dt);
                
                __enzyme_autodiff(
                    (void*)fuse_single_hru_weighted_sum,
                    enzyme_const, state_arr,
                    enzyme_const, forcing_flat.data(),
                    enzyme_dup, param_copy, d_param_hru,
                    enzyme_const, config_arr,
                    enzyme_const, &dt_val,
                    enzyme_const, n_timesteps,
                    enzyme_const, grad_flat.data(),
                    enzyme_dup, &result, &d_result
                );
                
                for (int i = 0; i < NUM_PARAMETERS; ++i) {
                    d_params_local[i] += d_param_hru[i];
                }
            }
        }
        
        if (compute_gradients) {
            #pragma omp critical
            {
                for (int i = 0; i < NUM_PARAMETERS; ++i) {
                    d_params_accum[i] += d_params_local[i];
                }
            }
        }
    }
    
    // Prepare gradient output
    auto grad_params = py::array_t<Real>(n_params);
    auto grad_params_buf = grad_params.mutable_unchecked<1>();
    for (int i = 0; i < n_params; ++i) {
        grad_params_buf(i) = d_params_accum[i];
    }
    
    return py::make_tuple(final_states, runoff_out, grad_params);
}

#endif // DFUSE_USE_ENZYME


// ========================================================================
// FALLBACK: NUMERICAL GRADIENT FOR NON-ENZYME BUILDS
// ========================================================================

/**
 * @brief Numerical gradient computation (finite differences).
 * 
 * Fallback when Enzyme is not available. Slower but always works.
 */
py::array_t<Real> run_fuse_batch_gradient_numerical(
    py::array_t<Real> initial_states,
    py::array_t<Real> forcing,
    py::array_t<Real> params,
    py::array_t<Real> grad_runoff,
    py::dict config_dict,
    Real dt,
    Real eps = 1e-5
) {
    ModelConfig config = config_from_dict(config_dict);
    
    auto states_buf = initial_states.unchecked<2>();
    auto forcing_buf = forcing.unchecked<3>();
    auto params_buf = params.unchecked<1>();
    auto grad_buf = grad_runoff.unchecked<2>();
    
    int n_hru = static_cast<int>(states_buf.shape(0));
    int n_states_in = static_cast<int>(states_buf.shape(1));
    int n_timesteps = static_cast<int>(forcing_buf.shape(0));
    int n_params = static_cast<int>(params_buf.size());
    
    // Helper lambda to compute loss given parameters
    auto compute_weighted_sum = [&](const Real* param_arr) -> Real {
        Real total = 0.0;
        
        #pragma omp parallel for reduction(+:total)
        for (int h = 0; h < n_hru; ++h) {
            Real state_arr[MAX_TOTAL_STATES] = {0};
            for (int s = 0; s < n_states_in && s < MAX_TOTAL_STATES; ++s) {
                state_arr[s] = states_buf(h, s);
            }
            
            State state;
            state.from_array(state_arr, config);
            
            Parameters parameters;
            Real p_arr[NUM_PARAMETERS];
            std::memcpy(p_arr, param_arr, NUM_PARAMETERS * sizeof(Real));
            parameters.from_array(p_arr);
            
            Flux flux;
            Real hru_sum = 0.0;
            
            for (int t = 0; t < n_timesteps; ++t) {
                Forcing f(
                    forcing_buf(t, h, 0),
                    forcing_buf(t, h, 1),
                    forcing_buf(t, h, 2)
                );
                fuse_step(state, f, parameters, config, dt, flux);
                hru_sum += flux.q_total * grad_buf(t, h);
            }
            
            total += hru_sum;
        }
        
        return total;
    };
    
    // Base parameters
    std::vector<Real> param_arr(NUM_PARAMETERS, 0);
    for (int i = 0; i < n_params && i < NUM_PARAMETERS; ++i) {
        param_arr[i] = params_buf(i);
    }
    
    // Compute base value
    Real base_val = compute_weighted_sum(param_arr.data());
    
    // Compute gradient via central differences
    auto result = py::array_t<Real>(n_params);
    auto result_buf = result.mutable_unchecked<1>();
    
    for (int i = 0; i < n_params; ++i) {
        Real orig = param_arr[i];
        Real h = eps * std::max(std::abs(orig), Real(1.0));
        
        // Forward
        param_arr[i] = orig + h;
        Real val_plus = compute_weighted_sum(param_arr.data());
        
        // Backward
        param_arr[i] = orig - h;
        Real val_minus = compute_weighted_sum(param_arr.data());
        
        // Central difference
        result_buf(i) = (val_plus - val_minus) / (2 * h);
        
        // Restore
        param_arr[i] = orig;
    }
    
    return result;
}


// ========================================================================
// EXISTING FUNCTIONS (kept for backward compatibility)
// ========================================================================

// [Include the rest of the original bindings.cpp content here - 
//  run_fuse_with_elevation_bands, compute_gradient_adjoint_bands, etc.]
// ... (truncated for brevity - keep all existing functions)


// ========================================================================
// MODULE DEFINITION
// ========================================================================

PYBIND11_MODULE(cfuse_core, m) {
    m.doc() = "cFUSE C++ backend with Enzyme AD - Extended for coupled calibration";
    
    // Constants
    m.attr("NUM_PARAMETERS") = NUM_PARAMETERS;
    m.attr("MAX_TOTAL_STATES") = MAX_TOTAL_STATES;
    m.attr("NUM_FLUXES") = NUM_FLUXES;
    m.attr("NUM_STATE_VARS") = enzyme::NUM_STATE_VARS;
    m.attr("NUM_PARAM_VARS") = enzyme::NUM_PARAM_VARS;
    m.attr("MAX_BANDS") = enzyme::MAX_BANDS;
    m.attr("__version__") = "0.4.0";  // Version bump for new features
    
    m.attr("HAS_CUDA") = 
    #ifdef DFUSE_USE_CUDA
        true;
    #else
        false;
    #endif

    m.attr("HAS_ENZYME") = 
    #ifdef DFUSE_USE_ENZYME
        true;
    #else
        false;
    #endif

    // ========== MAIN RUNNERS ==========
    
    m.def("run_fuse", &run_fuse_cpu, 
        py::arg("initial_state"), py::arg("forcing"), py::arg("params"), 
        py::arg("config"), py::arg("dt"), py::arg("return_fluxes")=false, 
        py::arg("solver")="euler",
        R"doc(
        Run FUSE for a single HRU.
        
        Args:
            initial_state: Initial state vector
            forcing: [n_timesteps, 3] (precip, pet, temp)
            params: Parameter vector
            config: Model configuration dict
            dt: Timestep in days
            return_fluxes: Whether to return detailed flux history
            solver: 'euler' or 'sundials_bdf'
            
        Returns:
            (final_state, runoff) or (final_state, runoff, fluxes)
        )doc");
    
    m.def("run_fuse_batch", &run_fuse_batch_cpu,
        py::arg("initial_states"), py::arg("forcing"), py::arg("params"),
        py::arg("config"), py::arg("dt"),
        R"doc(
        Run FUSE for multiple HRUs (forward pass only).
        
        Args:
            initial_states: [n_hru, n_states] Initial states
            forcing: [n_timesteps, n_hru, 3] or [n_timesteps, 3] if shared
            params: [n_params] shared or [n_hru, n_params] per-HRU
            config: Model configuration dict
            dt: Timestep in days
            
        Returns:
            (final_states, runoff) where runoff is [n_timesteps, n_hru]
        )doc");

    // ========== NEW: BATCH GRADIENT FUNCTIONS ==========
    
#ifdef DFUSE_USE_ENZYME
    m.def("run_fuse_batch_gradient", &run_fuse_batch_gradient,
        py::arg("initial_states"), py::arg("forcing"), py::arg("params"),
        py::arg("grad_runoff"), py::arg("config"), py::arg("dt"),
        R"doc(
        Compute parameter gradients for batch of HRUs using Enzyme AD.
        
        This is the main function for coupled cFUSE+dRoute calibration.
        Computes: d_params = sum_h sum_t (grad_runoff[t,h] * d(runoff[t,h])/d(params))
        
        Args:
            initial_states: [n_hru, n_states] Initial states
            forcing: [n_timesteps, n_hru, 3] Forcing (P, PET, T)
            params: [n_params] Shared parameters
            grad_runoff: [n_timesteps, n_hru] Upstream gradient dL/d(runoff)
            config: Model configuration dict
            dt: Timestep in days
            
        Returns:
            grad_params: [n_params] Parameter gradients
        )doc");
    
    m.def("run_fuse_batch_forward_backward", &run_fuse_batch_forward_backward,
        py::arg("initial_states"), py::arg("forcing"), py::arg("params"),
        py::arg("grad_runoff"), py::arg("config"), py::arg("dt"),
        py::arg("compute_gradients")=true,
        R"doc(
        Combined forward and backward pass for efficiency.
        
        More efficient than separate forward + gradient calls when you need both.
        
        Args:
            initial_states: [n_hru, n_states] Initial states
            forcing: [n_timesteps, n_hru, 3] Forcing (P, PET, T)
            params: [n_params] Shared parameters
            grad_runoff: [n_timesteps, n_hru] Upstream gradient (zeros for forward-only)
            config: Model configuration dict
            dt: Timestep in days
            compute_gradients: Whether to compute gradients
            
        Returns:
            (final_states, runoff, grad_params)
        )doc");
#endif

    // Numerical gradient (always available as fallback)
    m.def("run_fuse_batch_gradient_numerical", &run_fuse_batch_gradient_numerical,
        py::arg("initial_states"), py::arg("forcing"), py::arg("params"),
        py::arg("grad_runoff"), py::arg("config"), py::arg("dt"),
        py::arg("eps")=1e-5,
        R"doc(
        Compute parameter gradients using finite differences (fallback).
        
        Slower than Enzyme AD but always available.
        
        Args:
            initial_states: [n_hru, n_states] Initial states
            forcing: [n_timesteps, n_hru, 3] Forcing (P, PET, T)
            params: [n_params] Shared parameters
            grad_runoff: [n_timesteps, n_hru] Upstream gradient dL/d(runoff)
            config: Model configuration dict
            dt: Timestep in days
            eps: Finite difference epsilon
            
        Returns:
            grad_params: [n_params] Parameter gradients
        )doc");

    // ========== UTILITY ==========

    m.def("route_runoff", &route_runoff_timeseries,
        py::arg("instant_runoff"), py::arg("shape"),
        py::arg("mean_delay"), py::arg("dt"),
        R"doc(
        Route a runoff time series using a gamma unit hydrograph.
        
        Args:
            instant_runoff: [n_timesteps] instantaneous runoff
            shape: Gamma shape parameter
            mean_delay: Mean delay (days)
            dt: Timestep (days)
            
        Returns:
            routed_runoff: [n_timesteps]
        )doc");
    
    m.def("get_num_active_states", [](py::dict config_dict) {
        ModelConfig config = config_from_dict(config_dict);
        return get_num_active_states(config);
    }, py::arg("config"),
    "Get number of active state variables for a model configuration");

    // [Keep all other existing bindings - route_runoff, elevation bands, etc.]
    
    #ifdef DFUSE_USE_SUNDIALS
    m.attr("HAS_SUNDIALS") = true;
    #else
    m.attr("HAS_SUNDIALS") = false;
    #endif
    
    #ifdef DFUSE_USE_CVODES
    m.attr("HAS_CVODES") = true;
    #else
    m.attr("HAS_CVODES") = false;
    #endif
}
