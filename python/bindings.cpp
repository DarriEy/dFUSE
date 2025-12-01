/**
 * @file bindings.cpp
 * @brief Python bindings for dFUSE via pybind11 (Enzyme Enabled)
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
 #include <pybind11/stl.h>
 
 #include "dfuse/dfuse.hpp"
 #include "dfuse/kernels.hpp"
 #include "dfuse/physics.hpp"
 #include "dfuse/enzyme_ad.hpp"
 #include "dfuse/solver.hpp"
 #include "dfuse/routing.hpp"
 
 #include <vector>
 #include <cstring>
 #include <algorithm>
 #include <stdexcept>
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
 
 // ========================================================================
 // RUNNERS
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
 
 py::tuple run_fuse_forward_with_trajectory(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::dict config_dict,
     Real dt_total,
     std::string solver_type = "euler"
 ) {
     ModelConfig config = config_from_dict(config_dict);
     auto forcing_buf = forcing.unchecked<2>();
     auto params_buf = params.unchecked<1>();
     int n_timesteps = static_cast<int>(forcing_buf.shape(0));
     
     Real state_arr[enzyme::NUM_STATE_VARS];
     auto state_buf = initial_state.unchecked<1>();
     for(int i=0; i<std::min((int)state_buf.size(), enzyme::NUM_STATE_VARS); ++i) state_arr[i] = state_buf(i);
     
     Parameters parameters;
     Real param_arr[NUM_PARAMETERS];
     for (ssize_t i = 0; i < params_buf.size() && i < NUM_PARAMETERS; ++i) param_arr[i] = params_buf(i);
     parameters.from_array(param_arr);
     
     auto runoff = py::array_t<Real>(n_timesteps);
     auto runoff_buf = runoff.mutable_unchecked<1>();
     auto state_trajectory = py::none();
 
     constexpr int SUBSTEPS = 8;
     Real dt = dt_total / Real(SUBSTEPS);
     State state;
     state.from_array(state_arr, config);
     Flux flux;
 
     for (int t = 0; t < n_timesteps; ++t) {
         Forcing f(forcing_buf(t, 0), forcing_buf(t, 1), forcing_buf(t, 2));
         Real runoff_accum = 0.0;
         for (int sub = 0; sub < SUBSTEPS; ++sub) {
             fuse_step(state, f, parameters, config, dt, flux);
             runoff_accum += flux.q_total;
         }
         runoff_buf(t) = runoff_accum / Real(SUBSTEPS);
     }
     return py::make_tuple(state_to_numpy(state, config), runoff, state_trajectory);
 }
 
 py::tuple run_fuse_with_elevation_bands(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::dict config_dict,
     py::array_t<Real> area_frac,
     py::array_t<Real> mean_elev,
     Real ref_elev,
     py::object initial_swe_py,
     Real dt,
     bool return_fluxes = false,
     bool return_swe_trajectory = false,
     int start_day_of_year = 0,
     std::string solver = "euler"
 ) {
     ModelConfig config = config_from_dict(config_dict);
     auto forcing_buf = forcing.unchecked<2>();
     int n_timesteps = static_cast<int>(forcing_buf.shape(0));
     auto area_frac_buf = area_frac.unchecked<1>();
     auto mean_elev_buf = mean_elev.unchecked<1>();
     int n_bands = static_cast<int>(area_frac_buf.shape(0));
     State state = state_from_numpy(initial_state, config);
     Parameters parameters = params_from_numpy(params);
     
     constexpr int MAX_BANDS = 30;
     if (n_bands > MAX_BANDS) throw std::runtime_error("Too many elevation bands");
     
     Real area_frac_arr[MAX_BANDS], mean_elev_arr[MAX_BANDS], swe_bands[MAX_BANDS], swe_bands_new[MAX_BANDS];
     for (int b = 0; b < n_bands; ++b) {
         area_frac_arr[b] = area_frac_buf(b);
         mean_elev_arr[b] = mean_elev_buf(b);
         swe_bands[b] = Real(0);
     }
     
     if (!initial_swe_py.is_none()) {
         auto initial_swe = initial_swe_py.cast<py::array_t<Real>>();
         auto swe_buf = initial_swe.unchecked<1>();
         for (int b = 0; b < n_bands; ++b) swe_bands[b] = swe_buf(b);
     }
     
     auto runoff = py::array_t<Real>(n_timesteps);
     auto runoff_buf = runoff.mutable_unchecked<1>();
     py::array_t<Real> swe_traj;
     Real* swe_traj_ptr = nullptr;
     if (return_swe_trajectory) {
         swe_traj = py::array_t<Real>({n_timesteps, n_bands});
         swe_traj_ptr = swe_traj.mutable_data();
     }
     
     std::vector<Flux> flux_history;
     if (return_fluxes) flux_history.resize(n_timesteps);
     
     #ifdef DFUSE_USE_SUNDIALS
     std::unique_ptr<solver::SundialsSolver> sundials_solver;
     if (solver == "sundials_bdf" || solver == "sundials_adams") {
         solver::SolverConfig solver_cfg;
         solver_cfg.method = (solver == "sundials_bdf") ? solver::SolverMethod::SUNDIALS_BDF : solver::SolverMethod::SUNDIALS_ADAMS;
         solver_cfg.rel_tol = 1e-4;
         solver_cfg.abs_tol = 1e-6;
         sundials_solver = std::make_unique<solver::SundialsSolver>(solver_cfg);
     }
     #endif
     
     Flux flux;
     for (int t = 0; t < n_timesteps; ++t) {
         Real precip = forcing_buf(t, 0), pet = forcing_buf(t, 1), temp = forcing_buf(t, 2);
         int day_of_year = (start_day_of_year > 0) ? ((start_day_of_year - 1 + t) % 365) + 1 : 0;
         
         Real rain_eff, melt_eff;
         physics::compute_snow_elevation_bands<MAX_BANDS>(
             precip, temp, swe_bands, n_bands, area_frac_arr, mean_elev_arr, ref_elev,
             parameters, rain_eff, melt_eff, swe_bands_new, day_of_year
         );
         for (int b = 0; b < n_bands; ++b) swe_bands[b] = swe_bands_new[b];
         if (return_swe_trajectory) for (int b = 0; b < n_bands; ++b) swe_traj_ptr[t * n_bands + b] = swe_bands[b];
         
         Forcing f(rain_eff + melt_eff, pet, temp);
         ModelConfig config_no_snow = config;
         config_no_snow.enable_snow = false;
         
         #ifdef DFUSE_USE_SUNDIALS
         if (sundials_solver) sundials_solver->solve(state, f, parameters, config_no_snow, dt, flux);
         else fuse_step(state, f, parameters, config_no_snow, dt, flux);
         #else
         fuse_step(state, f, parameters, config_no_snow, dt, flux);
         #endif
         
         Real total_swe = 0;
         for (int b = 0; b < n_bands; ++b) total_swe += swe_bands[b] * area_frac_arr[b];
         state.SWE = total_swe;
         flux.rain = rain_eff; flux.melt = melt_eff;
         runoff_buf(t) = flux.q_total;
         if (return_fluxes) flux_history[t] = flux;
     }
     
     Real total_swe_final = 0;
     for (int b = 0; b < n_bands; ++b) total_swe_final += swe_bands[b] * area_frac_arr[b];
     state.SWE = total_swe_final;
     
     auto final_state = state_to_numpy(state, config);
     
     auto final_swe_bands = py::array_t<Real>(n_bands);
     auto swe_final_buf = final_swe_bands.mutable_unchecked<1>();
     for (int b = 0; b < n_bands; ++b) swe_final_buf(b) = swe_bands[b];
     
     if (return_fluxes && return_swe_trajectory) {
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
         return py::make_tuple(final_state, runoff, fluxes, final_swe_bands, swe_traj);
     } else if (return_fluxes) {
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
         return py::make_tuple(final_state, runoff, fluxes, final_swe_bands);
     } else if (return_swe_trajectory) {
         return py::make_tuple(final_state, runoff, final_swe_bands, swe_traj);
     }
     
     return py::make_tuple(final_state, runoff, final_swe_bands);
 }
 
 // 4. Batch Runner
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
         for (int s = 0; s < MAX_TOTAL_STATES; ++s) state_arr[s] = states_buf(h, s);
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
 // ENZYME ADJOINT IMPLEMENTATION
 // ========================================================================
 
 #ifdef DFUSE_USE_ENZYME
 
 // This function performs the entire simulation (Physics + Routing) and returns
 // VOID, outputting result via pointer. This allows explicit seed gradient passing.
 void fuse_physics_dot_product(
     const Real* state_in,       // [TOTAL_VARS_BANDS]
     const Real* forcing_flat,   // [nt * 3]
     const Real* param_arr,      // [NUM_PARAM_VARS]
     const int* config_arr,      // [NUM_CONFIG_VARS]
     const Real* dt_ptr,         // &dt
     const Real* band_props,     // [MAX_BANDS * 2]
     int n_bands,
     const Real* ref_elev_ptr,   // &ref_elev
     const Real* uh_weights,     // [uh_len] Pre-calculated unit hydrograph
     int uh_len,
     const Real* grad_output,    // [nt]
     int n_timesteps,
     Real* runoff_instant_workspace, // Pre-allocated buffer [nt]
     Real* weighted_sum_out      // [1] Output
 ) {
     // 1. Setup State
     Real state_arr[enzyme::TOTAL_VARS_BANDS];
     std::memcpy(state_arr, state_in, enzyme::TOTAL_VARS_BANDS * sizeof(Real));
     
     Real dt_total = *dt_ptr;
     constexpr int SUBSTEPS = 8;
     Real dt = dt_total / Real(SUBSTEPS);
     
     Real sum = 0.0;
     
     // 2. Physics Loop
     for (int t = 0; t < n_timesteps; ++t) {
         const Real* f_ptr = &forcing_flat[t * 3];
         Real daily_runoff = 0.0;
         
         for (int sub = 0; sub < SUBSTEPS; ++sub) {
             Real state_next[enzyme::TOTAL_VARS_BANDS];
             Real step_runoff;
             
             enzyme::fuse_step_bands_flat(
                 state_arr, f_ptr, param_arr, config_arr, dt,
                 band_props, n_bands, *ref_elev_ptr,
                 state_next, &step_runoff
             );
             
             daily_runoff += step_runoff;
             std::memcpy(state_arr, state_next, enzyme::TOTAL_VARS_BANDS * sizeof(Real));
         }
         // Average rate over the day
         runoff_instant_workspace[t] = daily_runoff / Real(SUBSTEPS);
     }
     
     // 3. Routing Loop (Convolution)
     for (int t = 0; t < n_timesteps; ++t) {
         Real routed_val = 0.0;
         for (int i = 0; i < uh_len && t - i >= 0; ++i) {
             routed_val += runoff_instant_workspace[t - i] * uh_weights[i];
         }
         sum += routed_val * grad_output[t];
     }
     
     *weighted_sum_out = sum;
 }
 
 #endif
 
 // The Python Binding
 py::array_t<Real> compute_gradient_adjoint_bands(
     py::array_t<Real> initial_state, // [TOTAL_VARS_BANDS]
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::array_t<Real> grad_runoff,
     py::dict config_dict,
     py::array_t<Real> area_frac,
     py::array_t<Real> mean_elev,
     Real ref_elev,
     Real route_shape,
     Real route_delay,
     Real dt
 ) {
     // Unpack Config
     ModelConfig config = config_from_dict(config_dict);
     int config_arr[enzyme::NUM_CONFIG_VARS];
     config_to_int_array(config, config_arr);
     
     // Unpack Buffers
     auto forcing_buf = forcing.unchecked<2>();
     auto params_buf = params.unchecked<1>();
     auto grad_buf = grad_runoff.unchecked<1>();
     auto state_buf = initial_state.unchecked<1>();
     auto area_buf = area_frac.unchecked<1>();
     auto elev_buf = mean_elev.unchecked<1>();
     
     int n_bands = area_buf.size();
     int n_timesteps = forcing_buf.shape(0);
     int n_params = params_buf.size();
     
     // Prepare Flat Forcing
     std::vector<Real> forcing_flat(n_timesteps * 3);
     for(int t=0; t<n_timesteps; ++t) {
         forcing_flat[t*3+0] = forcing_buf(t,0);
         forcing_flat[t*3+1] = forcing_buf(t,1);
         forcing_flat[t*3+2] = forcing_buf(t,2);
     }
     
     std::vector<Real> grad_out_flat(n_timesteps);
     for(int t=0; t<n_timesteps; ++t) grad_out_flat[t] = grad_buf(t);
     
     // Prepare Band Props [MAX_BANDS * 2]
     Real band_props[enzyme::MAX_BANDS * 2] = {0};
     for(int b=0; b<n_bands; ++b) {
         band_props[b] = area_buf(b);
         band_props[enzyme::MAX_BANDS + b] = elev_buf(b);
     }
 
     // Prepare Params & State
     Real param_arr[enzyme::NUM_PARAM_VARS] = {0};
     for(int i=0; i<n_params; ++i) param_arr[i] = params_buf(i);
     
     Real state_arr[enzyme::TOTAL_VARS_BANDS] = {0};
     for(int i=0; i<state_buf.size(); ++i) state_arr[i] = state_buf(i);
 
     // Generate Unit Hydrograph OUTSIDE of Enzyme
     std::vector<Real> uh;
     routing::generate_unit_hydrograph(route_shape, route_delay, dt, uh);
     int uh_len = uh.size();
 
     // Outputs
     Real d_params[enzyme::NUM_PARAM_VARS] = {0};
     
     // Workspaces
     std::vector<Real> runoff_ws(n_timesteps, 0.0);
     std::vector<Real> d_runoff_ws(n_timesteps, 0.0);
     
     Real result_val = 0.0;
     Real d_result_val = 1.0; // Seed gradient (dL/d(DotProduct) = 1)
 
 #ifdef DFUSE_USE_ENZYME
     __enzyme_autodiff(
         (void*)fuse_physics_dot_product,
         enzyme_const, state_arr,
         enzyme_const, forcing_flat.data(),
         enzyme_dup,   param_arr, d_params, // GRADIENT HERE
         enzyme_const, config_arr,
         enzyme_const, &dt,
         enzyme_const, band_props,
         enzyme_const, n_bands,
         enzyme_const, &ref_elev,        
         enzyme_const, uh.data(),        
         enzyme_const, uh_len,
         enzyme_const, grad_out_flat.data(),
         enzyme_const, n_timesteps,
         enzyme_dup,   runoff_ws.data(), d_runoff_ws.data(), // Workspace
         enzyme_dup,   &result_val, &d_result_val // Explicit Seed
     );
 #else
     throw std::runtime_error("Enzyme not enabled");
 #endif
 
     auto result = py::array_t<Real>(n_params);
     auto res_buf = result.mutable_unchecked<1>();
     for(int i=0; i<n_params; ++i) res_buf(i) = d_params[i];
     return result;
 }
 
 // Diagnostic function: compute numerical gradient for comparison
 py::tuple compute_gradient_numerical_debug(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::array_t<Real> grad_runoff,
     py::dict config_dict,
     py::array_t<Real> area_frac,
     py::array_t<Real> mean_elev,
     Real ref_elev,
     Real route_shape,
     Real route_delay,
     Real dt,
     Real eps = 1e-4
 ) {
     ModelConfig config = config_from_dict(config_dict);
     int config_arr[enzyme::NUM_CONFIG_VARS];
     config_to_int_array(config, config_arr);
     
     auto forcing_buf = forcing.unchecked<2>();
     auto params_buf = params.unchecked<1>();
     auto grad_buf = grad_runoff.unchecked<1>();
     auto state_buf = initial_state.unchecked<1>();
     auto area_buf = area_frac.unchecked<1>();
     auto elev_buf = mean_elev.unchecked<1>();
     
     int n_bands = area_buf.size();
     int n_timesteps = forcing_buf.shape(0);
     int n_params = params_buf.size();
     
     std::vector<Real> forcing_flat(n_timesteps * 3);
     for(int t=0; t<n_timesteps; ++t) {
         forcing_flat[t*3+0] = forcing_buf(t,0);
         forcing_flat[t*3+1] = forcing_buf(t,1);
         forcing_flat[t*3+2] = forcing_buf(t,2);
     }
     
     std::vector<Real> grad_out_flat(n_timesteps);
     for(int t=0; t<n_timesteps; ++t) grad_out_flat[t] = grad_buf(t);
     
     Real band_props[enzyme::MAX_BANDS * 2] = {0};
     for(int b=0; b<n_bands; ++b) {
         band_props[b] = area_buf(b);
         band_props[enzyme::MAX_BANDS + b] = elev_buf(b);
     }
     
     Real state_arr[enzyme::TOTAL_VARS_BANDS] = {0};
     for(int i=0; i<state_buf.size(); ++i) state_arr[i] = state_buf(i);
     
     std::vector<Real> uh;
     routing::generate_unit_hydrograph(route_shape, route_delay, dt, uh);
     int uh_len = uh.size();
     
     // Compute base value
     Real param_arr[enzyme::NUM_PARAM_VARS] = {0};
     for(int i=0; i<n_params; ++i) param_arr[i] = params_buf(i);
     
     std::vector<Real> runoff_ws(n_timesteps, 0.0);
     Real base_result = 0.0;
     
     fuse_physics_dot_product(
         state_arr, forcing_flat.data(), param_arr, config_arr, &dt,
         band_props, n_bands, &ref_elev, uh.data(), uh_len,
         grad_out_flat.data(), n_timesteps, runoff_ws.data(), &base_result
     );
     
     // Compute numerical gradient
     auto num_grad = py::array_t<Real>(n_params);
     auto num_grad_buf = num_grad.mutable_unchecked<1>();
     
     for(int i=0; i<n_params; ++i) {
         // Reset
         for(int j=0; j<n_params; ++j) param_arr[j] = params_buf(j);
         std::fill(runoff_ws.begin(), runoff_ws.end(), 0.0);
         
         // Perturb +eps
         Real orig = param_arr[i];
         Real h = eps * std::max(std::abs(orig), Real(1.0));
         param_arr[i] = orig + h;
         
         Real result_plus = 0.0;
         fuse_physics_dot_product(
             state_arr, forcing_flat.data(), param_arr, config_arr, &dt,
             band_props, n_bands, &ref_elev, uh.data(), uh_len,
             grad_out_flat.data(), n_timesteps, runoff_ws.data(), &result_plus
         );
         
         // Compute gradient
         num_grad_buf(i) = (result_plus - base_result) / h;
         param_arr[i] = orig;
     }
     
     // Also return base result to check for NaN in forward pass
     return py::make_tuple(num_grad, base_result);
 }
 
 // ========================================================================
 // LEGACY
 // ========================================================================
 py::array_t<Real> compute_gradient_adjoint(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::array_t<Real> grad_runoff,
     py::object state_trajectory,
     py::dict config_dict,
     Real dt
 ) {
     throw std::runtime_error("Use compute_gradient_adjoint_bands for full physics gradients");
 }
 
 py::array_t<Real> compute_gradient_numerical(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::array_t<Real> grad_runoff,
     py::dict config_dict,
     Real dt,
     Real eps = 1e-4
 ) {
     // Stub implementation
     return py::array_t<Real>(NUM_PARAMETERS);
 }
 
 
 // ========================================================================
 // MODULE DEFINITION
 // ========================================================================
 
 PYBIND11_MODULE(dfuse_core, m) {
     m.doc() = "dFUSE C++ backend with Enzyme AD";
     m.attr("NUM_PARAMETERS") = NUM_PARAMETERS;
     m.attr("MAX_TOTAL_STATES") = MAX_TOTAL_STATES;
     m.attr("NUM_FLUXES") = NUM_FLUXES;
     m.attr("NUM_STATE_VARS") = enzyme::NUM_STATE_VARS;
     m.attr("NUM_PARAM_VARS") = enzyme::NUM_PARAM_VARS;
     m.attr("MAX_BANDS") = enzyme::MAX_BANDS;
     m.attr("__version__") = "0.2.0";
     
     m.attr("HAS_CUDA") = 
     #ifdef DFUSE_USE_CUDA
         true;
     #else
         false;
     #endif
 
     // Main Runners
     m.def("run_fuse", &run_fuse_cpu, 
         py::arg("initial_state"), py::arg("forcing"), py::arg("params"), 
         py::arg("config"), py::arg("dt"), py::arg("return_fluxes")=false, 
         py::arg("solver")="euler");
 
     m.def("run_fuse_elevation_bands", &run_fuse_with_elevation_bands, 
         py::arg("initial_state"), py::arg("forcing"), py::arg("params"), 
         py::arg("config"), py::arg("area_frac"), py::arg("mean_elev"), 
         py::arg("ref_elev"), py::arg("initial_swe")=py::none(), 
         py::arg("dt")=1.0, py::arg("return_fluxes")=false, 
         py::arg("return_swe_trajectory")=false, py::arg("start_day_of_year")=0, 
         py::arg("solver")="euler");
 
     // New Gradient Function
     m.def("compute_gradient_adjoint_bands", &compute_gradient_adjoint_bands);
     
     // Debug: Numerical gradient for comparison
     m.def("compute_gradient_numerical_debug", &compute_gradient_numerical_debug,
         py::arg("initial_state"), py::arg("forcing"), py::arg("params"),
         py::arg("grad_runoff"), py::arg("config"), py::arg("area_frac"),
         py::arg("mean_elev"), py::arg("ref_elev"), py::arg("route_shape"),
         py::arg("route_delay"), py::arg("dt"), py::arg("eps")=1e-4f);
     
     // Legacy/Utility
     m.def("run_fuse_forward_with_trajectory", &run_fuse_forward_with_trajectory, 
         py::arg("initial_state"), py::arg("forcing"), py::arg("params"), 
         py::arg("config"), py::arg("dt"), py::arg("solver")="euler");
 
     m.def("compute_gradient_adjoint", &compute_gradient_adjoint);
     m.def("compute_gradient_numerical", &compute_gradient_numerical, 
         py::arg("initial_state"), py::arg("forcing"), py::arg("params"), 
         py::arg("grad_runoff"), py::arg("config"), py::arg("dt"), py::arg("eps")=1e-4f);
     
     m.def("route_runoff", [](py::array_t<Real> r, Real shape, Real delay, Real dt) {
         auto r_buf = r.unchecked<1>();
         int n = r_buf.shape(0);
         auto out = py::array_t<Real>(n);
         auto out_buf = out.mutable_unchecked<1>();
         std::vector<Real> uh;
         int len = routing::generate_unit_hydrograph(shape, delay, dt, uh);
         for(int t=0; t<n; ++t) {
              Real sum = 0;
              for(int i=0; i<len && i<=t; ++i) sum += r_buf(t-i) * uh[i];
              out_buf(t) = sum;
         }
         return out;
     });
     
     m.def("get_unit_hydrograph", [](Real shape, Real delay, Real dt, int max_len) {
         std::vector<Real> uh;
         int len = routing::generate_unit_hydrograph(shape, delay, dt, uh, max_len);
         auto out = py::array_t<Real>(len);
         auto buf = out.mutable_unchecked<1>();
         for(int i=0; i<len; ++i) buf(i) = uh[i];
         return out;
     });
     
     m.def("run_fuse_batch", &run_fuse_batch_cpu);
 
     #ifdef DFUSE_USE_CUDA
     m.def("run_fuse_cuda", &run_fuse_cuda_batch, py::arg("initial_states"), py::arg("forcing"), py::arg("params"), py::arg("config"), py::arg("dt"));
     m.def("run_fuse_cuda_workspace", &run_fuse_cuda_batch_workspace, py::arg("initial_states"), py::arg("forcing"), py::arg("params"), py::arg("config"), py::arg("dt"), py::arg("workspace_ptr"), py::arg("workspace_size"));
     m.def("compute_cuda_workspace_size", &compute_cuda_workspace_size);
     #endif
     
     #ifdef DFUSE_USE_SUNDIALS
     m.attr("HAS_SUNDIALS") = true;
     #else
     m.attr("HAS_SUNDIALS") = false;
     #endif
 }