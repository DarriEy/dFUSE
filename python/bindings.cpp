/**
 * @file bindings.cpp
 * @brief Python bindings for dFUSE via pybind11
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
 #include <pybind11/stl.h>
 
 #include "dfuse/dfuse.hpp"
 #include "dfuse/kernels.hpp"
 #include "dfuse/physics.hpp"
 #include "dfuse/enzyme_ad.hpp"
 #include "dfuse/solver.hpp"
 
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
 
 // Helpers
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
     arr[4] = static_cast<int>(config.baseflow);
     arr[5] = static_cast<int>(config.surface_runoff);
     arr[6] = static_cast<int>(config.interflow);
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
 
 // Forward Runner
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
     if (solver_type == "sundials") {
         #ifdef DFUSE_USE_SUNDIALS
         solver_cfg.method = solver::SolverMethod::SUNDIALS_BDF;
         solver_cfg.rel_tol = 1e-5;
         solver_cfg.abs_tol = 1e-7;
         #else
         throw std::runtime_error("SUNDIALS not compiled. Rebuild with -DDFUSE_USE_SUNDIALS=ON");
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
 
 // Trajectory Runner
 py::tuple run_fuse_forward_with_trajectory(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::dict config_dict,
     Real dt,
     std::string solver_type = "euler"
 ) {
     ModelConfig config = config_from_dict(config_dict);
     auto forcing_buf = forcing.unchecked<2>();
     int n_timesteps = static_cast<int>(forcing_buf.shape(0));
     
     State state = state_from_numpy(initial_state, config);
     Parameters parameters = params_from_numpy(params);
     
     solver::SolverConfig solver_cfg;
     if (solver_type == "sundials") {
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
     
     auto state_trajectory = py::array_t<Real>({n_timesteps + 1, enzyme::NUM_STATE_VARS});
     auto state_traj_buf = state_trajectory.mutable_unchecked<2>();
     
     Real state_arr[enzyme::NUM_STATE_VARS];
     state.to_array(state_arr, config);
     for (int i = 0; i < enzyme::NUM_STATE_VARS; ++i) state_traj_buf(0, i) = state_arr[i];
     
     Flux flux;
     for (int t = 0; t < n_timesteps; ++t) {
         Forcing f(forcing_buf(t, 0), forcing_buf(t, 1), forcing_buf(t, 2));
         solver_obj.solve(state, f, parameters, config, dt, flux);
         runoff_buf(t) = flux.q_total;
         
         state.to_array(state_arr, config);
         for (int i = 0; i < enzyme::NUM_STATE_VARS; ++i) state_traj_buf(t + 1, i) = state_arr[i];
     }
     
     return py::make_tuple(state_to_numpy(state, config), runoff, state_trajectory);
 }
 
 // Elevation Bands Runner
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
             flux_buf(t, 11) = flux_history[t].qufof;
             flux_buf(t, 12) = flux_history[t].qsfof;
             flux_buf(t, 13) = flux_history[t].throughfall;
             flux_buf(t, 14) = flux_history[t].qutof;
             flux_buf(t, 15) = flux_history[t].qstof;
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
             flux_buf(t, 11) = flux_history[t].qufof;
             flux_buf(t, 12) = flux_history[t].qsfof;
             flux_buf(t, 13) = flux_history[t].throughfall;
             flux_buf(t, 14) = flux_history[t].qutof;
             flux_buf(t, 15) = flux_history[t].qstof;
         }
         return py::make_tuple(final_state, runoff, fluxes, final_swe_bands);
     } else if (return_swe_trajectory) {
         return py::make_tuple(final_state, runoff, final_swe_bands, swe_traj);
     }
     
     return py::make_tuple(final_state, runoff, final_swe_bands);
 }
 
 // Adjoint Gradient
 py::array_t<Real> compute_gradient_adjoint(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::array_t<Real> grad_runoff,
     py::array_t<Real> state_trajectory,
     py::dict config_dict,
     Real dt
 ) {
     ModelConfig config = config_from_dict(config_dict);
     int config_arr[enzyme::NUM_CONFIG_VARS];
     config_to_int_array(config, config_arr);
     
     auto forcing_buf = forcing.unchecked<2>();
     auto params_buf = params.unchecked<1>();
     auto grad_runoff_buf = grad_runoff.unchecked<1>();
     auto state_traj_buf = state_trajectory.unchecked<2>();
     
     int n_timesteps = static_cast<int>(forcing_buf.shape(0));
     int n_params = static_cast<int>(params_buf.size());
     
     std::vector<Real> forcing_flat(n_timesteps * 3);
     for (int t = 0; t < n_timesteps; ++t) {
         forcing_flat[t * 3 + 0] = forcing_buf(t, 0);
         forcing_flat[t * 3 + 1] = forcing_buf(t, 1);
         forcing_flat[t * 3 + 2] = forcing_buf(t, 2);
     }
     
     Real param_arr[enzyme::NUM_PARAM_VARS] = {0};
     for (int i = 0; i < n_params && i < enzyme::NUM_PARAM_VARS; ++i) param_arr[i] = params_buf(i);
     Real grad_params_arr[enzyme::NUM_PARAM_VARS] = {0};
     
     std::vector<Real> adjoint_state(enzyme::NUM_STATE_VARS, 0.0);
     std::vector<Real> adjoint_next(enzyme::NUM_STATE_VARS, 0.0);
     const Real eps = 1e-5;
     const Real grad_clip = 1e6;
     auto clip = [&](Real v) { return std::max(std::min(v, grad_clip), -grad_clip); };
     
     for (int t = n_timesteps - 1; t >= 0; --t) {
         const Real* forcing_t = &forcing_flat[t * 3];
         Real state_t[enzyme::NUM_STATE_VARS];
         for(int i=0; i<enzyme::NUM_STATE_VARS; ++i) state_t[i] = state_traj_buf(t, i);
         
         Real dloss_drunoff = grad_runoff_buf(t);
         if (!std::isfinite(dloss_drunoff)) dloss_drunoff = 0.0;
         
         Real state_out_base[enzyme::NUM_STATE_VARS], runoff_base;
         enzyme::fuse_step_flat(state_t, forcing_t, param_arr, config_arr, dt, state_out_base, &runoff_base);
         
         Real p_pert[enzyme::NUM_PARAM_VARS];
         std::memcpy(p_pert, param_arr, sizeof(param_arr));
         
         for (int p=0; p < enzyme::NUM_PARAM_VARS; ++p) {
             Real orig = p_pert[p];
             // FIX: Explicit cast to Real to resolve float/double ambiguity
             Real h = std::max(eps * std::abs(orig), Real(1e-8));
             p_pert[p] += h;
             Real s_tmp[enzyme::NUM_STATE_VARS], r_pert;
             enzyme::fuse_step_flat(state_t, forcing_t, p_pert, config_arr, dt, s_tmp, &r_pert);
             
             Real dR_dP = (r_pert - runoff_base) / h;
             Real dS_dP[enzyme::NUM_STATE_VARS];
             for(int k=0; k<enzyme::NUM_STATE_VARS; ++k) dS_dP[k] = (s_tmp[k] - state_out_base[k])/h;
             
             grad_params_arr[p] += dloss_drunoff * dR_dP;
             for(int k=0; k<enzyme::NUM_STATE_VARS; ++k) grad_params_arr[p] += adjoint_state[k] * dS_dP[k];
             p_pert[p] = orig;
         }
         
         Real s_pert[enzyme::NUM_STATE_VARS];
         std::memcpy(s_pert, state_t, sizeof(state_t));
         std::fill(adjoint_next.begin(), adjoint_next.end(), 0.0);
         
         for(int j=0; j<enzyme::NUM_STATE_VARS; ++j) {
             Real orig = s_pert[j];
             // FIX: Explicit cast
             Real h = std::max(eps * std::abs(orig), Real(1e-8));
             s_pert[j] += h;
             Real s_out_pert[enzyme::NUM_STATE_VARS], r_pert;
             enzyme::fuse_step_flat(s_pert, forcing_t, param_arr, config_arr, dt, s_out_pert, &r_pert);
             
             Real dR_dS = (r_pert - runoff_base) / h;
             adjoint_next[j] += dloss_drunoff * dR_dS;
             
             for(int k=0; k<enzyme::NUM_STATE_VARS; ++k) {
                 Real dS_dS = (s_out_pert[k] - state_out_base[k]) / h;
                 adjoint_next[j] += adjoint_state[k] * dS_dS;
             }
             s_pert[j] = orig;
         }
         for(int k=0; k<enzyme::NUM_STATE_VARS; ++k) adjoint_state[k] = clip(adjoint_next[k]);
     }
     
     auto grad_params = py::array_t<Real>(n_params);
     auto grad_buf = grad_params.mutable_unchecked<1>();
     for (int i = 0; i < n_params && i < enzyme::NUM_PARAM_VARS; ++i) grad_buf(i) = grad_params_arr[i];
     return grad_params;
 }
 
 // Batch Runner
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
     
     // FIX: Use pointer arithmetic for std::fill on unchecked array
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
 
 // Numerical Gradient
 py::array_t<Real> compute_gradient_numerical(
     py::array_t<Real> initial_state,
     py::array_t<Real> forcing,
     py::array_t<Real> params,
     py::array_t<Real> grad_runoff,
     py::dict config_dict,
     Real dt,
     Real eps = 1e-4
 ) {
     ModelConfig config = config_from_dict(config_dict);
     auto forcing_buf = forcing.unchecked<2>();
     auto params_buf = params.unchecked<1>();
     auto grad_runoff_buf = grad_runoff.unchecked<1>();
     int n_timesteps = static_cast<int>(forcing_buf.shape(0));
     int n_params = static_cast<int>(params_buf.size());
     
     auto compute_weighted_runoff = [&](const std::vector<Real>& param_vec) -> Real {
         Parameters parameters;
         Real param_arr[NUM_PARAMETERS];
         for (int i = 0; i < n_params && i < NUM_PARAMETERS; ++i) param_arr[i] = param_vec[i];
         parameters.from_array(param_arr);
         State state = state_from_numpy(initial_state, config);
         
         Real weighted_sum = 0.0;
         Flux flux;
         for (int t = 0; t < n_timesteps; ++t) {
             Forcing f(forcing_buf(t, 0), forcing_buf(t, 1), forcing_buf(t, 2));
             fuse_step(state, f, parameters, config, dt, flux);
             weighted_sum += flux.q_total * grad_runoff_buf(t);
         }
         return weighted_sum;
     };
     
     std::vector<Real> param_vec(n_params);
     for (int i = 0; i < n_params; ++i) param_vec[i] = params_buf(i);
     
     auto grad_params = py::array_t<Real>(n_params);
     auto grad_buf = grad_params.mutable_unchecked<1>();
     
     for (int i = 0; i < n_params; ++i) {
         Real orig = param_vec[i];
         // FIX: Explicit cast
         Real h = std::max(eps * std::abs(orig), Real(1e-8));
         param_vec[i] = orig + h;
         Real val_plus = compute_weighted_runoff(param_vec);
         param_vec[i] = orig - h;
         Real val_minus = compute_weighted_runoff(param_vec);
         grad_buf(i) = (val_plus - val_minus) / (2 * h);
         param_vec[i] = orig;
     }
     return grad_params;
 }
 
 // Module
 PYBIND11_MODULE(dfuse_core, m) {
     m.doc() = "dFUSE C++ backend";
     m.attr("NUM_PARAMETERS") = NUM_PARAMETERS;
     m.attr("MAX_TOTAL_STATES") = MAX_TOTAL_STATES;
     m.attr("NUM_FLUXES") = NUM_FLUXES;
     m.attr("NUM_STATE_VARS") = enzyme::NUM_STATE_VARS;
     m.attr("NUM_PARAM_VARS") = enzyme::NUM_PARAM_VARS;
     m.attr("__version__") = "0.2.0";
     
     m.def("run_fuse", &run_fuse_cpu, py::arg("initial_state"), py::arg("forcing"), py::arg("params"), py::arg("config"), py::arg("dt"), py::arg("return_fluxes")=false, py::arg("solver")="euler");
     m.def("run_fuse_forward_with_trajectory", &run_fuse_forward_with_trajectory, py::arg("initial_state"), py::arg("forcing"), py::arg("params"), py::arg("config"), py::arg("dt"), py::arg("solver")="euler");
     m.def("run_fuse_elevation_bands", &run_fuse_with_elevation_bands, py::arg("initial_state"), py::arg("forcing"), py::arg("params"), py::arg("config"), py::arg("area_frac"), py::arg("mean_elev"), py::arg("ref_elev"), py::arg("initial_swe")=py::none(), py::arg("dt")=1.0, py::arg("return_fluxes")=false, py::arg("return_swe_trajectory")=false, py::arg("start_day_of_year")=0, py::arg("solver")="euler");
     
     m.def("compute_gradient_adjoint", &compute_gradient_adjoint);
     m.def("compute_gradient_numerical", &compute_gradient_numerical, py::arg("initial_state"), py::arg("forcing"), py::arg("params"), py::arg("grad_runoff"), py::arg("config"), py::arg("dt"), py::arg("eps")=1e-4f);
     
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
     m.attr("HAS_CUDA") = true;
     #else
     m.attr("HAS_CUDA") = false;
     #endif
     
     #ifdef DFUSE_USE_SUNDIALS
     m.attr("HAS_SUNDIALS") = true;
     #else
     m.attr("HAS_SUNDIALS") = false;
     #endif
 }