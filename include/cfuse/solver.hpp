/**
 * @file solver.hpp
 * @brief ODE solver integration for cFUSE using SUNDIALS CVODE
 * 
 * Provides robust implicit/explicit ODE solving with adaptive timestepping.
 * SUNDIALS CVODE handles stiff systems that arise from fast baseflow dynamics.
 * 
 * References:
 * - Clark & Kavetski (2010) "Ancient numerical daemons" WRR
 * - Kavetski & Clark (2010) "Numerical solver effects" WRR
 */

 #pragma once

 #include "config.hpp"
 #include "state.hpp"
 #include "physics.hpp"
 
 // SUNDIALS headers
 #ifdef DFUSE_USE_SUNDIALS
 #include <cvode/cvode.h>
 #include <nvector/nvector_serial.h>
 #include <sunmatrix/sunmatrix_dense.h>
 #include <sunlinsol/sunlinsol_dense.h>
 #include <sundials/sundials_types.h>
 
 // SUNDIALS 7.x changed realtype to sunrealtype
 #if SUNDIALS_VERSION_MAJOR >= 7
 using realtype = sunrealtype;
 #endif
 
 // SUN_COMM_NULL may not be defined in all configurations
 #ifndef SUN_COMM_NULL
 #define SUN_COMM_NULL 0
 #endif
 
 #ifdef DFUSE_USE_CUDA
 #include <nvector/nvector_cuda.h>
 #include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
 #endif
 #endif
 
 #include <cmath>
 #include <algorithm>
 #include <memory>
 #include <stdexcept>
 
 namespace dfuse {
 namespace solver {
 
 // ============================================================================
 // SOLVER CONFIGURATION
 // ============================================================================
 
 /**
  * @brief Solver method selection
  */
 enum class SolverMethod {
     EXPLICIT_EULER,      ///< Simple forward Euler (fast but may be unstable)
     IMPLICIT_EULER,      ///< Backward Euler with Newton iteration
     SUNDIALS_ADAMS,      ///< SUNDIALS CVODE Adams-Moulton (non-stiff)
     SUNDIALS_BDF         ///< SUNDIALS CVODE BDF (stiff systems)
 };
 
 /**
  * @brief Solver configuration options
  */
 struct SolverConfig {
     SolverMethod method = SolverMethod::IMPLICIT_EULER;
     
     // Tolerance settings
     Real rel_tol = 0.1;           ///< Relative tolerance (fraction of state change allowed)
     Real abs_tol = 1e-6;          ///< Absolute tolerance
     
     // Adaptive stepping
     bool adaptive = true;
     Real max_step = 1.0;          ///< Maximum step size (days)
     Real min_step = 1e-6;         ///< Minimum step size (days)
     int max_substeps = 100;       ///< Maximum substeps per timestep
     
     // Newton iteration (for implicit methods)
     int max_newton_iter = 10;
     Real newton_tol = 1e-8;
     
     // SUNDIALS specific
     int max_order = 5;            ///< Maximum BDF/Adams order
     int max_num_steps = 500;      ///< Max internal steps per call
 };
 
 // ============================================================================
 // EXPLICIT EULER SOLVER (Simple fallback)
 // ============================================================================
 
 /**
  * @brief Simple explicit Euler with adaptive substepping
  */
 class ExplicitEulerSolver {
 public:
     ExplicitEulerSolver(const SolverConfig& config = SolverConfig())
         : config_(config) {}
     
     /**
      * @brief Integrate state from t to t+dt
      */
     void solve(
         State& state,
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& model_config,
         Real dt,
         Flux& flux_out
     ) {
         // Handle snow module once at the beginning (not in substeps)
         Real throughfall;
         if (model_config.enable_snow) {
             Real rain, melt, SWE_new;
             physics::compute_snow(forcing.precip, forcing.temp, state.SWE, params,
                                   rain, melt, SWE_new);
             state.SWE = SWE_new;
             throughfall = rain + melt;
             flux_out.rain = rain;
             flux_out.melt = melt;
         } else {
             throughfall = forcing.precip;
             flux_out.rain = forcing.precip;
             flux_out.melt = Real(0);
         }
         flux_out.throughfall = throughfall;
         
         Real t = 0;
         Real remaining = dt;
         
         int substeps = 0;
         while (remaining > config_.min_step && substeps < config_.max_substeps) {
             // Compute safe step size based on flux magnitudes
             Real h = std::min(remaining, config_.max_step);
             
             // Compute fluxes at current state (using throughfall from snow)
             Flux flux;
             compute_bucket_fluxes(state, throughfall, forcing.pet, params, model_config, flux);
             
             // Estimate maximum rate of change
             Real max_rate = estimate_max_rate(state, flux, params);
             if (max_rate > 0) {
                 Real safe_h = config_.rel_tol * params.S1_max / max_rate;
                 h = std::min(h, safe_h);
             }
             h = std::max(h, config_.min_step);
             h = std::min(h, remaining);
             
             // Forward Euler update
             update_state_euler(state, flux, params, model_config, h);
             
             remaining -= h;
             substeps++;
             
             // Copy bucket fluxes to output, keep snow fluxes from earlier
             flux.rain = flux_out.rain;
             flux.melt = flux_out.melt;
             flux.throughfall = throughfall;
             flux_out = flux;
         }
         
         // Ensure non-negative states
         enforce_constraints(state, params);
     }
     
 private:
     SolverConfig config_;
     
     void compute_bucket_fluxes(
         State& state,  // Made mutable to sync S1_F
         Real throughfall,
         Real pet,
         const Parameters& params,
         const ModelConfig& config,
         Flux& flux
     ) {
         using namespace physics;
         
         // CRITICAL FIX: For SINGLE_STATE architecture, compute S1_F from S1
         // This is needed because interflow uses S1_F, but in SINGLE_STATE
         // only S1 is tracked as the state variable. S1_F = excess above S1_T_max.
         if (config.upper_arch == UpperLayerArch::SINGLE_STATE) {
             Real excess = state.S1 - params.S1_T_max;
             state.S1_F = (excess > Real(0)) ? excess : Real(0);
         }
         
         flux.throughfall = throughfall;
         
         // Surface runoff
         compute_surface_runoff(throughfall, state, params, config,
                                flux.Ac, flux.qsx);
         
         // Baseflow
         compute_baseflow(state, params, config, flux.qb, flux.qb_A, flux.qb_B);
         
         // Percolation
         Real q0 = flux.qb;
         flux.q12 = compute_percolation(state, params, config, q0);
         
         // Interflow
         flux.qif = compute_interflow(state, params, config);
         
         // Evaporation
         compute_evaporation(pet, state, params, config, flux.e1, flux.e2);
         
         // Overflow
         Real infiltration = throughfall - flux.qsx;
         compute_overflow(infiltration, flux.q12, state, params, config,
                          flux.qurof, flux.qutof, flux.qufof,
                          flux.qstof, flux.qsfof, flux.qsfofa, flux.qsfofb);
         
         flux.compute_totals();
     }
     
     void compute_fluxes(
         State& state,  // Made mutable for SWE update
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& config,
         Flux& flux
     ) {
         using namespace physics;
         
         // Snow - update SWE directly
         Real rain, melt, SWE_new;
         if (config.enable_snow) {
             compute_snow(forcing.precip, forcing.temp, state.SWE, params,
                          rain, melt, SWE_new);
             state.SWE = SWE_new;  // Update snow state
         } else {
             rain = forcing.precip;
             melt = Real(0);
         }
         flux.rain = rain;
         flux.melt = melt;
         flux.throughfall = rain + melt;
         
         // Surface runoff
         compute_surface_runoff(flux.throughfall, state, params, config,
                                flux.Ac, flux.qsx);
         
         // Baseflow
         compute_baseflow(state, params, config, flux.qb, flux.qb_A, flux.qb_B);
         
         // Percolation
         Real q0 = flux.qb;
         flux.q12 = compute_percolation(state, params, config, q0);
         
         // Interflow
         flux.qif = compute_interflow(state, params, config);
         
         // Evaporation
         compute_evaporation(forcing.pet, state, params, config, flux.e1, flux.e2);
         
         // Overflow
         Real infiltration = flux.throughfall - flux.qsx;
         compute_overflow(infiltration, flux.q12, state, params, config,
                          flux.qurof, flux.qutof, flux.qufof,
                          flux.qstof, flux.qsfof, flux.qsfofa, flux.qsfofb);
         
         flux.compute_totals();
     }
     
     Real estimate_max_rate(const State& state, const Flux& flux, const Parameters& params) {
         // Estimate maximum rate of state change
         Real infiltration = flux.throughfall - flux.qsx;
         Real rate_S1 = std::abs(infiltration - flux.e1 - flux.q12 - flux.qif);
         Real rate_S2 = std::abs(flux.q12 - flux.qb);
         return std::max(rate_S1, rate_S2);
     }
     
     void update_state_euler(
         State& state,
         const Flux& flux,
         const Parameters& params,
         const ModelConfig& config,
         Real h
     ) {
         Real infiltration = flux.throughfall - flux.qsx;
         
         switch (config.upper_arch) {
             case UpperLayerArch::SINGLE_STATE:
                 state.S1 += (infiltration - flux.e1 - flux.q12 - flux.qif - flux.qufof) * h;
                 break;
             case UpperLayerArch::TENSION_FREE:
                 state.S1_T += (infiltration - flux.e1 - flux.qutof) * h;
                 state.S1_F += (flux.qutof - flux.q12 - flux.qif - flux.qufof) * h;
                 break;
             case UpperLayerArch::TENSION2_FREE:
                 state.S1_TA += (infiltration - flux.e1_A - flux.qurof) * h;
                 state.S1_TB += (flux.qurof - flux.e1_B - flux.qutof) * h;
                 state.S1_F += (flux.qutof - flux.q12 - flux.qif - flux.qufof) * h;
                 break;
         }
         
         switch (config.lower_arch) {
             case LowerLayerArch::SINGLE_NOEVAP:
                 state.S2 += (flux.q12 - flux.qb - flux.qsfof) * h;
                 break;
             case LowerLayerArch::SINGLE_EVAP:
                 state.S2 += (flux.q12 - flux.qb - flux.e2 - flux.qsfof) * h;
                 break;
             case LowerLayerArch::TENSION_2RESERV:
                 state.S2_T += (params.kappa * flux.q12 - flux.e2 - flux.qstof) * h;
                 Real to_free = (1 - params.kappa) * flux.q12 / 2 + flux.qstof / 2;
                 state.S2_FA += (to_free - flux.qb_A - flux.qsfofa) * h;
                 state.S2_FB += (to_free - flux.qb_B - flux.qsfofb) * h;
                 break;
         }
         
         state.sync_derived(config);
     }
     
     void enforce_constraints(State& state, const Parameters& params) {
         state.S1 = std::max(Real(0), state.S1);
         state.S1_T = std::max(Real(0), state.S1_T);
         state.S1_TA = std::max(Real(0), state.S1_TA);
         state.S1_TB = std::max(Real(0), state.S1_TB);
         state.S1_F = std::max(Real(0), state.S1_F);
         state.S2 = std::max(Real(0), state.S2);
         state.S2_T = std::max(Real(0), state.S2_T);
         state.S2_FA = std::max(Real(0), state.S2_FA);
         state.S2_FB = std::max(Real(0), state.S2_FB);
         state.SWE = std::max(Real(0), state.SWE);
     }
 };
 
 // ============================================================================
 // IMPLICIT EULER SOLVER (Newton-Raphson)
 // ============================================================================
 
 /**
  * @brief Implicit Euler solver with Newton-Raphson iteration
  * 
  * Implements the approach from Clark & Kavetski (2010).
  * More stable for stiff systems but requires Jacobian computation.
  */
 class ImplicitEulerSolver {
 public:
     ImplicitEulerSolver(const SolverConfig& config = SolverConfig())
         : config_(config) {}
     
     void solve(
         State& state,
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& model_config,
         Real dt,
         Flux& flux_out
     ) {
         const int n_states = get_num_states(model_config);
         
         // Current state as vector
         Real y[MAX_TOTAL_STATES];
         Real y_new[MAX_TOTAL_STATES];
         Real f[MAX_TOTAL_STATES];
         Real J[MAX_TOTAL_STATES * MAX_TOTAL_STATES];
         Real delta[MAX_TOTAL_STATES];
         
         state.to_array(y, model_config);
         
         // Initial guess: explicit Euler
         compute_rhs(y, forcing, params, model_config, f);
         for (int i = 0; i < n_states; ++i) {
             y_new[i] = y[i] + dt * f[i];
         }
         
         // Newton iteration
         for (int iter = 0; iter < config_.max_newton_iter; ++iter) {
             // Compute residual: G(y_new) = y_new - y - dt*f(y_new)
             compute_rhs(y_new, forcing, params, model_config, f);
             
             Real residual_norm = 0;
             for (int i = 0; i < n_states; ++i) {
                 delta[i] = y_new[i] - y[i] - dt * f[i];  // G(y_new)
                 residual_norm += delta[i] * delta[i];
             }
             residual_norm = std::sqrt(residual_norm);
             
             if (residual_norm < config_.newton_tol) break;
             
             // Compute Jacobian: dG/dy = I - dt * df/dy
             compute_jacobian(y_new, forcing, params, model_config, dt, J);
             
             // Solve J * correction = -G (simple Gaussian elimination)
             solve_linear_system(J, delta, n_states);
             
             // Update: y_new = y_new - correction
             for (int i = 0; i < n_states; ++i) {
                 y_new[i] -= delta[i];
                 y_new[i] = std::max(Real(0), y_new[i]);  // Non-negativity
             }
         }
         
         // Update state
         state.from_array(y_new, model_config);
         
         // Compute final fluxes
         ExplicitEulerSolver euler_solver;
         // Use explicit solver's compute_fluxes via the flux computation
         Flux flux;
         compute_final_flux(state, forcing, params, model_config, flux);
         flux_out = flux;
     }
     
 private:
     SolverConfig config_;
     
     int get_num_states(const ModelConfig& config) {
         int n = 0;
         switch (config.upper_arch) {
             case UpperLayerArch::SINGLE_STATE: n += 1; break;
             case UpperLayerArch::TENSION_FREE: n += 2; break;
             case UpperLayerArch::TENSION2_FREE: n += 3; break;
         }
         switch (config.lower_arch) {
             case LowerLayerArch::SINGLE_NOEVAP:
             case LowerLayerArch::SINGLE_EVAP: n += 1; break;
             case LowerLayerArch::TENSION_2RESERV: n += 3; break;
         }
         if (config.enable_snow) n += 1;
         return n;
     }
     
     void compute_rhs(
         const Real* y,
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& config,
         Real* f
     ) {
         // Reconstruct state from array
         State state;
         state.from_array(y, config);
         
         // Compute fluxes
         Flux flux;
         compute_final_flux(state, forcing, params, config, flux);
         
         // Compute derivatives
         physics::compute_derivatives(state, flux, params, config, f);
     }
     
     void compute_jacobian(
         const Real* y,
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& config,
         Real dt,
         Real* J
     ) {
         const int n = get_num_states(config);
         const Real eps = 1e-7;
         
         Real f0[MAX_TOTAL_STATES];
         Real f1[MAX_TOTAL_STATES];
         Real y_pert[MAX_TOTAL_STATES];
         
         compute_rhs(y, forcing, params, config, f0);
         
         for (int j = 0; j < n; ++j) {
             // Perturb state j
             for (int i = 0; i < n; ++i) y_pert[i] = y[i];
             Real h = eps * std::max(std::abs(y[j]), Real(1));
             y_pert[j] += h;
             
             compute_rhs(y_pert, forcing, params, config, f1);
             
             // dG/dy_j = I_j - dt * df/dy_j
             for (int i = 0; i < n; ++i) {
                 Real df_dy = (f1[i] - f0[i]) / h;
                 J[i * n + j] = (i == j ? 1 : 0) - dt * df_dy;
             }
         }
     }
     
     void solve_linear_system(Real* A, Real* b, int n) {
         // Simple Gaussian elimination with partial pivoting
         for (int k = 0; k < n; ++k) {
             // Find pivot
             int max_row = k;
             Real max_val = std::abs(A[k * n + k]);
             for (int i = k + 1; i < n; ++i) {
                 if (std::abs(A[i * n + k]) > max_val) {
                     max_val = std::abs(A[i * n + k]);
                     max_row = i;
                 }
             }
             
             // Swap rows
             if (max_row != k) {
                 for (int j = k; j < n; ++j) {
                     std::swap(A[k * n + j], A[max_row * n + j]);
                 }
                 std::swap(b[k], b[max_row]);
             }
             
             // Eliminate
             for (int i = k + 1; i < n; ++i) {
                 Real factor = A[i * n + k] / A[k * n + k];
                 for (int j = k; j < n; ++j) {
                     A[i * n + j] -= factor * A[k * n + j];
                 }
                 b[i] -= factor * b[k];
             }
         }
         
         // Back substitution
         for (int i = n - 1; i >= 0; --i) {
             for (int j = i + 1; j < n; ++j) {
                 b[i] -= A[i * n + j] * b[j];
             }
             b[i] /= A[i * n + i];
         }
     }
     
     void compute_final_flux(
         const State& state,
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& config,
         Flux& flux
     ) {
         using namespace physics;
         
         Real rain, melt, SWE_new;
         compute_snow(forcing.precip, forcing.temp, state.SWE, params,
                      rain, melt, SWE_new);
         flux.rain = rain;
         flux.melt = melt;
         flux.throughfall = rain + melt;
         
         compute_surface_runoff(flux.throughfall, state, params, config,
                                flux.Ac, flux.qsx);
         compute_baseflow(state, params, config, flux.qb, flux.qb_A, flux.qb_B);
         flux.q12 = compute_percolation(state, params, config, flux.qb);
         flux.qif = compute_interflow(state, params, config);
         compute_evaporation(forcing.pet, state, params, config, flux.e1, flux.e2);
         
         Real infiltration = flux.throughfall - flux.qsx;
         compute_overflow(infiltration, flux.q12, state, params, config,
                          flux.qurof, flux.qutof, flux.qufof,
                          flux.qstof, flux.qsfof, flux.qsfofa, flux.qsfofb);
         flux.compute_totals();
     }
 };
 
 // ============================================================================
 // SUNDIALS CVODE SOLVER
 // ============================================================================
 
 #ifdef DFUSE_USE_SUNDIALS
 
 /**
  * @brief User data for SUNDIALS callbacks
  */
 struct SundialsUserData {
     const Forcing* forcing;
     const Parameters* params;
     const ModelConfig* config;
     int n_states;
 };
 
 /**
  * @brief RHS function for SUNDIALS CVODE
  * 
  * Note: SUNDIALS uses double precision (realtype = double), but cFUSE uses
  * single precision (Real = float). We need to convert between types.
  */
 static int sundials_rhs(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
     SundialsUserData* data = static_cast<SundialsUserData*>(user_data);
     
     // Get SUNDIALS arrays (double precision)
     realtype* y_arr = N_VGetArrayPointer(y);
     realtype* ydot_arr = N_VGetArrayPointer(ydot);
     
     // Convert from double to float for state
     Real y_float[MAX_TOTAL_STATES];
     for (int i = 0; i < data->n_states; ++i) {
         y_float[i] = static_cast<Real>(y_arr[i]);
     }
     
     // Reconstruct state from float array
     State state;
     state.from_array(y_float, *data->config);
     
     // Sync free storage for single-state architecture if needed
     if (data->config->upper_arch == UpperLayerArch::SINGLE_STATE) {
         Real excess = state.S1 - data->params->S1_T_max;
         state.S1_F = (excess > Real(0)) ? excess : Real(0);
     }    
 
     // Compute fluxes
     Flux flux;
     using namespace physics;
     
     // Store snow derivative to apply later
     Real swe_rate = Real(0);
 
     // --- UPDATED SNOW LOGIC ---
     // If snow is enabled, we compute it. If not (e.g. elevation bands handled externally),
     // we bypass snow and assume precipitation is liquid.
     if (data->config->enable_snow) {
         Real rain, melt, SWE_new;
         compute_snow(data->forcing->precip, data->forcing->temp, state.SWE, 
                      *data->params, rain, melt, SWE_new);
         
         flux.throughfall = rain + melt;
         
         // Calculate derivative: dSWE/dt ≈ (SWE_new - SWE_old) / dt (dt=1.0)
         // This is approximate but consistent with the operator splitting
         swe_rate = SWE_new - state.SWE;
     } else {
         // Snow disabled (or handled externally via elevation bands)
         // Pass precipitation directly through to soil
         flux.throughfall = data->forcing->precip;
         swe_rate = Real(0);
     }
     // --------------------------
     
     compute_surface_runoff(flux.throughfall, state, *data->params, *data->config,
                            flux.Ac, flux.qsx);
     compute_baseflow(state, *data->params, *data->config, flux.qb, flux.qb_A, flux.qb_B);
     flux.q12 = compute_percolation(state, *data->params, *data->config, flux.qb);
     flux.qif = compute_interflow(state, *data->params, *data->config);
     compute_evaporation(data->forcing->pet, state, *data->params, *data->config, 
                         flux.e1, flux.e2);
     
     Real infiltration = flux.throughfall - flux.qsx;
     compute_overflow(infiltration, flux.q12, state, *data->params, *data->config,
                      flux.qurof, flux.qutof, flux.qufof,
                      flux.qstof, flux.qsfof, flux.qsfofa, flux.qsfofb);
     
     // Compute derivatives to float buffer (handles soil moisture states)
     Real ydot_float[MAX_TOTAL_STATES];
     compute_derivatives(state, flux, *data->params, *data->config, ydot_float);
     
     // Convert back to double for SUNDIALS
     for (int i = 0; i < data->n_states; ++i) {
         ydot_arr[i] = static_cast<realtype>(ydot_float[i]);
     }
 
     // Manually set SWE derivative if snow is enabled
     // (compute_derivatives does not calculate dSWE/dt)
     if (data->config->enable_snow) {
         int swe_idx = data->n_states - 1;
         ydot_arr[swe_idx] = static_cast<realtype>(swe_rate);
     }
     
     return 0;
 }
 
 /**
  * @brief SUNDIALS CVODE solver wrapper
  */
 class SundialsSolver {
 public:
     SundialsSolver(const SolverConfig& config = SolverConfig())
         : config_(config), cvode_mem_(nullptr), y_(nullptr), 
           A_(nullptr), LS_(nullptr) {}
     
     ~SundialsSolver() {
         cleanup();
     }
     
     void initialize(const ModelConfig& model_config) {
         cleanup();
         
         n_states_ = get_num_states(model_config);
         
         // Create SUNDIALS context (SUNDIALS 6.0+)
         #if SUNDIALS_VERSION_MAJOR >= 6
         // For SUNDIALS 7.x, SUNComm is int type, not a pointer
         // SUN_COMM_NULL is defined as 0 for non-MPI builds
         SUNContext_Create(SUN_COMM_NULL, &sunctx_);
         y_ = N_VNew_Serial(n_states_, sunctx_);
         #else
         y_ = N_VNew_Serial(n_states_);
         #endif
         
         // Create CVODE solver
         int lmm = (config_.method == SolverMethod::SUNDIALS_BDF) ? CV_BDF : CV_ADAMS;
         
         #if SUNDIALS_VERSION_MAJOR >= 6
         cvode_mem_ = CVodeCreate(lmm, sunctx_);
         #else
         cvode_mem_ = CVodeCreate(lmm);
         #endif
         
         // Initialize
         CVodeInit(cvode_mem_, sundials_rhs, 0.0, y_);
         CVodeSStolerances(cvode_mem_, config_.rel_tol, config_.abs_tol);
         CVodeSetMaxNumSteps(cvode_mem_, config_.max_num_steps);
         CVodeSetMaxOrd(cvode_mem_, config_.max_order);
         
         // Create matrix and linear solver
         #if SUNDIALS_VERSION_MAJOR >= 6
         A_ = SUNDenseMatrix(n_states_, n_states_, sunctx_);
         LS_ = SUNLinSol_Dense(y_, A_, sunctx_);
         #else
         A_ = SUNDenseMatrix(n_states_, n_states_);
         LS_ = SUNLinSol_Dense(y_, A_);
         #endif
         
         CVodeSetLinearSolver(cvode_mem_, LS_, A_);
         
         initialized_ = true;
     }
     
     void solve(
         State& state,
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& model_config,
         Real dt,
         Flux& flux_out
     ) {
         if (!initialized_) {
             initialize(model_config);
         }
         
         // Set user data (include n_states for RHS function)
         SundialsUserData user_data = {&forcing, &params, &model_config, n_states_};
         CVodeSetUserData(cvode_mem_, &user_data);
         
         // Convert state to float array, then to SUNDIALS double array
         Real y_float[MAX_TOTAL_STATES];
         state.to_array(y_float, model_config);
         
         realtype* y_arr = N_VGetArrayPointer(y_);
         for (int i = 0; i < n_states_; ++i) {
             y_arr[i] = static_cast<realtype>(y_float[i]);
         }
         
         // Reinitialize at t=0
         CVodeReInit(cvode_mem_, 0.0, y_);
         
         // Integrate to t=dt
         realtype t_out;
         int flag = CVode(cvode_mem_, static_cast<realtype>(dt), y_, &t_out, CV_NORMAL);
         
         if (flag < 0) {
             throw std::runtime_error("SUNDIALS CVODE solver failed");
         }
         
         // Convert SUNDIALS double array back to float, then update state
         for (int i = 0; i < n_states_; ++i) {
             y_float[i] = static_cast<Real>(y_arr[i]);
         }
         state.from_array(y_float, model_config);
         
         // Compute final fluxes (for output)
         compute_flux_output(state, forcing, params, model_config, flux_out);
     }
     
 private:
     SolverConfig config_;
     void* cvode_mem_;
     N_Vector y_;
     SUNMatrix A_;
     SUNLinearSolver LS_;
     int n_states_;
     bool initialized_ = false;
     
     #if SUNDIALS_VERSION_MAJOR >= 6
     SUNContext sunctx_;
     #endif
     
     void cleanup() {
         if (LS_) SUNLinSolFree(LS_);
         if (A_) SUNMatDestroy(A_);
         if (y_) N_VDestroy(y_);
         if (cvode_mem_) CVodeFree(&cvode_mem_);
         #if SUNDIALS_VERSION_MAJOR >= 6
         if (initialized_) SUNContext_Free(&sunctx_);
         #endif
         initialized_ = false;
     }
     
     int get_num_states(const ModelConfig& config) {
         int n = 0;
         switch (config.upper_arch) {
             case UpperLayerArch::SINGLE_STATE: n += 1; break;
             case UpperLayerArch::TENSION_FREE: n += 2; break;
             case UpperLayerArch::TENSION2_FREE: n += 3; break;
         }
         switch (config.lower_arch) {
             case LowerLayerArch::SINGLE_NOEVAP:
             case LowerLayerArch::SINGLE_EVAP: n += 1; break;
             case LowerLayerArch::TENSION_2RESERV: n += 3; break;
         }
         if (config.enable_snow) n += 1;
         return n;
     }
     
     void compute_flux_output(
         State& state,
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& config,
         Flux& flux
     ) {
         using namespace physics;
         
         // Sync derived state for reporting
         if (config.upper_arch == UpperLayerArch::SINGLE_STATE) {
             Real excess = state.S1 - params.S1_T_max;
             state.S1_F = (excess > Real(0)) ? excess : Real(0);
         }
 
         // --- UPDATED SNOW LOGIC (Reporting) ---
         // If snow enabled, process snow physics.
         // If disabled, bypass and pass precip through.
         if (config.enable_snow) {
             Real rain, melt, SWE_new;
             compute_snow(forcing.precip, forcing.temp, state.SWE, params,
                          rain, melt, SWE_new);
             flux.rain = rain;
             flux.melt = melt;
             flux.throughfall = rain + melt;
         } else {
             // Snow handled externally (e.g. elevation bands) or model disabled
             flux.rain = forcing.precip;
             flux.melt = Real(0);
             flux.throughfall = forcing.precip;
         }
         // --------------------------------------
         
         compute_surface_runoff(flux.throughfall, state, params, config,
                                flux.Ac, flux.qsx);
         compute_baseflow(state, params, config, flux.qb, flux.qb_A, flux.qb_B);
         flux.q12 = compute_percolation(state, params, config, flux.qb);
         flux.qif = compute_interflow(state, params, config);
         compute_evaporation(forcing.pet, state, params, config, flux.e1, flux.e2);
         
         Real infiltration = flux.throughfall - flux.qsx;
         compute_overflow(infiltration, flux.q12, state, params, config,
                          flux.qurof, flux.qutof, flux.qufof,
                          flux.qstof, flux.qsfof, flux.qsfofa, flux.qsfofb);
         flux.compute_totals();
     }
 };
 
 #endif // DFUSE_USE_SUNDIALS
 
 // ============================================================================
 // UNIFIED SOLVER INTERFACE
 // ============================================================================
 
 /**
  * @brief Unified solver interface supporting multiple backends
  */
 class Solver {
 public:
     Solver(const SolverConfig& config = SolverConfig())
         : config_(config),
           euler_solver_(config),
           implicit_solver_(config)
     {
         #ifdef DFUSE_USE_SUNDIALS
         sundials_solver_ = std::make_unique<SundialsSolver>(config);
         #endif
     }
     
     void solve(
         State& state,
         const Forcing& forcing,
         const Parameters& params,
         const ModelConfig& model_config,
         Real dt,
         Flux& flux_out
     ) {
         switch (config_.method) {
             case SolverMethod::EXPLICIT_EULER:
                 euler_solver_.solve(state, forcing, params, model_config, dt, flux_out);
                 break;
                 
             case SolverMethod::IMPLICIT_EULER:
                 implicit_solver_.solve(state, forcing, params, model_config, dt, flux_out);
                 break;
                 
             #ifdef DFUSE_USE_SUNDIALS
             case SolverMethod::SUNDIALS_ADAMS:
             case SolverMethod::SUNDIALS_BDF:
                 sundials_solver_->solve(state, forcing, params, model_config, dt, flux_out);
                 break;
             #else
             case SolverMethod::SUNDIALS_ADAMS:
             case SolverMethod::SUNDIALS_BDF:
                 throw std::runtime_error("SUNDIALS not available. Compile with -DDFUSE_USE_SUNDIALS=ON");
             #endif
         }
     }
     
 private:
     SolverConfig config_;
     ExplicitEulerSolver euler_solver_;
     ImplicitEulerSolver implicit_solver_;
     
     #ifdef DFUSE_USE_SUNDIALS
     std::unique_ptr<SundialsSolver> sundials_solver_;
     #endif
 };
 
 } // namespace solver
 } // namespace dfuse