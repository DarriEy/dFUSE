/**
 * @file physics.hpp
 * @brief Core physics computations for cFUSE
 * 
 * Differentiable Framework for Understanding Structural Errors (cFUSE)
 * Based on Clark et al. (2008) WRR, doi:10.1029/2007WR006735
 * 
 * This file implements all flux equations as device-callable functions.
 * Each function corresponds to a specific equation in Clark et al. (2008).
 * 
 * Key design principles:
 * - All functions are differentiable (smooth approximations for discontinuities)
 * - Functions can be called from both CPU and GPU
 * - Template dispatch enables compile-time physics selection
 * 
 * Updates:
 * - Redefined safe_pow to use exp(y*log(x)) formulation for better AD stability.
 * - Increased epsilon slightly to 1e-5.
 */

 #pragma once

 #include "config.hpp"
 #include "state.hpp"
 #include <cmath>
 
 namespace dfuse {
 namespace physics {
 
 // ============================================================================
 // UTILITIES
 // ============================================================================
 
 // Safe power function for AD (BRANCH-FREE for Enzyme AD)
 // Using exp(b * log(a)) is often more robust for automatic differentiation
 // than direct pow(a,b) when a is near zero.
 // Uses smooth soft-clamping via sqrt for completely branch-free operation.
 DFUSE_HOST_DEVICE inline Real safe_pow(Real base, Real exponent) {
     constexpr Real eps = Real(1e-5);
     
     // Branch-free smooth clamping using quadratic smoothing
     // safe_base ≈ base when base >> eps, ≈ eps when base << eps
     // smooth_max(base, eps) = 0.5*(base + eps + sqrt((base-eps)^2 + 4*k^2))
     constexpr Real k = Real(1e-6);  // Smoothing width
     Real diff = base - eps;
     Real smooth_abs = std::sqrt(diff * diff + Real(4) * k * k);
     Real safe_base = Real(0.5) * (base + eps + smooth_abs);
     
     return std::exp(exponent * std::log(safe_base));
 }
 
 // ============================================================================
 // SMOOTH THRESHOLD FUNCTIONS (Kavetski & Kuczera 2007)
 // ============================================================================
 
 /**
  * @brief Logistic smoothing function for bucket overflow (BRANCH-FREE for Enzyme AD)
  * 
  * Implements Clark et al. (2008) equation (12h):
  *   F(S, Smax, w) = 1 / (1 + exp(-(S - Smax - w*e) / w))
  * 
  * This provides a differentiable approximation to the step function
  * that triggers overflow when storage exceeds capacity.
  * Uses tanh identity for branch-free computation.
  * 
  * @param S Current storage
  * @param S_max Maximum storage capacity
  * @param w Smoothing width (typically r * S_max where r ~ 0.01)
  * @return Overflow fraction [0, 1]
  */
 DFUSE_HOST_DEVICE inline Real logistic_overflow(Real S, Real S_max, Real w) {
     constexpr Real e_mult = Real(5);  // Ensures S < S_max always
     // Protect against division by zero - BRANCH-FREE using smooth_max
     constexpr Real eps_w = Real(1e-6);
     constexpr Real k = Real(1e-8);
     Real diff = w - eps_w;
     Real smooth_abs = std::sqrt(diff * diff + Real(4) * k * k);
     Real safe_w = Real(0.5) * (w + eps_w + smooth_abs);
     
     Real x = (S - S_max - safe_w * e_mult) / safe_w;
     // Branch-free sigmoid using tanh identity
     return Real(0.5) * (Real(1) + std::tanh(x * Real(0.5)));
 }
 
 /**
  * @brief Smooth minimum function (BRANCH-FREE for Enzyme AD)
  * 
  * Differentiable approximation to min(a, b) using softmin.
  * Uses the identity: softmin(a,b) = -softmax(-a,-b)
  * softmax(a,b) ≈ max(a,b) + softplus(min(a,b) - max(a,b))
  * 
  * Simplified: softmin(a,b) = 0.5*(a+b) - 0.5*sqrt((a-b)^2 + 4*k^2)
  * This is smooth, branch-free, and well-behaved for AD.
  */
 DFUSE_HOST_DEVICE inline Real smooth_min(Real a, Real b, Real k = Real(0.01)) {
     // Branch-free smooth minimum using quadratic smoothing
     Real diff = a - b;
     Real smooth_abs = std::sqrt(diff * diff + Real(4) * k * k);
     return Real(0.5) * (a + b - smooth_abs);
 }
 
 /**
  * @brief Smooth maximum function (BRANCH-FREE for Enzyme AD)
  * 
  * Differentiable approximation to max(a, b).
  * Uses: softmax(a,b) = 0.5*(a+b) + 0.5*sqrt((a-b)^2 + 4*k^2)
  */
 DFUSE_HOST_DEVICE inline Real smooth_max(Real a, Real b, Real k = Real(0.01)) {
     // Branch-free smooth maximum using quadratic smoothing
     Real diff = a - b;
     Real smooth_abs = std::sqrt(diff * diff + Real(4) * k * k);
     return Real(0.5) * (a + b + smooth_abs);
 }
 
 /**
  * @brief Smooth clamp to [0, max]
  */
 DFUSE_HOST_DEVICE inline Real smooth_clamp(Real x, Real max_val, Real k = Real(0.01)) {
     return smooth_min(smooth_max(x, Real(0), k), max_val, k);
 }
 
 /**
  * @brief Smooth clamp to [min, max]
  */
 DFUSE_HOST_DEVICE inline Real smooth_clamp(Real x, Real min_val, Real max_val, Real k) {
     return smooth_min(smooth_max(x, min_val, k), max_val, k);
 }
 
 /**
  * @brief Smooth sigmoid function (BRANCH-FREE for Enzyme AD)
  * 
  * Differentiable approximation to step function.
  * smooth_sigmoid(x, k) ≈ 1 for x >> k, ≈ 0 for x << -k
  * 
  * Uses tanh identity: sigmoid(z) = 0.5 * (1 + tanh(z/2))
  * This is numerically stable and completely branch-free.
  */
 DFUSE_HOST_DEVICE inline Real smooth_sigmoid(Real x, Real k = Real(1)) {
     Real z = x / k;
     // Branch-free sigmoid using tanh identity
     // sigmoid(z) = 0.5 * (1 + tanh(z/2))
     return Real(0.5) * (Real(1) + std::tanh(z * Real(0.5)));
 }
 
 /**
  * @brief Smooth overflow computation
  * 
  * Computes overflow when storage exceeds capacity, with smooth transition.
  */
 DFUSE_HOST_DEVICE inline Real smooth_overflow(Real S, Real S_max, Real inflow, Real w) {
     Real excess = S + inflow - S_max;
     Real overflow_frac = logistic_overflow(S + inflow, S_max, w);
     return smooth_max(excess * overflow_frac, Real(0), Real(0.001));
 }
 
 // ============================================================================
 // SNOW MODULE (Extension from Henn et al. 2015 / SNOW-17)
 // ============================================================================
 
 /**
  * @brief Compute seasonal melt factor following SNOW-17 convention
  * 
  * Melt factor varies sinusoidally between MFMIN (Dec 21) and MFMAX (June 21).
  * This matches the NWS SNOW-17 model approach.
  * 
  * MF = (MFMAX + MFMIN)/2 + (MFMAX - MFMIN)/2 * sin(2π * (doy - 81) / 365)
  * 
  * @param day_of_year Day of year (1-365/366)
  * @param MFMAX Maximum melt factor on June 21 (mm/°C/day)
  * @param MFMIN Minimum melt factor on Dec 21 (mm/°C/day)
  * @return Melt factor for the given day (mm/°C/day)
  */
 DFUSE_HOST_DEVICE inline Real compute_seasonal_melt_factor(
     int day_of_year, Real MFMAX, Real MFMIN
 ) {
     constexpr Real TWO_PI = Real(6.283185307179586);
     // Day 81 = ~March 21 (spring equinox), sin=0
     // Day 172 = ~June 21 (summer solstice), sin=1, MF=MFMAX
     // Day 355 = ~Dec 21 (winter solstice), sin=-1, MF=MFMIN
     Real phase = TWO_PI * Real(day_of_year - 81) / Real(365);
     Real MF_avg = (MFMAX + MFMIN) / Real(2);
     Real MF_amp = (MFMAX - MFMIN) / Real(2);
     return MF_avg + MF_amp * std::sin(phase);
 }
 
 /**
  * @brief Snow accumulation and melt (single layer)
  * 
  * Simple temperature-index snow model.
  * 
  * @param precip Total precipitation (mm/day)
  * @param temp Air temperature (°C)
  * @param SWE Current snow water equivalent (mm)
  * @param params Model parameters
  * @param[out] rain Rainfall component (mm/day)
  * @param[out] melt Snowmelt (mm/day)
  * @param[out] SWE_new Updated SWE (mm)
  * @param day_of_year Day of year for seasonal melt factor (1-365, default 1=no seasonal variation)
  */
 DFUSE_HOST_DEVICE inline void compute_snow(
     Real precip, Real temp, Real SWE,
     const Parameters& params,
     Real& rain, Real& melt, Real& SWE_new,
     int day_of_year = 0
 ) {
     // Smoothing parameter for AD-compatible operations
     constexpr Real smooth_k = Real(0.01);
     
     // Smooth rain-snow partition using numerically stable sigmoid
     // FIXED: Use smooth_sigmoid which handles exp overflow/underflow
     // snow_frac approaches 1 when temp << T_rain (cold), 0 when temp >> T_rain (warm)
     // Width parameter 1.0 gives gradual transition (k=1 in original formula)
     Real snow_frac = smooth_sigmoid(params.T_rain - temp, Real(1.0));
     Real snow = precip * snow_frac;
     rain = precip * (Real(1) - snow_frac);
     
     // Compute melt factor (seasonal if MFMAX/MFMIN are set, else use melt_rate)
     Real MF;
     if (params.MFMAX > Real(0) && params.MFMIN > Real(0) && day_of_year > 0) {
         MF = compute_seasonal_melt_factor(day_of_year, params.MFMAX, params.MFMIN);
     } else {
         MF = params.melt_rate;
     }
     
     // Potential melt (degree-day method) - FIXED: use smooth operations
     Real temp_above_melt = temp - params.T_melt;
     Real melt_factor = smooth_sigmoid(temp_above_melt, Real(0.5));
     Real pot_melt = MF * smooth_max(temp_above_melt, Real(0), smooth_k) * melt_factor;
     
     // Actual melt limited by available SWE - FIXED: use smooth_min/max
     Real SWE_after_snow = SWE + snow;
     melt = smooth_min(pot_melt, SWE_after_snow, smooth_k);
     melt = smooth_max(melt, Real(0), smooth_k);  // Ensure non-negative
     
     // Update SWE - FIXED: use smooth_max
     SWE_new = smooth_max(SWE_after_snow - melt, Real(0), smooth_k);
 }
 
 /**
  * @brief Snow accumulation and melt with elevation bands
  * 
  * Temperature-index snow model with elevation band discretization.
  * Matches Fortran FUSE snow module with seasonal melt factor variation.
  * 
  * @param precip Total precipitation at reference elevation (mm/day)
  * @param temp Air temperature at reference elevation (°C)
  * @param SWE_bands Current SWE per elevation band (mm) [n_bands]
  * @param n_bands Number of elevation bands
  * @param area_frac Fraction of catchment area per band [n_bands]
  * @param mean_elev Mean elevation per band (m) [n_bands]
  * @param ref_elev Reference elevation for forcing data (m)
  * @param params Model parameters (includes lapse_rate, opg, MFMAX, MFMIN)
  * @param[out] rain_eff Effective rainfall (basin average, mm/day)
  * @param[out] melt_eff Effective snowmelt (basin average, mm/day)
  * @param[out] SWE_bands_new Updated SWE per band (mm) [n_bands]
  * @param day_of_year Day of year for seasonal melt factor (1-365, default 0=no seasonal variation)
  */
 template<int MAX_BANDS = 30>
 DFUSE_HOST_DEVICE inline void compute_snow_elevation_bands(
     Real precip, Real temp,
     const Real* SWE_bands,
     int n_bands,
     const Real* area_frac,
     const Real* mean_elev,
     Real ref_elev,
     const Parameters& params,
     Real& rain_eff, Real& melt_eff,
     Real* SWE_bands_new,
     int day_of_year = 0
 ) {
     rain_eff = Real(0);
     melt_eff = Real(0);
     
     // Compute seasonal melt factor (or use constant if not specified)
     Real MF;
     if (params.MFMAX > Real(0) && params.MFMIN > Real(0) && day_of_year > 0) {
         MF = compute_seasonal_melt_factor(day_of_year, params.MFMAX, params.MFMIN);
     } else {
         MF = params.melt_rate;
     }
     
     // Smoothing parameter for AD-compatible operations
     constexpr Real smooth_k = Real(0.01);
     
     for (int b = 0; b < n_bands; ++b) {
         // Elevation difference from reference (km)
         Real elev_diff_km = (mean_elev[b] - ref_elev) / Real(1000);
         
         // Adjust temperature for elevation (lapse rate, typically -5 to -7 °C/km)
         Real temp_band = temp + params.lapse_rate * elev_diff_km;
         
         // Adjust precipitation for elevation (orographic gradient, typically 0-0.5 km-1)
         Real precip_band = precip * (Real(1) + params.opg * elev_diff_km);
         // FIXED: Use smooth_max instead of ternary for AD compatibility
         precip_band = smooth_max(precip_band, Real(0), smooth_k);
         
         // Rain-snow partition using numerically stable sigmoid
         // FIXED: Use smooth_sigmoid which handles exp overflow/underflow
         // snow_frac approaches 1 when temp_band << T_rain (cold), 0 when temp_band >> T_rain (warm)
         // Width parameter 0.2 gives sharper transition (equivalent to k=5 in original formula)
         Real snow_frac = smooth_sigmoid(params.T_rain - temp_band, Real(0.2));
         Real snow_band = precip_band * snow_frac;
         Real rain_band = precip_band * (Real(1) - snow_frac);
         
         // Potential melt (degree-day method with seasonal factor)
         // FIXED: Use smooth sigmoid ramp instead of hard threshold
         Real temp_above_melt = temp_band - params.T_melt;
         Real melt_factor = smooth_sigmoid(temp_above_melt, Real(0.5));  // Smooth transition around T_melt
         Real pot_melt = MF * smooth_max(temp_above_melt, Real(0), smooth_k) * melt_factor;
         
         // Actual melt limited by available SWE
         // FIXED: Use smooth_min instead of ternary
         Real SWE_after_snow = SWE_bands[b] + snow_band;
         Real melt_band = smooth_min(pot_melt, SWE_after_snow, smooth_k);
         // FIXED: Use smooth_max instead of ternary
         melt_band = smooth_max(melt_band, Real(0), smooth_k);
         
         // Update SWE for this band
         // FIXED: Use smooth_max instead of ternary
         SWE_bands_new[b] = smooth_max(SWE_after_snow - melt_band, Real(0), smooth_k);
         
         // Accumulate basin-average values weighted by area fraction
         rain_eff += rain_band * area_frac[b];
         melt_eff += melt_band * area_frac[b];
     }
 }
 
 // ============================================================================
 // EVAPORATION PARAMETERIZATIONS
 // ============================================================================
 
 /**
  * @brief Sequential evaporation - equations (3a), (3b)
  * 
  * Evaporative demand is first satisfied from upper layer,
  * residual demand is then satisfied from lower layer.
  */
 DFUSE_HOST_DEVICE inline void evap_sequential(
     Real pet, const State& state, const Parameters& params,
     Real& e1, Real& e2
 ) {
     // Upper layer evap (3a)
     Real S1_T_frac = smooth_min(state.S1_T, params.S1_T_max) / params.S1_T_max;
     e1 = pet * S1_T_frac;
     
     // Lower layer evap (3b) - from residual demand
     Real residual_pet = pet - e1;
     Real S2_T_frac = smooth_min(state.S2_T, params.S2_T_max) / params.S2_T_max;
     e2 = residual_pet * S2_T_frac;
 }
 
 /**
  * @brief Root-weighted evaporation - equations (3c), (3d)
  * 
  * Evaporation partitioned based on root distribution.
  */
 DFUSE_HOST_DEVICE inline void evap_root_weight(
     Real pet, const State& state, const Parameters& params,
     Real& e1, Real& e2
 ) {
     Real r2 = Real(1) - params.r1;
     
     // Upper layer (3c)
     Real S1_T_frac = smooth_min(state.S1_T, params.S1_T_max) / params.S1_T_max;
     e1 = pet * params.r1 * S1_T_frac;
     
     // Lower layer (3d)
     Real S2_T_frac = smooth_min(state.S2_T, params.S2_T_max) / params.S2_T_max;
     e2 = pet * r2 * S2_T_frac;
 }
 
 /**
  * @brief Dispatch evaporation based on configuration
  */
 DFUSE_HOST_DEVICE inline void compute_evaporation(
     Real pet, const State& state, const Parameters& params,
     const ModelConfig& config, Real& e1, Real& e2
 ) {
     switch (config.evaporation) {
         case EvaporationType::SEQUENTIAL:
             evap_sequential(pet, state, params, e1, e2);
             break;
         case EvaporationType::ROOT_WEIGHT:
             evap_root_weight(pet, state, params, e1, e2);
             break;
     }
     
     // No lower layer evap for certain architectures
     if (config.lower_arch == LowerLayerArch::SINGLE_NOEVAP) {
         e2 = Real(0);
     }
 }
 
 // ============================================================================
 // PERCOLATION PARAMETERIZATIONS
 // ============================================================================
 
 /**
  * @brief Percolation based on total storage - equation (4a)
  * 
  * VIC-style: q12 = ku * (S1/S1_max)^c
  */
 DFUSE_HOST_DEVICE inline Real percol_total_storage(
     const State& state, const Parameters& params
 ) {
     Real S1_frac = state.S1 / params.S1_max;
     // Ensure positive base
     Real base = smooth_max(S1_frac, Real(0));
     return params.ku * safe_pow(base, params.c);
 }
 
 /**
  * @brief Percolation based on free storage - equation (4b)
  * 
  * PRMS-style: q12 = ku * (S1_F/S1_F_max)^c
  * Only percolates from free storage (excess above tension capacity)
  * FIXED: Use smooth operations for AD compatibility
  */
 DFUSE_HOST_DEVICE inline Real percol_free_storage(
     const State& state, const Parameters& params
 ) {
     // FIXED: Use smaller smooth_k to prevent spurious percolation when S1_F=0
     // Same issue as interflow - smooth_max(0,0,0.01)=0.01, not 0
     constexpr Real smooth_k = Real(0.0001);
     Real safe_S1_F_max = smooth_max(params.S1_F_max, Real(0.1), Real(0.001));
     Real S1_F_frac = smooth_max(state.S1_F, Real(0), smooth_k) / safe_S1_F_max;
     S1_F_frac = smooth_clamp(S1_F_frac, Real(1.0), smooth_k);  // Clamp to [0, 1]
     return params.ku * safe_pow(S1_F_frac, params.c);
 }
 
 /**
  * @brief Percolation with lower zone demand - equation (4c)
  * 
  * Sacramento-style: q12 = q0 * dlz * (S1_F/S1_F_max)
  * where dlz = 1 + alpha * (S2/S2_max)^psi
  */
 DFUSE_HOST_DEVICE inline Real percol_lower_demand(
     const State& state, const Parameters& params, Real q0
 ) {
     Real S2_frac = state.S2 / params.S2_max;
     Real dlz = Real(1) + params.alpha * safe_pow(S2_frac, params.psi);
     Real S1_F_frac = state.S1_F / params.S1_F_max;
     return q0 * dlz * smooth_max(S1_F_frac, Real(0));
 }
 
 /**
  * @brief Dispatch percolation based on configuration
  * 
  * Note: For SINGLE_STATE upper layer architecture with FREE_STORAGE percolation,
  * percolation is zero because there's no separate free storage in onestate_1.
  * This matches Fortran FUSE behavior.
  */
 DFUSE_HOST_DEVICE inline Real compute_percolation(
     const State& state, const Parameters& params,
     const ModelConfig& config, Real q0 = Real(0)
 ) {
     // Note: For SINGLE_STATE (onestate_1), S1_F is computed as max(0, S1 - S1_T_max)
     // in solver.hpp, so FREE_STORAGE percolation works correctly using this derived S1_F.
     // This matches Fortran FUSE behavior in xtry_2_str.f90 and qpercolate.f90.
     
     switch (config.percolation) {
         case PercolationType::TOTAL_STORAGE:
             return percol_total_storage(state, params);
         case PercolationType::FREE_STORAGE:
             return percol_free_storage(state, params);
         case PercolationType::LOWER_DEMAND:
             return percol_lower_demand(state, params, q0);
     }
     return Real(0);
 }
 
 // ============================================================================
 // INTERFLOW PARAMETERIZATIONS
 // ============================================================================
 
 /**
  * @brief Linear interflow - equation (5b)
  * 
  * qif = ki * (S1_F/S1_F_max)
  * 
  * With flux limiting for explicit Euler stability.
  */
 DFUSE_HOST_DEVICE inline Real compute_interflow(
     const State& state, const Parameters& params,
     const ModelConfig& config, Real dt = Real(1)
 ) {
     if (config.interflow == InterflowType::NONE) {
         return Real(0);
     }
     
     constexpr Real smooth_k = Real(0.01);
     
     // Safety check with smooth protection
     Real safe_S1_F_max = smooth_max(params.S1_F_max, Real(0.1), smooth_k);
     
     Real S1_F_frac = smooth_max(state.S1_F, Real(0), smooth_k) / safe_S1_F_max;
     
     // Potential interflow rate
     Real qif_potential = params.ki * S1_F_frac;
     
     // Flux limiting: can't drain more than available storage in one timestep
     Real safe_dt = smooth_max(dt, Real(0.01), smooth_k);
     Real qif_max = Real(0.9) * smooth_max(state.S1_F, Real(0), smooth_k) / safe_dt;
     
     return smooth_min(qif_potential, qif_max, smooth_k);
 }
 
 // ============================================================================
 // BASEFLOW PARAMETERIZATIONS
 // ============================================================================
 
 /**
  * @brief Linear baseflow - equation (6a)
  * 
  * PRMS: qb = v * S2
  */
 DFUSE_HOST_DEVICE inline Real baseflow_linear(
     const State& state, const Parameters& params
 ) {
     return params.v * state.S2;
 }
 
 /**
  * @brief Parallel linear reservoirs - equation (6b)
  * 
  * Sacramento: qb = v_A * S2_FA + v_B * S2_FB
  */
 DFUSE_HOST_DEVICE inline void baseflow_parallel(
     const State& state, const Parameters& params,
     Real& qb_A, Real& qb_B, Real& qb_total
 ) {
     qb_A = params.v_A * state.S2_FA;
     qb_B = params.v_B * state.S2_FB;
     qb_total = qb_A + qb_B;
 }
 
 /**
  * @brief Nonlinear baseflow - equation (6c)
  * 
  * ARNO/VIC: qb = ks * (S2/S2_max)^n
  */
 DFUSE_HOST_DEVICE inline Real baseflow_nonlinear(
     const State& state, const Parameters& params
 ) {
     Real S2_frac = state.S2 / params.S2_max;
     // Use safe_pow for nonlinear baseflow
     return params.ks * safe_pow(smooth_max(S2_frac, Real(0)), params.n);
 }
 
 /**
  * @brief TOPMODEL power law baseflow - equation (6d)
  * 
  * TOPMODEL: qb = ks * (S2/S2_max)^n
  * 
  * Note: The original Clark et al. formulation uses transmissivity profiles,
  * but for compatibility we normalize by S2_max similar to equation (6c).
  * The lambda parameter influences saturated area, not baseflow rate directly.
  */
 DFUSE_HOST_DEVICE inline Real baseflow_topmodel(
     const State& state, const Parameters& params
 ) {
     // Use normalized storage for dimensional consistency
     Real S2_frac = smooth_max(state.S2, Real(0)) / params.S2_max;
     // Use safe_pow
     return params.ks * safe_pow(S2_frac, params.n);
 }
 
 /**
  * @brief Dispatch baseflow computation
  */
 DFUSE_HOST_DEVICE inline void compute_baseflow(
     const State& state, const Parameters& params,
     const ModelConfig& config,
     Real& qb, Real& qb_A, Real& qb_B
 ) {
     qb_A = Real(0);
     qb_B = Real(0);
     
     switch (config.baseflow) {
         case BaseflowType::LINEAR:
             qb = baseflow_linear(state, params);
             break;
         case BaseflowType::PARALLEL_LINEAR:
             baseflow_parallel(state, params, qb_A, qb_B, qb);
             break;
         case BaseflowType::NONLINEAR:
             qb = baseflow_nonlinear(state, params);
             break;
         case BaseflowType::TOPMODEL:
             qb = baseflow_topmodel(state, params);
             break;
     }
 }
 
 // ============================================================================
 // SURFACE RUNOFF / SATURATED AREA
 // ============================================================================
 
 /**
  * @brief Linear saturated area - equation (9a)
  * 
  * PRMS: Ac = (S1_T/S1_T_max) * Ac_max
  */
 DFUSE_HOST_DEVICE inline Real satarea_linear(
     const State& state, const Parameters& params
 ) {
     Real S1_T_frac = smooth_min(state.S1_T, params.S1_T_max) / params.S1_T_max;
     return S1_T_frac * params.Ac_max;
 }
 
 /**
  * @brief Pareto/VIC saturated area - equation (9b)
  * 
  * ARNO/VIC: Ac = 1 - (1 - S1/S1_max)^b
  */
 DFUSE_HOST_DEVICE inline Real satarea_pareto(
     const State& state, const Parameters& params
 ) {
     Real S1_frac = smooth_min(state.S1, params.S1_max) / params.S1_max;
     
     // Robust power law calculation (BRANCH-FREE for Enzyme AD):
     // Use smooth_max to ensure term >= eps without branching
     Real term = Real(1) - S1_frac;
     constexpr Real eps = Real(1e-5);
     constexpr Real k = Real(1e-6);
     Real diff = term - eps;
     Real smooth_abs = std::sqrt(diff * diff + Real(4) * k * k);
     Real safe_term = Real(0.5) * (term + eps + smooth_abs);
     
     return Real(1) - safe_pow(safe_term, params.b);
 }
 
 /**
  * @brief TOPMODEL saturated area from topographic index - equation (9c)
  * 
  * Uses incomplete gamma function for the integral.
  * Simplified implementation using approximation.
  * FIXED: Use branch-free sigmoid for Enzyme AD.
  */
 DFUSE_HOST_DEVICE inline Real satarea_topmodel(
     const State& state, const Parameters& params
 ) {
     // Critical topographic index for saturation (10a)
     Real S2_frac = state.S2 / params.S2_max;
     
     // Approximate: when S2/S2_max -> 1, Ac -> 1
     //              when S2/S2_max -> 0, Ac -> 0
     // Use branch-free sigmoid via tanh for smoothness
     Real shape = params.chi;
     Real x = shape * (S2_frac - Real(0.5));
     Real Ac = Real(0.5) * (Real(1) + std::tanh(x * Real(0.5)));
     
     return smooth_clamp(Ac, params.Ac_max);
 }
 
 /**
  * @brief Compute saturated area and surface runoff
  */
 DFUSE_HOST_DEVICE inline void compute_surface_runoff(
     Real throughfall, const State& state, const Parameters& params,
     const ModelConfig& config, Real& Ac, Real& qsx
 ) {
     switch (config.surface_runoff) {
         case SurfaceRunoffType::UZ_LINEAR:
             Ac = satarea_linear(state, params);
             break;
         case SurfaceRunoffType::UZ_PARETO:
             Ac = satarea_pareto(state, params);
             break;
         case SurfaceRunoffType::LZ_GAMMA:
             Ac = satarea_topmodel(state, params);
             break;
     }
     
     // Surface runoff is rain on saturated area - equation (11)
     qsx = Ac * throughfall;
 }
 
 // ============================================================================
 // BUCKET OVERFLOW FLUXES
 // ============================================================================
 
 /**
  * @brief Compute all overflow fluxes
  * 
  * Implements equations (12a)-(12g) with smooth thresholds.
  */
 DFUSE_HOST_DEVICE inline void compute_overflow(
     Real infiltration, Real q12,
     const State& state, const Parameters& params,
     const ModelConfig& config,
     Real& qurof, Real& qutof, Real& qufof,
     Real& qstof, Real& qsfof, Real& qsfofa, Real& qsfofb
 ) {
     Real w_up = params.smooth_frac * params.S1_max;
     Real w_lo = params.smooth_frac * params.S2_max;
     
     qurof = Real(0);
     qutof = Real(0);
     qufof = Real(0);
     qstof = Real(0);
     qsfof = Real(0);
     qsfofa = Real(0);
     qsfofb = Real(0);
     
     // Upper layer overflow depends on architecture
     switch (config.upper_arch) {
         case UpperLayerArch::SINGLE_STATE:
             // (12c) variant for single state
             qufof = infiltration * logistic_overflow(state.S1, params.S1_max, w_up);
             break;
             
         case UpperLayerArch::TENSION_FREE:
             // (12b) tension to free overflow
             qutof = infiltration * logistic_overflow(state.S1_T, params.S1_T_max, 
                                                       params.smooth_frac * params.S1_T_max);
             // (12c) free overflow
             qufof = qutof * logistic_overflow(state.S1_F, params.S1_F_max,
                                                params.smooth_frac * params.S1_F_max);
             break;
             
         case UpperLayerArch::TENSION2_FREE:
             // (12a) primary tension overflow
             qurof = infiltration * logistic_overflow(state.S1_TA, params.S1_TA_max,
                                                       params.smooth_frac * params.S1_TA_max);
             // (12b) secondary tension overflow  
             qutof = qurof * logistic_overflow(state.S1_TB, params.S1_TB_max,
                                                params.smooth_frac * params.S1_TB_max);
             // (12c) free overflow
             qufof = qutof * logistic_overflow(state.S1_F, params.S1_F_max,
                                                params.smooth_frac * params.S1_F_max);
             break;
     }
     
     // Lower layer overflow
     switch (config.lower_arch) {
         case LowerLayerArch::SINGLE_NOEVAP:
             // No overflow, infinite capacity
             break;
             
         case LowerLayerArch::SINGLE_EVAP:
             // (12e) single reservoir overflow
             qsfof = q12 * logistic_overflow(state.S2, params.S2_max, w_lo);
             break;
             
         case LowerLayerArch::TENSION_2RESERV:
             // (12d) tension overflow
             qstof = params.kappa * q12 * logistic_overflow(state.S2_T, params.S2_T_max,
                                                             params.smooth_frac * params.S2_T_max);
             // (12f) primary reservoir overflow
             Real to_free = (Real(1) - params.kappa) * q12 / Real(2) + qstof / Real(2);
             qsfofa = to_free * logistic_overflow(state.S2_FA, params.S2_FA_max,
                                                   params.smooth_frac * params.S2_FA_max);
             // (12g) secondary reservoir overflow
             qsfofb = to_free * logistic_overflow(state.S2_FB, params.S2_FB_max,
                                                   params.smooth_frac * params.S2_FB_max);
             break;
     }
 }
 
 // ============================================================================
 // STATE DERIVATIVE COMPUTATION
 // ============================================================================
 
 /**
  * @brief Compute time derivatives of all state variables
  * 
  * This is the core function for numerical integration.
  * Implements state equations (1a)-(1c) and (2a)-(2c).
  */
 DFUSE_HOST_DEVICE inline void compute_derivatives(
     const State& state, const Flux& flux, const Parameters& params,
     const ModelConfig& config, Real* dSdt
 ) {
     int idx = 0;
     
     // Infiltration = throughfall - surface runoff
     Real infiltration = flux.throughfall - flux.qsx;
     
     switch (config.upper_arch) {
         case UpperLayerArch::SINGLE_STATE:
             // dS1/dt = (p - qsx) - e1 - q12 - qif - qufof   (1a)
             dSdt[idx++] = infiltration - flux.e1 - flux.q12 - flux.qif - flux.qufof;
             break;
             
         case UpperLayerArch::TENSION_FREE:
             // dS1_T/dt = (p - qsx) - e1 - qutof             (1b)
             dSdt[idx++] = infiltration - flux.e1 - flux.qutof;
             // dS1_F/dt = qutof - q12 - qif - qufof
             dSdt[idx++] = flux.qutof - flux.q12 - flux.qif - flux.qufof;
             break;
             
         case UpperLayerArch::TENSION2_FREE:
             // dS1_TA/dt = (p - qsx) - e1_A - qurof          (1c)
             dSdt[idx++] = infiltration - flux.e1_A - flux.qurof;
             // dS1_TB/dt = qurof - e1_B - qutof
             dSdt[idx++] = flux.qurof - flux.e1_B - flux.qutof;
             // dS1_F/dt = qutof - q12 - qif - qufof
             dSdt[idx++] = flux.qutof - flux.q12 - flux.qif - flux.qufof;
             break;
     }
     
     switch (config.lower_arch) {
         case LowerLayerArch::SINGLE_NOEVAP:
             // dS2/dt = q12 - qb                              (2a)
             dSdt[idx++] = flux.q12 - flux.qb;
             break;
             
         case LowerLayerArch::SINGLE_EVAP:
             // dS2/dt = q12 - e2 - qb - qsfof                (2b)
             dSdt[idx++] = flux.q12 - flux.e2 - flux.qb - flux.qsfof;
             break;
             
         case LowerLayerArch::TENSION_2RESERV:
             // dS2_T/dt = kappa*q12 - e2 - qstof             (2c)
             dSdt[idx++] = params.kappa * flux.q12 - flux.e2 - flux.qstof;
             // dS2_FA/dt = (1-kappa)*q12/2 + qstof/2 - qb_A - qsfofa
             Real to_fa = (Real(1) - params.kappa) * flux.q12 / Real(2) + flux.qstof / Real(2);
             dSdt[idx++] = to_fa - flux.qb_A - flux.qsfofa;
             // dS2_FB/dt = (1-kappa)*q12/2 + qstof/2 - qb_B - qsfofb
             Real to_fb = (Real(1) - params.kappa) * flux.q12 / Real(2) + flux.qstof / Real(2);
             dSdt[idx++] = to_fb - flux.qb_B - flux.qsfofb;
             break;
     }
     
     // Snow derivative
     if (config.enable_snow) {
         // dSWE/dt handled separately in snow module
         // Not included in dSdt as it uses its own update
     }
 }
 
 } // namespace physics
 } // namespace dfuse