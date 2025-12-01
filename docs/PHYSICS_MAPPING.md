# dFUSE Physics Documentation

## Overview

dFUSE (differentiable Framework for Understanding Structural Errors) is a PyTorch-compatible
reimplementation of the FUSE hydrological model framework (Clark et al., 2008).

This document maps the physics equations between Fortran FUSE and dFUSE, identifies known
differences, and provides guidance for interpreting comparison results.

## Reference

Clark, M.P., Slater, A.G., Rupp, D.E., Woods, R.A., Vrugt, J.A., Gupta, H.V., Wagener, T.,
and Hay, L.E. (2008). Framework for Understanding Structural Errors (FUSE): A modular
framework to diagnose differences between hydrological models. Water Resources Research,
44, W00B02. doi:10.1029/2007WR006735

---

## Model Decisions and Physics Equations

### 1. Upper Layer Architecture (ARCH1)

#### onestate_1 (Single State)
VIC/ARNO style single storage.

**State equation (Eq 1a):**
```
dS₁/dt = P_eff - E₁ - Q₁₂ - Q_if - Q_ufof
```

Where:
- `P_eff` = effective precipitation (throughfall after snow)
- `E₁` = evaporation from upper layer
- `Q₁₂` = percolation to lower layer
- `Q_if` = interflow
- `Q_ufof` = upper layer overflow

**dFUSE implementation:** `include/dfuse/physics.hpp`, `compute_derivatives()`, case `SINGLE_STATE`

**Note:** For interflow with SINGLE_STATE, dFUSE computes free storage as:
```
S₁_F = max(S₁ - S₁_T_max, 0)
```
This matches Fortran FUSE behavior where interflow only occurs when storage exceeds tension capacity.

#### tension1_1 (Tension + Free)
PRMS style with explicit tension and free storage.

**State equations (Eq 1b):**
```
dS₁_T/dt = P_eff - E₁ - Q_utof
dS₁_F/dt = Q_utof - Q₁₂ - Q_if - Q_ufof
```

#### tension2_1 (Two Tension + Free)
Sacramento style with primary and secondary tension storage.

**State equations (Eq 1c, 1d):**
```
dS₁_TA/dt = P_eff - E₁_A - Q_urof
dS₁_TB/dt = Q_urof - E₁_B - Q_utof
dS₁_F/dt = Q_utof - Q₁₂ - Q_if - Q_ufof
```

---

### 2. Lower Layer Architecture (ARCH2)

#### unlimfrc_2 (Unlimited, No Evap)
Simple baseflow reservoir without evaporation.

**State equation (Eq 2a):**
```
dS₂/dt = Q₁₂ - Q_b
```

**Baseflow:** Linear: `Q_b = v × S₂`

#### unlimpow_2 (Unlimited, With Evap)
Baseflow reservoir with evaporation and optional nonlinear baseflow.

**State equation (Eq 2b):**
```
dS₂/dt = Q₁₂ - E₂ - Q_b - Q_sfof
```

**Baseflow:** Typically nonlinear when paired with ARNO/VIC:
```
Q_b = k_s × (S₂/S₂_max)^n
```

**dFUSE mapping:** When decisions specify `unlimpow_2` with `arno_x_vic`, dFUSE uses
`BaseflowType::NONLINEAR`.

#### fixedsiz_2 (Fixed Size, TOPMODEL)
TOPMODEL style with fixed maximum storage.

**State equation:** Same as unlimpow_2

**Baseflow:** TOPMODEL power law:
```
Q_b = k_s × (S₂/S₂_max)^n
```

#### tens2pll_2 (Tension + 2 Parallel Reservoirs)
Sacramento style with tension storage and two parallel baseflow reservoirs.

**State equations (Eq 2c, 2d, 2e):**
```
dS₂_T/dt = κ×Q₁₂ - E₂ - Q_stof
dS₂_FA/dt = (1-κ)×Q₁₂/2 + Q_stof/2 - Q_b_A - Q_sfofa
dS₂_FB/dt = (1-κ)×Q₁₂/2 + Q_stof/2 - Q_b_B - Q_sfofb
```

**Baseflow:**
```
Q_b = Q_b_A + Q_b_B = v_A × S₂_FA + v_B × S₂_FB
```

---

### 3. Evaporation (ESOIL)

#### sequential
Upper layer satisfied first, then lower layer from residual.

**Equations (Eq 3a, 3b):**
```
E₁ = PET × (S₁_T / S₁_T_max)
E₂ = (PET - E₁) × (S₂_T / S₂_T_max)
```

#### rootweight
Evaporation partitioned by root distribution.

**Equations (Eq 3c, 3d):**
```
E₁ = PET × r₁ × (S₁_T / S₁_T_max)
E₂ = PET × (1-r₁) × (S₂_T / S₂_T_max)
```

**Note:** For SINGLE_STATE architecture, dFUSE uses `S₁_T = min(S₁, S₁_T_max)`.

---

### 4. Percolation (QPERC)

#### perc_w2sat (Total Storage Based)
VIC style percolation from total water content.

**Equation (Eq 4a):**
```
Q₁₂ = k_u × (S₁/S₁_max)^c
```

**dFUSE:** `PercolationType::TOTAL_STORAGE`

#### perc_f2sat (Free Storage Based)
PRMS style percolation from free storage only.

**Equation (Eq 4b):**
```
Q₁₂ = k_u × (S₁_F/S₁_F_max)^c
```

**dFUSE:** `PercolationType::FREE_STORAGE`

**Critical note:** This requires `S₁_F` to be computed first. For SINGLE_STATE architecture,
`S₁_F = max(S₁ - S₁_T_max, 0)`.

#### perc_lower (Lower Zone Demand)
Sacramento style with lower zone demand factor.

**Equation (Eq 4c):**
```
Q₁₂ = Q₀ × dlz × (S₁_F/S₁_F_max)
dlz = 1 + α × (S₂/S₂_max)^ψ
```

**dFUSE:** `PercolationType::LOWER_DEMAND`

---

### 5. Interflow (QINTF)

#### intflwnone
No interflow.

#### intflwsome
Linear interflow from free storage.

**Equation (Eq 5b):**
```
Q_if = k_i × (S₁_F/S₁_F_max)
```

**dFUSE notes:**
- Uses hard threshold (no smoothing) to match Fortran behavior
- Flux limiting prevents numerical instability: `Q_if ≤ 0.9 × S₁_F/dt`
- For SINGLE_STATE, `S₁_F = max(S₁ - S₁_T_max, 0)`

---

### 6. Surface Runoff (QSURF)

All methods use:
```
Q_sx = A_c × P_throughfall
```

Where `A_c` is the saturated contributing area fraction.

#### arno_x_vic (VIC/ARNO Pareto)
**Equation (Eq 9b):**
```
A_c = 1 - (1 - S₁/S₁_max)^b
```

#### prms_varnt (Linear)
**Equation (Eq 9a):**
```
A_c = (S₁_T/S₁_T_max) × A_c_max
```

#### tmdl_param (TOPMODEL)
**Equation (Eq 9c):**
```
A_c = ∫[z_crit to ∞] f(z)dz
```
Where f(z) is the topographic index distribution (gamma).

---

### 7. Snow (SNOWM)

#### no_snowmod
No snow accumulation or melt.

#### temp_index
Temperature index model with elevation bands.

**For each elevation band b:**
```
T_b = T + lapse_rate × (elev_b - elev_ref)/1000
P_b = P × (1 + opg × (elev_b - elev_ref)/1000)

snow_frac = 1 / (1 + exp(k × (T_b - T_rain)))
snow = P_b × snow_frac
rain = P_b × (1 - snow_frac)

melt = melt_rate × max(T_b - T_melt, 0)
melt = min(melt, SWE_b)
SWE_b = SWE_b + snow - melt
```

**Basin average:**
```
rain_eff = Σ(area_frac_b × rain_b)
melt_eff = Σ(area_frac_b × melt_b)
throughfall = rain_eff + melt_eff
```

**dFUSE vs Fortran differences:**
- Fortran uses MFMAX/MFMIN with seasonal variation
- dFUSE uses average: `melt_rate = (MFMAX + MFMIN) / 2`
- dFUSE uses sharper rain-snow transition (k=5) for differentiability

---

### 8. Routing (Q_TDH)

#### no_routing
No delay: `Q_routed = Q_instnt`

#### rout_gamma
Gamma distribution unit hydrograph.

**NOT IMPLEMENTED in dFUSE.** Currently `Q_routed = Q_instnt`.

This explains timing differences, especially for:
- Peak flow timing (peaks may be shifted)
- Recession limb shape
- Low flow tails

---

## Numerical Solver Comparison

### Fortran FUSE
Based on Clark & Kavetski (2010) "Ancient numerical daemons of conceptual hydrological modeling":

Uses **implicit Euler with Newton iteration and adaptive sub-stepping**:
- NOT SUNDIALS - this is a custom implementation
- Newton iteration for mass balance convergence at each timestep
- Adaptive sub-stepping when convergence is difficult
- Jacobian computed analytically or via finite differences

Output diagnostics (in NetCDF):
- `num_funcs`: RHS function evaluations per step (more = harder to converge)
- `numjacobian`: Jacobian evaluations per step
- `sub_accept`: Number of accepted sub-steps per day
- `sub_reject`: Number of rejected sub-steps per day
- `sub_noconv`: Non-convergent steps (should be 0 or near 0)
- `max_iterns`: Maximum Newton iterations used

### dFUSE

**Default: Explicit Euler** (in `kernels.hpp`, `fuse_step()`)
- Single step per day (dt=1.0)
- No iteration required
- Fast and differentiable
- May be less accurate for stiff systems

**Optional: Implicit Euler with Newton** (in `solver.hpp`, `ImplicitEulerSolver`)
- Matches Fortran approach more closely
- Requires building with solver support
- Currently not exposed in Python bindings

**Optional: SUNDIALS CVODE** (requires `-DDFUSE_USE_SUNDIALS=ON`)
- Adams-Moulton (non-stiff) or BDF (stiff) methods
- Professional-grade adaptive timestepping
- Best for production use with stiff systems

### Why Explicit Euler in dFUSE?

The primary design goal of dFUSE is **differentiability** for gradient-based optimization.
Explicit Euler has several advantages:
1. **Simple gradient computation**: No need to differentiate through Newton iteration
2. **Predictable behavior**: Fixed computational graph
3. **GPU-friendly**: No iterative convergence required
4. **Fast**: Single matrix multiply per timestep

For most applications with reasonable parameters and daily timesteps, explicit Euler
provides acceptable accuracy. For high-rate systems (ki > 100 mm/day), consider:
- Using smaller timesteps (dt = 0.5 or 0.25 days)
- Building with SUNDIALS for production runs
- Accepting slightly lower numerical precision

---

## Known Differences and Expected Discrepancies

### 1. Low Flows
- dFUSE maintains small baseflow (~0.01-0.05 mm/day) when S2 > 0
- Fortran may zero out very small flows
- Impact: Flow duration curve tail diverges below ~0.1 mm/day

### 2. Peak Timing
- Without routing, dFUSE peaks occur instantaneously
- Fortran gamma routing delays and smooths peaks
- Impact: 0.5-1 day timing offset possible

### 3. Numerical Precision
- Smooth approximations introduce small biases near thresholds
- Explicit Euler may overshoot with high rate constants
- Impact: Generally < 1% difference in total volume

### 4. Melt Seasonality
- Fortran varies melt rate seasonally
- dFUSE uses constant average
- Impact: Slight timing differences in spring melt

---

## Validation Criteria

**Excellent match:** NSE > 0.9, Correlation > 0.95
**Good match:** NSE > 0.8, Correlation > 0.90
**Acceptable:** NSE > 0.6, Correlation > 0.80
**Investigate:** NSE < 0.6 or systematic bias > 10%

For publication, aim for NSE > 0.8 across multiple decision combinations.

---

## Troubleshooting

### Low NSE despite correct decisions

1. **Check percolation type mapping**
   - `perc_f2sat` → FREE_STORAGE (index 1)
   - `perc_w2sat` → TOTAL_STORAGE (index 0)

2. **Check baseflow type**
   - `unlimpow_2` with `arno_x_vic` → NONLINEAR (index 2)
   - `tens2pll_2` → PARALLEL_LINEAR (index 1)

3. **Check initial conditions**
   - S1 should start at tension capacity
   - S2 should start near equilibrium (~25% for nonlinear baseflow)

4. **Check elevation bands**
   - Snow basins need elevation band data
   - Reference elevation should be area-weighted mean

5. **Enable SUNDIALS for stiff systems**
   - High ki (>100 mm/day) may need implicit solver

### Systematic bias

- Positive bias: Too much runoff → check percolation, interflow rates
- Negative bias: Too little runoff → check snow melt, infiltration
- Seasonal bias: Check snow parameters (lapse, opg, T_rain, T_melt)
