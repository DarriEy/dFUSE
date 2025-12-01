#!/usr/bin/env python
"""
Compare dFUSE with Fortran FUSE for the Klondike Bonanza Creek basin.
"""

import sys
import subprocess
from pathlib import Path
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dfuse_netcdf import (
    read_fuse_forcing, 
    read_elevation_bands,
    parse_fuse_decisions, 
    parse_file_manager,
    parse_fortran_constraints,
    FUSERunner
)

# Paths
BASE_PATH = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Klondike_Bonanza_Creek")
FM_PATH = BASE_PATH / "settings/FUSE/fm_catch.txt"
FORTRAN_EXE = Path("/Users/darrieythorsson/compHydro/data/SYMFLUENCE_data/installs/fuse/bin/fuse.exe")
BASIN_ID = "Klondike_Bonanza_Creek"

# Default routing parameters (gamma distribution)
GAMMA_SHAPE = 2.5  # Typical shape parameter for gamma routing


def apply_routing(instant_runoff: np.ndarray, decisions, fortran_params, dt: float = 1.0) -> np.ndarray:
    """
    Apply gamma routing to instantaneous runoff if routing is enabled.
    
    Args:
        instant_runoff: Instantaneous runoff from bucket model
        decisions: FUSE decisions object with q_tdh
        fortran_params: Fortran parameters with TIMEDELAY
        dt: Time step in days
        
    Returns:
        Routed runoff (or unchanged if no routing)
    """
    if decisions.q_tdh == 'no_routing':
        return instant_runoff
    
    # Get routing parameters
    mean_delay = fortran_params.TIMEDELAY
    
    # Import routing function from dfuse_core
    import dfuse_core
    
    # Apply gamma routing
    routed = dfuse_core.route_runoff(
        instant_runoff.astype(np.float32),
        GAMMA_SHAPE,
        mean_delay,
        dt
    )
    
    return routed


def main():
    print("=" * 60)
    print("dFUSE vs Fortran FUSE Comparison")
    print("=" * 60)
    
    # Parse file manager
    print("\n1. Parsing file manager...")
    fm = parse_file_manager(FM_PATH)
    print(f"   Input path: {fm['input_path']}")
    print(f"   Output path: {fm['output_path']}")
    print(f"   Simulation: {fm['date_start_sim']} to {fm['date_end_sim']}")
    
    # Compute start day of year for seasonal melt factor
    from datetime import datetime
    try:
        date_start = datetime.strptime(fm['date_start_sim'], '%Y-%m-%d')
        start_day_of_year = date_start.timetuple().tm_yday  # 1-365
        print(f"   Start day of year: {start_day_of_year}")
    except:
        start_day_of_year = 1  # Default to Jan 1 if parsing fails
        print(f"   Warning: Could not parse start date, using day_of_year=1")
    
    # Parse decisions
    print("\n2. Parsing model decisions...")
    decisions_path = Path(fm['setngs_path']) / fm['m_decisions']
    decisions = parse_fuse_decisions(decisions_path)
    print(f"   ARCH1 (upper layer): {decisions.arch1}")
    print(f"   ARCH2 (lower layer): {decisions.arch2}")
    print(f"   QSURF (surface runoff): {decisions.qsurf}")
    print(f"   QPERC (percolation): {decisions.qperc}")
    print(f"   ESOIL (evaporation): {decisions.esoil}")
    print(f"   QINTF (interflow): {decisions.qintf}")
    print(f"   Q_TDH (routing): {decisions.q_tdh}")
    print(f"   SNOWM (snow): {decisions.snowmod}")
    
    # Convert to dFUSE config
    config_dict = decisions.to_config_dict()
    print(f"\n   dFUSE config: {config_dict}")
    
    # Parse Fortran parameters
    print("\n2b. Parsing Fortran parameters...")
    constraints_path = Path(fm['setngs_path']) / fm['constraints']
    fortran_params = parse_fortran_constraints(constraints_path)
    print(f"   MAXWATR_1 (S1_max): {fortran_params.MAXWATR_1} mm")
    print(f"   MAXWATR_2 (S2_max): {fortran_params.MAXWATR_2} mm")
    print(f"   BASERTE (ks): {fortran_params.BASERTE} mm/day")
    print(f"   QB_POWR (n): {fortran_params.QB_POWR}")
    print(f"   AXV_BEXP (b): {fortran_params.AXV_BEXP}")
    print(f"   PXTEMP (T_rain): {fortran_params.PXTEMP} °C")
    print(f"   MBASE (T_melt): {fortran_params.MBASE} °C")
    print(f"   MFMAX/MFMIN: {fortran_params.MFMAX}/{fortran_params.MFMIN} mm/°C/day")
    print(f"   Seasonal melt: MFMAX on June 21, MFMIN on Dec 21")
    print(f"   PERCRTE (ku): {fortran_params.PERCRTE} mm/day")
    print(f"   PERCEXP (c): {fortran_params.PERCEXP}")
    print(f"   IFLWRTE (ki): {fortran_params.IFLWRTE} mm/day")
    print(f"   FRACTEN (f_tens): {fortran_params.FRACTEN}")
    print(f"   OPG: {fortran_params.OPG} km⁻¹")
    print(f"   LAPSE: {fortran_params.LAPSE} °C/km")
    
    # Load forcing
    print("\n3. Loading forcing data...")
    forcing_path = Path(fm['input_path']) / f"{BASIN_ID}{fm['suffix_forcing']}"
    forcing = read_fuse_forcing(forcing_path)
    print(f"   Timesteps: {forcing.n_timesteps}")
    print(f"   Precip: {np.nanmean(forcing.precip):.2f} mm/day (mean), {np.nanmax(forcing.precip):.2f} mm/day (max)")
    print(f"   Temp: {np.nanmean(forcing.temp):.2f} °C (mean), range [{np.nanmin(forcing.temp):.1f}, {np.nanmax(forcing.temp):.1f}]")
    print(f"   PET: {np.nanmean(forcing.pet):.2f} mm/day (mean)")
    if forcing.q_obs is not None:
        valid_q = forcing.q_obs[~np.isnan(forcing.q_obs) & (forcing.q_obs > -9000)]
        print(f"   Q_obs: {np.mean(valid_q):.2f} mm/day (mean), {len(valid_q)} valid values")
    
    # Run dFUSE
    print("\n4. Running dFUSE...")
    from dfuse import (
        FUSE, FUSEConfig,
        UpperLayerArch, LowerLayerArch, BaseflowType, PercolationType,
        SurfaceRunoffType, EvaporationType, InterflowType
    )
    import torch
    
    # Create config with proper enum types
    config = FUSEConfig(
        upper_arch=UpperLayerArch(config_dict['upper_arch']),
        lower_arch=LowerLayerArch(config_dict['lower_arch']),
        baseflow=BaseflowType(config_dict['baseflow']),
        percolation=PercolationType(config_dict['percolation']),
        surface_runoff=SurfaceRunoffType(config_dict['surface_runoff']),
        evaporation=EvaporationType(config_dict['evaporation']),
        interflow=InterflowType(config_dict['interflow']),
        enable_snow=config_dict['enable_snow']
    )
    print(f"   Config: {config}")
    
    model = FUSE(config=config, learnable_params=False)
    
    # Initialize SWE tracking variable (will be set if using elevation bands)
    swe_basin_avg = None
    
    # Prepare forcing tensor
    forcing_tensor = forcing.to_tensor()
    print(f"   Forcing tensor shape: {forcing_tensor.shape}")
    
    # Get initial state - size depends on config
    n_states = config.num_states
    print(f"   Number of states for this config: {n_states}")
    
    state = torch.zeros(n_states, dtype=torch.float32)
    # Use Fortran-like equilibrium initial conditions:
    # - S1 at tension capacity (so free storage = 0, interflow = 0)
    # - S2 at ~25% where nonlinear baseflow ≈ 0
    state[0] = fortran_params.MAXWATR_1 * fortran_params.FRACTEN  # S1 = S1_T_max
    
    # S2 index depends on upper layer architecture
    s2_idx = config.num_upper_states
    state[s2_idx] = fortran_params.MAXWATR_2 * 0.25  # S2 at Fortran equilibrium (~250mm)
    
    # SWE is the last state if snow is enabled
    if config.enable_snow:
        state[-1] = 0.0  # SWE starts at 0
    
    print(f"   Initial state: S1={state[0]:.1f}, S2={state[s2_idx]:.1f}, SWE={state[-1] if config.enable_snow else 'N/A'}")
    
    # Run model using dfuse_core directly with Fortran parameters
    import dfuse_core
    
    state_np = state.numpy().astype(np.float32)
    forcing_np = forcing_tensor.numpy().astype(np.float32)
    params_np = fortran_params.to_dfuse_params(arch2=decisions.arch2)
    
    print(f"   Using Fortran parameters:")
    s2_max_display = f"{params_np[1]:.0e}" if params_np[1] > 1e6 else f"{params_np[1]:.1f}"
    print(f"     S1_max={params_np[0]:.1f}, S2_max={s2_max_display} {'(unlimited)' if params_np[1] > 1e6 else ''}")
    print(f"     f_tens={params_np[2]:.2f} (S1_T_max={params_np[2]*params_np[0]:.1f}mm)")
    print(f"     ks={params_np[12]:.1f}, n={params_np[13]:.1f}, b={params_np[18]:.3f}")
    print(f"     ki={params_np[11]:.1f} (interflow rate)")
    print(f"     T_rain={params_np[22]:.1f}°C, T_melt={params_np[23]:.1f}°C, melt_rate={params_np[24]:.2f} mm/°C/day")
    print(f"     MFMAX={params_np[27]:.2f}, MFMIN={params_np[28]:.2f} mm/°C/day (seasonal)")
    print(f"     lapse_rate={params_np[25]:.1f}°C/km, opg={params_np[26]:.2f} km⁻¹")
    print(f"     ki (interflow)={params_np[11]:.1f} mm/day")
    print(f"     f_tens={params_np[2]:.3f}, f_rchr={params_np[3]:.3f}")
    print(f"     S1_T_max={params_np[0]*params_np[2]:.1f} mm, S1_F_max={params_np[0]*(1-params_np[2]):.1f} mm")
    
    # Load elevation bands
    elev_bands_path = Path(fm['input_path']) / f"{BASIN_ID}{fm['suffix_elev_bands']}"
    print(f"\n   Loading elevation bands from: {elev_bands_path}")
    
    try:
        elev_bands = read_elevation_bands(elev_bands_path)
        n_bands = elev_bands.n_bands
        print(f"   Elevation bands: {n_bands}")
        print(f"     Elevation range: {elev_bands.mean_elev.min():.0f} - {elev_bands.mean_elev.max():.0f} m")
        print(f"     Area-weighted mean elev: {np.sum(elev_bands.area_frac * elev_bands.mean_elev):.0f} m")
        ref_elev = np.sum(elev_bands.area_frac * elev_bands.mean_elev)  # Use area-weighted mean as reference
        use_elevation_bands = True
    except Exception as e:
        print(f"   Warning: Could not load elevation bands: {e}")
        print(f"   Falling back to single-layer snow model")
        use_elevation_bands = False
    
    # Check for SUNDIALS availability
    use_sundials = getattr(dfuse_core, 'HAS_SUNDIALS', False)
    solver = "sundials_bdf" if use_sundials else "euler"
    print(f"   Solver: {solver} {'(implicit, matches Fortran)' if use_sundials else '(explicit Euler)'}")
    
    if use_elevation_bands:
        # Run with elevation bands AND return fluxes AND SWE trajectory
        result = dfuse_core.run_fuse_elevation_bands(
            state_np, forcing_np, params_np, config.to_dict(),
            elev_bands.area_frac.astype(np.float32),
            elev_bands.mean_elev.astype(np.float32),
            float(ref_elev),
            None,  # Initial SWE = 0 for all bands
            1.0,   # dt
            True,  # return_fluxes
            True,  # return_swe_trajectory - enabled for comparison
            start_day_of_year,  # For seasonal MFMAX/MFMIN variation
            solver  # Use SUNDIALS if available
        )
        final_state, runoff_dfuse_np, fluxes_np, final_swe_bands, swe_trajectory = result
        
        # Compute basin-average SWE over time
        # swe_trajectory shape: [n_timesteps, n_bands]
        swe_basin_avg = np.sum(swe_trajectory * elev_bands.area_frac[np.newaxis, :], axis=1)
        
        # Flux indices: 0=q_total, 1=e_total, 2=qsx, 3=qb, 4=q12, 5=e1, 6=e2, 7=qif, 8=rain, 9=melt, 10=Ac
        total_swe = np.sum(final_swe_bands * elev_bands.area_frac)
        print(f"   dFUSE runoff: mean={np.mean(runoff_dfuse_np):.3f}, max={np.max(runoff_dfuse_np):.3f} mm/day")
        print(f"   Final state: S1={final_state[0]:.1f}, S2={final_state[s2_idx]:.1f}")
        print(f"   dFUSE SWE: mean={np.mean(swe_basin_avg):.1f}, max={np.max(swe_basin_avg):.1f} mm (basin-avg over time)")
        print(f"   Final SWE (total): {total_swe:.1f} mm")
        print(f"   Final SWE per band: min={final_swe_bands.min():.1f}, max={final_swe_bands.max():.1f} mm")
        
        # Print flux component breakdown
        print(f"\n   dFUSE Flux Components:")
        print(f"     qsx (surface runoff): mean={np.mean(fluxes_np[:,2]):.4f}, max={np.max(fluxes_np[:,2]):.3f} mm/day")
        print(f"     qb (baseflow):        mean={np.mean(fluxes_np[:,3]):.4f}, max={np.max(fluxes_np[:,3]):.3f} mm/day")
        print(f"     qif (interflow):      mean={np.mean(fluxes_np[:,7]):.4f}, max={np.max(fluxes_np[:,7]):.3f} mm/day")
        print(f"     q12 (percolation):    mean={np.mean(fluxes_np[:,4]):.4f}, max={np.max(fluxes_np[:,4]):.3f} mm/day")
        print(f"     rain (eff):           mean={np.mean(fluxes_np[:,8]):.4f}, max={np.max(fluxes_np[:,8]):.3f} mm/day")
        print(f"     melt (eff):           mean={np.mean(fluxes_np[:,9]):.4f}, max={np.max(fluxes_np[:,9]):.3f} mm/day")
        print(f"     Ac (sat area):        mean={np.mean(fluxes_np[:,10]):.4f}, max={np.max(fluxes_np[:,10]):.3f}")
        print(f"     qufof (upper oflow):  mean={np.mean(fluxes_np[:,11]):.4f}, max={np.max(fluxes_np[:,11]):.3f} mm/day")
        print(f"     qsfof (lower oflow):  mean={np.mean(fluxes_np[:,12]):.4f}, max={np.max(fluxes_np[:,12]):.3f} mm/day")
        print(f"     throughfall:          mean={np.mean(fluxes_np[:,13]):.4f}, max={np.max(fluxes_np[:,13]):.3f} mm/day")
        
        # Winter period analysis (first 100 days - should be near zero)
        print(f"\n   Winter Period (first 100 days):")
        print(f"     Input temp: mean={np.mean(forcing.temp[:100]):.1f}°C")
        print(f"     Rain (eff): mean={np.mean(fluxes_np[:100,8]):.4f} mm/day")
        print(f"     Melt (eff): mean={np.mean(fluxes_np[:100,9]):.4f} mm/day")
        print(f"     Eff precip: mean={np.mean(fluxes_np[:100,8]+fluxes_np[:100,9]):.4f} mm/day")
        print(f"     Q_total:    mean={np.mean(runoff_dfuse_np[:100]):.4f} mm/day")
        print(f"     qb:         mean={np.mean(fluxes_np[:100,3]):.4f} mm/day")
        print(f"     qif:        mean={np.mean(fluxes_np[:100,7]):.4f} mm/day")
        print(f"     qsx:        mean={np.mean(fluxes_np[:100,2]):.4f} mm/day")
        
        # Check first few days explicitly
        print(f"\n   First 5 days trace:")
        for d in range(5):
            print(f"     Day {d}: T={forcing.temp[d]:.1f}°C, P={forcing.precip[d]:.2f}mm, "
                  f"rain={fluxes_np[d,8]:.3f}, melt={fluxes_np[d,9]:.3f}, "
                  f"Q={runoff_dfuse_np[d]:.3f}, qb={fluxes_np[d,3]:.3f}, qif={fluxes_np[d,7]:.3f}")
        
        # Store instantaneous runoff before routing
        runoff_dfuse_instnt = runoff_dfuse_np.copy()
    else:
        final_state, runoff_dfuse_np = dfuse_core.run_fuse(
            state_np, forcing_np, params_np, config.to_dict(), 1.0
        )
        print(f"   dFUSE runoff: mean={np.mean(runoff_dfuse_np):.3f}, max={np.max(runoff_dfuse_np):.3f} mm/day")
        swe_final = final_state[-1] if config.enable_snow else 0
        print(f"   Final state: S1={final_state[0]:.1f}, S2={final_state[s2_idx]:.1f}, SWE={swe_final:.1f}")
        runoff_dfuse_instnt = runoff_dfuse_np.copy()
    
    # Apply routing if enabled
    if decisions.q_tdh == 'rout_gamma':
        print(f"\n   Applying gamma routing...")
        print(f"     Mean delay (TIMEDELAY): {fortran_params.TIMEDELAY:.2f} days")
        print(f"     Gamma shape: {GAMMA_SHAPE}")
        runoff_dfuse_routed = apply_routing(runoff_dfuse_instnt, decisions, fortran_params)
        print(f"   dFUSE routed: mean={np.mean(runoff_dfuse_routed):.3f}, max={np.max(runoff_dfuse_routed):.3f} mm/day")
        print(f"   Routing effect: mean |routed - instnt| = {np.mean(np.abs(runoff_dfuse_routed - runoff_dfuse_instnt)):.4f} mm/day")
    else:
        runoff_dfuse_routed = runoff_dfuse_instnt
        print(f"\n   No routing (Q_TDH: {decisions.q_tdh})")
    
    # Run Fortran FUSE
    print("\n5. Running Fortran FUSE...")
    if FORTRAN_EXE.exists():
        result = subprocess.run(
            [str(FORTRAN_EXE), str(FM_PATH), BASIN_ID, 'run_def'],
            capture_output=True,
            text=True,
            cwd=str(BASE_PATH)
        )
        
        if result.returncode != 0:
            print(f"   Fortran FUSE error: {result.stderr[:500]}")
        else:
            print(f"   Fortran FUSE completed successfully")
            
        # Load Fortran output
        output_path = Path(fm['output_path'])
        fortran_output = output_path / f"{BASIN_ID}_{fm['fmodel_id']}_runs_def.nc"
        
        if fortran_output.exists():
            print(f"\n6. Comparing outputs...")
            try:
                import xarray as xr
                ds = xr.open_dataset(fortran_output)
                print(f"   Fortran output variables: {list(ds.data_vars)}")
                
                if 'q_routed' in ds:
                    runoff_fortran = ds['q_routed'].values.squeeze()
                elif 'q_instnt' in ds:
                    runoff_fortran = ds['q_instnt'].values.squeeze()
                else:
                    # Try to find any runoff variable
                    for var in ds.data_vars:
                        if 'q' in var.lower() or 'runoff' in var.lower():
                            runoff_fortran = ds[var].values.squeeze()
                            print(f"   Using variable: {var}")
                            break
                    else:
                        print(f"   No runoff variable found!")
                        ds.close()
                        return
                
                ds.close()
                
                # Handle masked/fill values
                runoff_fortran = np.where(runoff_fortran < -9000, np.nan, runoff_fortran)
                
                print(f"   Fortran runoff: mean={np.nanmean(runoff_fortran):.3f}, max={np.nanmax(runoff_fortran):.3f} mm/day")
                
                # Compare ROUTED runoff (dFUSE routed vs Fortran q_routed)
                min_len = min(len(runoff_dfuse_routed), len(runoff_fortran))
                rd = runoff_dfuse_routed[:min_len]
                rf = runoff_fortran[:min_len]
                
                # Remove NaNs
                valid = ~(np.isnan(rd) | np.isnan(rf))
                rd_valid = rd[valid]
                rf_valid = rf[valid]
                
                # Compute comparison metrics
                if len(rd_valid) > 0:
                    rmse = np.sqrt(np.mean((rd_valid - rf_valid)**2))
                    mae = np.mean(np.abs(rd_valid - rf_valid))
                    corr = np.corrcoef(rd_valid, rf_valid)[0, 1] if len(rd_valid) > 1 else 0
                    bias = np.mean(rd_valid - rf_valid)
                    
                    # NSE (treating Fortran as "observed")
                    ss_res = np.sum((rd_valid - rf_valid)**2)
                    ss_tot = np.sum((rf_valid - np.mean(rf_valid))**2)
                    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    
                    print(f"\n   Comparison Metrics (ROUTED runoff):")
                    print(f"   ─────────────────────────────")
                    print(f"   Valid points: {len(rd_valid)}")
                    print(f"   RMSE: {rmse:.4f} mm/day")
                    print(f"   MAE: {mae:.4f} mm/day")
                    print(f"   Correlation: {corr:.4f}")
                    print(f"   NSE: {nse:.4f}")
                    print(f"   Bias (dFUSE - Fortran): {bias:.4f} mm/day")
                    
                    # Also compare instantaneous runoff if q_instnt is available
                    if 'q_instnt' in ds:
                        q_instnt = ds['q_instnt'].values.squeeze()
                        q_instnt = np.where(q_instnt < -9000, np.nan, q_instnt)
                        
                        min_len_i = min(len(runoff_dfuse_instnt), len(q_instnt))
                        rd_i = runoff_dfuse_instnt[:min_len_i]
                        rf_i = q_instnt[:min_len_i]
                        valid_i = ~(np.isnan(rd_i) | np.isnan(rf_i))
                        
                        if np.sum(valid_i) > 0:
                            corr_i = np.corrcoef(rd_i[valid_i], rf_i[valid_i])[0, 1]
                            nse_i = 1 - np.sum((rd_i[valid_i] - rf_i[valid_i])**2) / np.sum((rf_i[valid_i] - np.mean(rf_i[valid_i]))**2)
                            print(f"\n   Comparison Metrics (INSTANTANEOUS runoff):")
                            print(f"   ─────────────────────────────")
                            print(f"   Correlation: {corr_i:.4f}")
                            print(f"   NSE: {nse_i:.4f}")
                    
                    # Also compare state variables if available
                    print(f"\n   State Variable Analysis:")
                    print(f"   ─────────────────────────────")
                    
                    # SWE comparison
                    if 'swe_tot' in ds:
                        swe_fortran = ds['swe_tot'].values.squeeze()
                        swe_fortran = np.where(swe_fortran < -9000, np.nan, swe_fortran)
                        print(f"   Fortran SWE: mean={np.nanmean(swe_fortran):.1f}, max={np.nanmax(swe_fortran):.1f} mm")
                        
                        # Compare with dFUSE SWE if we have trajectory
                        if swe_basin_avg is not None:
                            min_len_swe = min(len(swe_basin_avg), len(swe_fortran))
                            swe_corr = np.corrcoef(swe_basin_avg[:min_len_swe], swe_fortran[:min_len_swe])[0, 1]
                            swe_bias = np.mean(swe_basin_avg[:min_len_swe] - swe_fortran[:min_len_swe])
                            print(f"   dFUSE SWE:   mean={np.mean(swe_basin_avg):.1f}, max={np.max(swe_basin_avg):.1f} mm")
                            print(f"   SWE correlation: {swe_corr:.4f}, bias: {swe_bias:+.1f} mm")
                    
                    # Soil water comparison  
                    if 'watr_1' in ds:
                        watr1_fortran = ds['watr_1'].values.squeeze()
                        watr1_fortran = np.where(watr1_fortran < -9000, np.nan, watr1_fortran)
                        print(f"   Fortran watr_1: mean={np.nanmean(watr1_fortran):.1f}, max={np.nanmax(watr1_fortran):.1f} mm")
                    
                    if 'watr_2' in ds:
                        watr2_fortran = ds['watr_2'].values.squeeze()
                        watr2_fortran = np.where(watr2_fortran < -9000, np.nan, watr2_fortran)
                        print(f"   Fortran watr_2: mean={np.nanmean(watr2_fortran):.1f}, max={np.nanmax(watr2_fortran):.1f} mm")
                    
                    # Effective precipitation (after snow)
                    if 'eff_ppt' in ds:
                        eff_ppt_fortran = ds['eff_ppt'].values.squeeze()
                        eff_ppt_fortran = np.where(eff_ppt_fortran < -9000, np.nan, eff_ppt_fortran)
                        print(f"   Fortran eff_ppt: mean={np.nanmean(eff_ppt_fortran):.2f}, max={np.nanmax(eff_ppt_fortran):.2f} mm/day")
                    
                    # Compare q_instnt vs q_routed
                    if 'q_instnt' in ds:
                        q_instnt = ds['q_instnt'].values.squeeze()
                        q_instnt = np.where(q_instnt < -9000, np.nan, q_instnt)
                        print(f"   Fortran q_instnt: mean={np.nanmean(q_instnt):.3f} mm/day (unrouted)")
                        print(f"   Fortran q_routed: mean={np.nanmean(runoff_fortran):.3f} mm/day (routed)")
                    
                    # Flow components
                    if 'qsurf' in ds:
                        qsurf = ds['qsurf'].values.squeeze()
                        qsurf = np.where(qsurf < -9000, np.nan, qsurf)
                        print(f"   Fortran qsurf: mean={np.nanmean(qsurf):.4f} mm/day")
                    
                    if 'qbase_2' in ds:
                        qbase = ds['qbase_2'].values.squeeze()
                        qbase = np.where(qbase < -9000, np.nan, qbase)
                        print(f"   Fortran qbase_2: mean={np.nanmean(qbase):.4f} mm/day")
                    
                    if 'qintf_1' in ds:
                        qintf = ds['qintf_1'].values.squeeze()
                        qintf = np.where(qintf < -9000, np.nan, qintf)
                        print(f"   Fortran qintf_1: mean={np.nanmean(qintf):.4f} mm/day")
                    
                    if 'qperc_12' in ds:
                        qperc = ds['qperc_12'].values.squeeze()
                        qperc = np.where(qperc < -9000, np.nan, qperc)
                        print(f"   Fortran qperc_12: mean={np.nanmean(qperc):.4f} mm/day")
                    
                    # Fortran FUSE numerical solver diagnostics
                    print(f"\n   Fortran FUSE Solver Diagnostics:")
                    print(f"   ─────────────────────────────")
                    if 'num_funcs' in ds:
                        num_funcs = ds['num_funcs'].values.squeeze()
                        print(f"   num_funcs: mean={np.nanmean(num_funcs):.1f} (RHS evaluations per step)")
                    if 'numjacobian' in ds:
                        numjac = ds['numjacobian'].values.squeeze()
                        print(f"   numjacobian: mean={np.nanmean(numjac):.1f} (Jacobian evaluations)")
                    if 'sub_accept' in ds:
                        sub_acc = ds['sub_accept'].values.squeeze()
                        print(f"   sub_accept: mean={np.nanmean(sub_acc):.1f} (accepted sub-steps)")
                    if 'sub_reject' in ds:
                        sub_rej = ds['sub_reject'].values.squeeze()
                        print(f"   sub_reject: mean={np.nanmean(sub_rej):.1f} (rejected sub-steps)")
                    if 'sub_noconv' in ds:
                        sub_noconv = ds['sub_noconv'].values.squeeze()
                        print(f"   sub_noconv: total={np.nansum(sub_noconv):.0f} (non-convergent steps)")
                    if 'max_iterns' in ds:
                        max_iter = ds['max_iterns'].values.squeeze()
                        print(f"   max_iterns: max={np.nanmax(max_iter):.0f} (max Newton iterations)")
                    
                    # Check routing delay in Fortran
                    if 'q_instnt' in ds:
                        q_inst = ds['q_instnt'].values.squeeze()
                        q_inst = np.where(q_inst < -9000, np.nan, q_inst)
                        routing_diff = np.nanmean(np.abs(q_inst - runoff_fortran))
                        if routing_diff > 0.01:
                            print(f"\n   NOTE: Fortran gamma routing active (mean |q_instnt-q_routed|={routing_diff:.4f})")
                            print(f"   dFUSE routing applied with shape={GAMMA_SHAPE}, delay={fortran_params.TIMEDELAY:.2f} days")
                    
                    # Plot if matplotlib available
                    try:
                        import matplotlib.pyplot as plt
                        
                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                        
                        # Time series - full
                        ax = axes[0, 0]
                        ax.plot(rf_valid, label='Fortran FUSE', alpha=0.7, linewidth=0.5)
                        ax.plot(rd_valid, label='dFUSE', alpha=0.7, linewidth=0.5)
                        ax.set_xlabel('Day')
                        ax.set_ylabel('Runoff [mm/day]')
                        ax.set_title('Full Simulation Period')
                        ax.legend()
                        
                        # Time series - first year detail
                        ax = axes[0, 1]
                        n_show = min(365, len(rd_valid))
                        ax.plot(rf_valid[:n_show], label='Fortran FUSE', alpha=0.8)
                        ax.plot(rd_valid[:n_show], label='dFUSE', alpha=0.8)
                        ax.set_xlabel('Day')
                        ax.set_ylabel('Runoff [mm/day]')
                        ax.set_title('First Year Detail')
                        ax.legend()
                        
                        # Scatter plot
                        ax = axes[1, 0]
                        ax.scatter(rf_valid, rd_valid, alpha=0.3, s=3)
                        max_val = max(np.percentile(rf_valid, 99), np.percentile(rd_valid, 99))
                        ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
                        ax.set_xlabel('Fortran FUSE [mm/day]')
                        ax.set_ylabel('dFUSE [mm/day]')
                        ax.set_title(f'Scatter Plot (R²={corr**2:.3f})')
                        ax.legend()
                        ax.set_xlim(0, max_val)
                        ax.set_ylim(0, max_val)
                        
                        # Flow duration curves
                        ax = axes[1, 1]
                        exceedance = np.linspace(0, 100, len(rf_valid))
                        ax.semilogy(exceedance, np.sort(rf_valid)[::-1], label='Fortran FUSE', linewidth=1.5)
                        ax.semilogy(exceedance, np.sort(rd_valid)[::-1], label='dFUSE', linewidth=1.5)
                        ax.set_xlabel('Exceedance Probability [%]')
                        ax.set_ylabel('Runoff [mm/day]')
                        ax.set_title('Flow Duration Curve')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        # Save
                        plot_path = output_path / f'{BASIN_ID}_dfuse_comparison.png'
                        plt.savefig(plot_path, dpi=150)
                        print(f"\n   Plot saved: {plot_path}")
                        plt.show()
                        
                    except ImportError:
                        print("   (matplotlib not available for plotting)")
                        
                else:
                    print("   No valid overlapping data points!")
                    
            except ImportError:
                print("   xarray not available, cannot load Fortran output")
        else:
            print(f"   Fortran output not found: {fortran_output}")
    else:
        print(f"   Fortran executable not found: {FORTRAN_EXE}")
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
