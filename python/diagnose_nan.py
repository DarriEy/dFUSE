"""
Diagnostic script to identify NaN gradient sources in dFUSE Enzyme AD
"""
import numpy as np
import dfuse_core
from pathlib import Path
from dfuse import FUSEConfig, VIC_CONFIG
from dfuse_netcdf import read_fuse_forcing, parse_file_manager, read_elevation_bands

# --- CONFIG ---
BASIN_ID = "Klondike_Bonanza_Creek"
BASE_PATH = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Klondike_Bonanza_Creek")
FM_PATH = BASE_PATH / "settings/FUSE/fm_catch.txt"

# --- PARAMETER DEFINITIONS ---
PARAM_NAMES = [
    'S1_max', 'S2_max', 'f_tens', 'f_rchr', 'f_base', 'r1',
    'ku', 'c', 'alpha', 'psi', 'kappa', 'ki',
    'ks', 'n', 'v', 'v_A', 'v_B',
    'Ac_max', 'b', 'lambda', 'chi', 'mu_t',
    'T_rain', 'T_melt', 'melt_rate', 'lapse_rate', 'opg',
    'MFMAX', 'MFMIN'
]

PARAM_BOUNDS = {
    'S1_max': (50.0, 5000.0), 'S2_max': (100.0, 10000.0),
    'f_tens': (0.05, 0.95), 'f_rchr': (0.05, 0.95), 'f_base': (0.05, 0.95),
    'r1': (0.05, 0.95), 'ku': (0.01, 1000.0), 'c': (1.0, 20.0),
    'alpha': (1.0, 250.0), 'psi': (1.0, 5.0), 'kappa': (0.05, 0.95),
    'ki': (0.01, 1000.0), 'ks': (0.001, 10000.0), 'n': (1.0, 10.0),
    'v': (0.001, 0.25), 'v_A': (0.001, 0.25), 'v_B': (0.001, 0.25),
    'Ac_max': (0.05, 0.95), 'b': (0.001, 3.0), 'lambda': (5.0, 10.0),
    'chi': (2.0, 5.0), 'mu_t': (0.01, 5.0), 'T_rain': (-2.0, 4.0),
    'T_melt': (-2.0, 4.0), 'melt_rate': (1.0, 10.0),
    'lapse_rate': (-9.8, 0.0), 'opg': (0.0, 1.0),
    'MFMAX': (1.0, 10.0), 'MFMIN': (0.0, 10.0)
}

def main():
    print("=" * 60)
    print("dFUSE Enzyme AD Diagnostic Tool")
    print("=" * 60)
    
    # Load data
    fm = parse_file_manager(FM_PATH)
    forcing = read_fuse_forcing(Path(fm['input_path']) / f"{BASIN_ID}{fm['suffix_forcing']}")
    elev_bands = read_elevation_bands(Path(fm['input_path']) / f"{BASIN_ID}{fm['suffix_elev_bands']}")
    
    f_np = forcing.to_tensor().numpy().astype(np.float32)
    obs = np.array(forcing.q_obs, dtype=np.float32)
    obs[obs < 0] = np.nan
    
    print(f"\nForcing shape: {f_np.shape}")
    print(f"Elevation bands: {len(elev_bands.area_frac)}")
    
    # Check forcing for NaN/Inf
    print(f"\nForcing NaN count: {np.isnan(f_np).sum()}")
    print(f"Forcing Inf count: {np.isinf(f_np).sum()}")
    print(f"Forcing min/max: {f_np.min():.4f} / {f_np.max():.4f}")
    
    # Set up parameters
    p0 = np.zeros(29, dtype=np.float32)
    p_map = {name: i for i, name in enumerate(PARAM_NAMES)}
    for n in PARAM_NAMES: 
        p0[p_map[n]] = sum(PARAM_BOUNDS[n]) / 2
    
    p0[p_map['S1_max']] = 100.0
    p0[p_map['S2_max']] = 1000.0
    p0[p_map['ks']] = 50.0
    p0[p_map['n']] = 5.0
    p0[p_map['ki']] = 500.0
    p0[p_map['opg']] = 0.5
    p0[p_map['lapse_rate']] = -5.0
    p0[p_map['mu_t']] = 0.9
    p0[p_map['MFMAX']] = 4.2
    p0[p_map['MFMIN']] = 2.4
    
    print("\nParameters:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name}: {p0[i]:.4f}")
    
    # Set up state and config
    config = VIC_CONFIG.to_dict()
    state = np.zeros(9 + 30, dtype=np.float32)
    state[0] = 50.0  # S1
    state[5] = 250.0  # S2
    
    # Step 1: Test forward pass (without routing)
    print("\n" + "=" * 60)
    print("Step 1: Testing forward pass (run_fuse_elevation_bands)")
    print("=" * 60)
    
    try:
        result = dfuse_core.run_fuse_elevation_bands(
            np.array([50.0, 250.0], dtype=np.float32),  # Simple state
            f_np, p0, config,
            elev_bands.area_frac.astype(np.float32),
            elev_bands.mean_elev.astype(np.float32),
            1018.0, None, 1.0, False, False, 1, "euler"
        )
        _, runoff, _ = result
        
        print(f"Runoff shape: {runoff.shape}")
        print(f"Runoff NaN count: {np.isnan(runoff).sum()}")
        print(f"Runoff Inf count: {np.isinf(runoff).sum()}")
        print(f"Runoff min/max: {np.nanmin(runoff):.4f} / {np.nanmax(runoff):.4f}")
        
        if np.isnan(runoff).any():
            print("WARNING: Forward pass produces NaN!")
            first_nan = np.where(np.isnan(runoff))[0][0]
            print(f"First NaN at timestep: {first_nan}")
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        return
    
    # Step 2: Test routing
    print("\n" + "=" * 60)
    print("Step 2: Testing routing")
    print("=" * 60)
    
    shape_param = 2.5
    delay_param = p0[p_map['mu_t']]
    
    routed = dfuse_core.route_runoff(runoff, shape_param, delay_param, 1.0)
    print(f"Routed NaN count: {np.isnan(routed).sum()}")
    print(f"Routed min/max: {np.nanmin(routed):.4f} / {np.nanmax(routed):.4f}")
    
    # Step 3: Compute loss gradient (NSE)
    print("\n" + "=" * 60)
    print("Step 3: Computing loss gradient")
    print("=" * 60)
    
    mask = ~np.isnan(obs)
    sse = np.sum((routed[mask] - obs[mask])**2)
    sst = np.sum((obs[mask] - obs[mask].mean())**2)
    nse = 1.0 - sse / sst
    print(f"NSE: {nse:.4f}")
    
    # Compute grad_runoff (dL/d_routed)
    grad_runoff = np.zeros_like(routed, dtype=np.float32)
    grad_runoff[mask] = 2.0 * (routed[mask] - obs[mask]) / sst
    
    print(f"grad_runoff NaN count: {np.isnan(grad_runoff).sum()}")
    print(f"grad_runoff min/max: {np.nanmin(grad_runoff):.6f} / {np.nanmax(grad_runoff):.6f}")
    
    # Step 4: Test numerical gradient
    print("\n" + "=" * 60)
    print("Step 4: Computing NUMERICAL gradient (finite differences)")
    print("=" * 60)
    
    try:
        num_grad, fwd_result = dfuse_core.compute_gradient_numerical_debug(
            state, f_np, p0, grad_runoff, config,
            elev_bands.area_frac.astype(np.float32),
            elev_bands.mean_elev.astype(np.float32),
            1018.0, shape_param, delay_param, 1.0, 1e-4
        )
        
        print(f"Forward pass result: {fwd_result:.6f}")
        print(f"Numerical gradient NaN count: {np.isnan(num_grad).sum()}")
        
        print("\nNumerical gradients:")
        for i, name in enumerate(PARAM_NAMES):
            val = num_grad[i]
            status = "NaN" if np.isnan(val) else f"{val:.6e}"
            print(f"  {name:12s}: {status}")
            
    except Exception as e:
        print(f"Numerical gradient FAILED: {e}")
    
    # Step 5: Test Enzyme gradient
    print("\n" + "=" * 60)
    print("Step 5: Computing ENZYME gradient (autodiff)")
    print("=" * 60)
    
    try:
        enzyme_grad = dfuse_core.compute_gradient_adjoint_bands(
            state, f_np, p0, grad_runoff, config,
            elev_bands.area_frac.astype(np.float32),
            elev_bands.mean_elev.astype(np.float32),
            1018.0, shape_param, delay_param, 1.0
        )
        
        print(f"Enzyme gradient NaN count: {np.isnan(enzyme_grad).sum()}")
        
        print("\nEnzyme gradients:")
        for i, name in enumerate(PARAM_NAMES):
            val = enzyme_grad[i]
            status = "NaN" if np.isnan(val) else f"{val:.6e}"
            print(f"  {name:12s}: {status}")
            
        # Compare
        if not np.isnan(num_grad).all() and not np.isnan(enzyme_grad).all():
            print("\nComparison (where both are finite):")
            for i, name in enumerate(PARAM_NAMES):
                if not np.isnan(num_grad[i]) and not np.isnan(enzyme_grad[i]):
                    rel_err = abs(enzyme_grad[i] - num_grad[i]) / (abs(num_grad[i]) + 1e-10)
                    print(f"  {name:12s}: num={num_grad[i]:.4e}, enz={enzyme_grad[i]:.4e}, rel_err={rel_err:.2%}")
                    
    except Exception as e:
        print(f"Enzyme gradient FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Diagnostic complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
