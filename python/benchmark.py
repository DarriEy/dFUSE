import time
import sys
import subprocess
import numpy as np
import torch
from pathlib import Path

# Import dFUSE components
sys.path.insert(0, str(Path(__file__).parent))
from dfuse_netcdf import (
    read_fuse_forcing, 
    read_elevation_bands,
    parse_fuse_decisions, 
    parse_file_manager,
    parse_fortran_constraints
)
from dfuse import FUSEConfig, UpperLayerArch, LowerLayerArch, BaseflowType, PercolationType, SurfaceRunoffType, EvaporationType, InterflowType
import dfuse_core

# --- CONFIGURATION ---
BASE_PATH = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Klondike_Bonanza_Creek")
FM_PATH = BASE_PATH / "settings/FUSE/fm_catch.txt"
FORTRAN_EXE = Path("/Users/darrieythorsson/compHydro/data/SYMFLUENCE_data/installs/fuse/bin/fuse.exe")
BASIN_ID = "Klondike_Bonanza_Creek"
N_FORTRAN_RUNS = 5       # Keep low, it's slow due to I/O
N_DFUSE_RUNS = 100       # Fast single runs
BATCH_SIZE = 1000        # Number of basins for batch test

def setup_dfuse():
    """Prepare dFUSE data in memory to simulate calibration workflow"""
    fm = parse_file_manager(FM_PATH)
    
    # Decisions & Config
    decisions = parse_fuse_decisions(Path(fm['setngs_path']) / fm['m_decisions'])
    cd = decisions.to_config_dict()
    
    # Parameters
    fortran_params = parse_fortran_constraints(Path(fm['setngs_path']) / fm['constraints'])
    params_np = fortran_params.to_dfuse_params()
    
    # Forcing
    forcing = read_fuse_forcing(Path(fm['input_path']) / f"{BASIN_ID}{fm['suffix_forcing']}")
    forcing_np = np.stack([forcing.precip, forcing.pet, forcing.temp], axis=1).astype(np.float32)
    forcing_np = np.nan_to_num(forcing_np)

    # Elevation Bands
    try:
        eb = read_elevation_bands(Path(fm['input_path']) / f"{BASIN_ID}{fm['suffix_elev_bands']}")
        use_bands = True
    except:
        use_bands = False
        eb = None

    # State
    n_states = 3 # Approximation for this setup
    state_np = np.zeros(n_states, dtype=np.float32)
    state_np[0] = fortran_params.MAXWATR_1 * 0.5
    state_np[1] = fortran_params.MAXWATR_2 * 0.25
    
    return cd, params_np, state_np, forcing_np, eb, use_bands

def benchmark_fortran():
    """Run Fortran executable via subprocess"""
    print(f"   Running Fortran FUSE ({N_FORTRAN_RUNS} times)...", end="", flush=True)
    timings = []
    
    for _ in range(N_FORTRAN_RUNS):
        start = time.perf_counter()
        subprocess.run(
            [str(FORTRAN_EXE), str(FM_PATH), BASIN_ID, 'run_def'],
            capture_output=True, cwd=str(BASE_PATH)
        )
        timings.append(time.perf_counter() - start)
    
    avg_time = np.mean(timings)
    print(f" Avg: {avg_time:.4f} s")
    return avg_time

def benchmark_dfuse_single(config, params, state, forcing, eb, use_bands, solver="euler"):
    """Run dFUSE Python binding in a loop"""
    print(f"   Running dFUSE [{solver}] ({N_DFUSE_RUNS} times)...", end="", flush=True)
    
    timings = []
    # Warmup
    if use_bands:
         dfuse_core.run_fuse_elevation_bands(
            state, forcing, params, config,
            eb.area_frac.astype(np.float32), eb.mean_elev.astype(np.float32),
            float(1018.0), None, 1.0, False, False, 1, solver
        )
    else:
        dfuse_core.run_fuse(state, forcing, params, config, 1.0)

    for _ in range(N_DFUSE_RUNS):
        # Reset state copy for fairness
        s_copy = state.copy()
        start = time.perf_counter()
        
        if use_bands:
            dfuse_core.run_fuse_elevation_bands(
                s_copy, forcing, params, config,
                eb.area_frac.astype(np.float32), eb.mean_elev.astype(np.float32),
                float(1018.0), None, 1.0, False, False, 1, solver
            )
        else:
            dfuse_core.run_fuse(s_copy, forcing, params, config, 1.0)
            
        timings.append(time.perf_counter() - start)
        
    avg_time = np.mean(timings)
    print(f" Avg: {avg_time:.4f} s")
    return avg_time

def benchmark_dfuse_batch(config, params, state, forcing):
    """Run dFUSE Batch (Multi-threaded C++)"""
    print(f"   Running dFUSE Batch (N={BATCH_SIZE})...", end="", flush=True)
    
    # Prepare Batch Data
    states_batch = np.tile(state, (BATCH_SIZE, 1))
    # Shared forcing, but independent states
    
    # Warmup
    dfuse_core.run_fuse_batch(states_batch, forcing, params, config, 1.0)
    
    start = time.perf_counter()
    dfuse_core.run_fuse_batch(states_batch, forcing, params, config, 1.0)
    duration = time.perf_counter() - start
    
    avg_per_sim = duration / BATCH_SIZE
    print(f" Total: {duration:.4f} s (Avg/Sim: {avg_per_sim:.6f} s)")
    return avg_per_sim

def main():
    print("="*60)
    print("dFUSE Performance Benchmark")
    print("="*60)
    
    # 1. Setup
    print("\n[1] Loading Data & Configuration...")
    try:
        config, params, state, forcing, eb, use_bands = setup_dfuse()
        print(f"    Forcing: {forcing.shape[0]} timesteps")
        print(f"    Bands: {len(eb.area_frac) if use_bands else 'None'}")
    except Exception as e:
        print(f"    Error setting up: {e}")
        return

    # 2. Fortran Benchmark
    print("\n[2] Benchmarking Fortran FUSE (Subprocess + IO)...")
    if FORTRAN_EXE.exists():
        t_fortran = benchmark_fortran()
    else:
        print("    Fortran executable not found, skipping.")
        t_fortran = 1.0 # Dummy for comparison

    # 3. dFUSE Benchmarks
    print("\n[3] Benchmarking dFUSE (In-Memory)...")
    
    # Implicit (Fair comparison)
    t_dfuse_implicit = benchmark_dfuse_single(config, params, state, forcing, eb, use_bands, solver="sundials_bdf")
    
    # Explicit (Fast approximation)
    t_dfuse_explicit = benchmark_dfuse_single(config, params, state, forcing, eb, use_bands, solver="euler")
    
    # 4. Batch Benchmark (Structural Advantage)
    print("\n[4] Benchmarking dFUSE Batch (C++ OpenMP)...")
    # Batch currently supports standard run, not bands in the binding demo, 
    # so we'll test the core physics engine throughput
    t_dfuse_batch = benchmark_dfuse_batch(config, params, state, forcing)
    
    # 5. Results
    print("\n" + "="*60)
    print(f"{'IMPLEMENTATION':<25} | {'TIME/RUN (s)':<12} | {'SPEEDUP (vs Fortran)':<20}")
    print("-" * 60)
    print(f"{'Fortran FUSE (IO)':<25} | {t_fortran:<12.4f} | {'1.0x':<20}")
    print(f"{'dFUSE (Implicit)':<25} | {t_dfuse_implicit:<12.4f} | {t_fortran/t_dfuse_implicit:<20.1f}")
    print(f"{'dFUSE (Explicit)':<25} | {t_dfuse_explicit:<12.4f} | {t_fortran/t_dfuse_explicit:<20.1f}")
    print(f"{'dFUSE (Batch/Core)':<25} | {t_dfuse_batch:<12.6f} | {t_fortran/t_dfuse_batch:<20.1f}")
    print("-" * 60)
    print("Note: Fortran time includes Disk I/O. dFUSE times are in-memory (Workflow Advantage).")

if __name__ == "__main__":
    main()