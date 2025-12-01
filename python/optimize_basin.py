import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube
from sklearn.ensemble import RandomForestRegressor

from dfuse import FUSE, FUSEConfig, VIC_CONFIG, TOPMODEL_CONFIG, PARAM_NAMES, PARAM_BOUNDS
from dfuse_netcdf import read_fuse_forcing, parse_file_manager

# ==========================================
# CONFIGURATION
# ==========================================
BASIN_ID = "Klondike_Bonanza_Creek"
BASE_PATH = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Klondike_Bonanza_Creek")
FM_PATH = BASE_PATH / "settings/FUSE/fm_catch.txt"

DEVICE = "cpu"
WARMUP_DAYS = 365 
NUM_TRAIN_SAMPLES = 3000   # Per structure (Total 6000)
NUM_SYNTH_SAMPLES = 50000 

# ==========================================
# HELPER UTILS
# ==========================================
def inverse_sigmoid(val, low, high):
    eps = 1e-5
    val = np.clip(val, low+eps, high-eps)
    norm = (val - low) / (high - low)
    return np.log(norm / (1 - norm))

def inverse_sigmoid_batch(vals, names):
    raw = torch.zeros_like(vals)
    for i, name in enumerate(names):
        low, high = PARAM_BOUNDS[name]
        norm = (vals[i] - low) / (high - low)
        norm = torch.clamp(norm, 0.001, 0.999)
        raw[i] = torch.log(norm / (1.0 - norm))
    return raw

# ==========================================
# MODEL COMPONENTS
# ==========================================
class GammaRouting(nn.Module):
    def __init__(self, max_length=60):
        super().__init__()
        self.max_length = max_length
        self.raw_delay = nn.Parameter(torch.tensor(0.0)) 
        self.raw_shape = nn.Parameter(torch.tensor(0.0))

    @property
    def delay(self):
        return 0.1 + 29.9 * torch.sigmoid(self.raw_delay) # 30 days max

    @property
    def shape(self):
        return 0.5 + 9.5 * torch.sigmoid(self.raw_shape)

    def get_kernel(self):
        device = self.raw_delay.device
        t = torch.arange(1, self.max_length + 1, dtype=torch.float32, device=device)
        alpha = self.shape
        theta = self.delay / alpha
        log_kernel = (alpha - 1) * torch.log(t) - (t / theta) - alpha * torch.log(theta) - torch.lgamma(alpha)
        kernel = torch.exp(log_kernel)
        return kernel / (kernel.sum() + 1e-8)

    def forward(self, runoff):
        runoff_in = runoff.view(1, 1, -1)
        kernel = self.get_kernel().view(1, 1, -1)
        padding = self.max_length - 1
        runoff_padded = torch.nn.functional.pad(runoff_in, (padding, 0))
        return torch.nn.functional.conv1d(runoff_padded, kernel).view(-1)

class CalibratableFUSE(nn.Module):
    def __init__(self, config, init_dict=None):
        super().__init__()
        self.fuse = FUSE(config=config, learnable_params=True)
        self.routing = GammaRouting()
        
        if init_dict:
            with torch.no_grad():
                self.fuse.raw_params.data = inverse_sigmoid_batch(init_dict['phys_params'], PARAM_NAMES)
                
            raw_delay = inverse_sigmoid(init_dict['delay'], 0.1, 30.0)
            raw_shape = inverse_sigmoid(init_dict['shape'], 0.5, 10.0)
            raw_pmult = inverse_sigmoid(init_dict['p_mult'], 0.5, 2.5)
            
            self.raw_precip_mult = nn.Parameter(torch.tensor(raw_pmult, dtype=torch.float32))
            self.routing.raw_delay = nn.Parameter(torch.tensor(raw_delay, dtype=torch.float32))
            self.routing.raw_shape = nn.Parameter(torch.tensor(raw_shape, dtype=torch.float32))
        else:
            self.raw_precip_mult = nn.Parameter(torch.tensor(-0.85))
            
        self.raw_init_frac = nn.Parameter(torch.tensor([-1.0, -1.0, -5.0]))

    @property
    def precip_mult(self):
        return 0.5 + 2.0 * torch.sigmoid(self.raw_precip_mult)

    def forward(self, forcing):
        forcing_adj = forcing.clone()
        forcing_adj[:, 0] = forcing[:, 0] * self.precip_mult
        
        params = self.fuse.params
        s1_max, s2_max = params[0], params[1]
        fracs = torch.sigmoid(self.raw_init_frac)
        
        initial_state = self.fuse.get_initial_state(
            S1_init=fracs[0] * s1_max, 
            S2_init=fracs[1] * s2_max, 
            SWE_init=fracs[2] * 200.0
        ).to(forcing.device)

        instant_runoff = self.fuse(initial_state, forcing_adj)
        return self.routing(instant_runoff)

# ==========================================
# SURROGATE OPTIMIZATION LOGIC
# ==========================================
def apply_gamma_routing_numpy(q, delay, shape):
    t = np.arange(1, 61)
    alpha = shape
    theta = delay / alpha
    try:
        log_k = (alpha-1)*np.log(t) - (t/theta) - alpha*np.log(theta) - torch.lgamma(torch.tensor(alpha)).item()
        kernel = np.exp(log_k)
        kernel = kernel / np.sum(kernel)
    except:
        return q
    return np.convolve(q, kernel)[:len(q)]

def generate_lhs_params(n_samples):
    sampler = LatinHypercube(d=len(PARAM_NAMES) + 3)
    sample = sampler.random(n=n_samples)
    
    phys_params = np.zeros((n_samples, len(PARAM_NAMES)))
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        phys_params[:, i] = low + sample[:, i] * (high - low)
        
    delay = 0.1 + sample[:, -3] * 29.9 
    shape = 1.0 + sample[:, -2] * 4.0
    p_mult = 0.5 + sample[:, -1] * 1.0 # 0.5 to 1.5 range
    
    return phys_params, delay, shape, p_mult

def run_training_corpus(config, forcing, obs, n_samples):
    # Quiet tqdm
    phys_params, delays, shapes, p_mults = generate_lhs_params(n_samples)
    forcing_np = forcing.cpu().numpy()
    obs_valid = obs.cpu().numpy()[WARMUP_DAYS:]
    obs_var = np.sum((obs_valid - np.nanmean(obs_valid))**2)
    
    base_model = FUSE(config=config, learnable_params=False).to(DEVICE)
    X, y = [], []
    
    for i in range(n_samples):
        current_forcing = forcing.clone()
        current_forcing[:, 0] *= float(p_mults[i])
        
        s1_max = phys_params[i, 0]
        s2_max = phys_params[i, 1]
        f_tens = phys_params[i, 2]
        init_state = torch.tensor([s1_max*f_tens, s2_max*0.3, 0.0], device=DEVICE)
        
        base_model.raw_params = nn.Parameter(inverse_sigmoid_batch(torch.tensor(phys_params[i]), PARAM_NAMES))
        
        try:
            with torch.no_grad():
                q_inst = base_model(init_state, current_forcing).cpu().numpy()
            
            if np.isnan(q_inst).any() or np.max(q_inst) > 1000: continue

            q_routed = apply_gamma_routing_numpy(q_inst, delays[i], shapes[i])
            sim = q_routed[WARMUP_DAYS:]
            sse = np.sum((sim - obs_valid)**2)
            nse = 1.0 - (sse / (obs_var + 1e-6))
            
            if nse > -1.0: 
                features = np.concatenate([phys_params[i], [delays[i], shapes[i], p_mults[i]]])
                X.append(features)
                y.append(nse)
        except: continue
        
    return np.array(X), np.array(y)

def get_candidates(X_train, y_train, n_synth=50000, top_k=3):
    y_train_clipped = np.clip(y_train, -1.0, 1.0)
    regr = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1)
    regr.fit(X_train, y_train_clipped)
    
    phys, delay, shape, p_mult = generate_lhs_params(n_synth)
    X_synth = np.column_stack([phys, delay, shape, p_mult])
    y_pred = regr.predict(X_synth)
    
    top_indices = np.argsort(y_pred)[-top_k:][::-1]
    candidates = []
    for idx in top_indices:
        candidates.append({
            'phys_params': torch.tensor(phys[idx], dtype=torch.float32),
            'delay': float(delay[idx]),
            'shape': float(shape[idx]),
            'p_mult': float(p_mult[idx]),
            'pred_nse': y_pred[idx]
        })
    return candidates

# ==========================================
# GRADIENT REFINEMENT
# ==========================================
def kge_loss(pred, target):
    mask = ~torch.isnan(target) & (target >= 0)
    p, t = pred[mask], target[mask]
    r = torch.corrcoef(torch.stack([p, t]))[0, 1]
    alpha = torch.std(p) / (torch.std(t) + 1e-6)
    beta = torch.mean(p) / (torch.mean(t) + 1e-6)
    return torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def nse_loss(pred, target):
    mask = ~torch.isnan(target) & (target >= 0)
    p, t = pred[mask], target[mask]
    return torch.sum((p - t)**2) / (torch.sum((t - t.mean())**2) + 1e-6)

def refine_candidate(config, init_params, forcing, obs, name):
    model = CalibratableFUSE(config=config, init_dict=init_params).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    pbar = tqdm(range(600), desc=f"Refining {name}", leave=False)
    best_loss = float('inf')
    best_state = None
    
    for i in pbar:
        optimizer.zero_grad()
        runoff = model(forcing)
        
        # 300 epochs KGE, 300 epochs NSE
        if i < 300: loss = kge_loss(runoff[WARMUP_DAYS:], obs[WARMUP_DAYS:])
        else: loss = nse_loss(runoff[WARMUP_DAYS:], obs[WARMUP_DAYS:])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())
        
        with torch.no_grad():
            # Check NSE for best model saving
            nse = 1.0 - nse_loss(runoff[WARMUP_DAYS:], obs[WARMUP_DAYS:]).item()
            
        # Minimize negative NSE
        if -nse < best_loss:
            best_loss = -nse
            best_state = model.state_dict()
            
        pbar.set_description(f"NSE: {nse:.3f}")
    
    model.load_state_dict(best_state)
    return 1.0 - nse_loss(model(forcing)[WARMUP_DAYS:], obs[WARMUP_DAYS:]).item(), best_state

# ==========================================
# MAIN
# ==========================================
def main():
    fm = parse_file_manager(FM_PATH)
    forcing_path = Path(fm['input_path']) / f"{BASIN_ID}{fm['suffix_forcing']}"
    fuse_data = read_fuse_forcing(forcing_path)
    forcing = fuse_data.to_tensor(device=DEVICE)
    obs = torch.tensor(fuse_data.q_obs, dtype=torch.float32).to(DEVICE)
    obs[obs < 0] = float('nan')

    print(f"Observed Mean: {torch.nanmean(obs):.3f} mm/day")

    final_results = []

    # --- TOURNAMENT LOOP ---
    structures = [
        (VIC_CONFIG, "VIC"),
        (TOPMODEL_CONFIG, "TOPMODEL")
    ]

    for config, name in structures:
        print(f"\n>> PROCESSING STRUCTURE: {name}")
        
        # 1. Generate & Train RF
        X, y = run_training_corpus(config, forcing, obs, NUM_TRAIN_SAMPLES)
        candidates = get_candidates(X, y, top_k=3)
        
        # 2. Refine Top Candidates
        for i, cand in enumerate(candidates):
            nse, state = refine_candidate(config, cand, forcing, obs, f"{name} #{i+1}")
            final_results.append({
                'name': f"{name} #{i+1}",
                'nse': nse,
                'state': state,
                'config': config
            })
            print(f"   {name} #{i+1}: NSE = {nse:.4f} (P_mult={cand['p_mult']:.2f})")

    # Pick Overall Winner
    best_run = max(final_results, key=lambda x: x['nse'])
    print(f"\n>> CHAMPION: {best_run['name']} with NSE {best_run['nse']:.4f}")
    
    # Load Winner for Analysis
    model = CalibratableFUSE(config=best_run['config']).to(DEVICE)
    model.load_state_dict(best_run['state'])
    
    with torch.no_grad():
        final_runoff = model(forcing).cpu().numpy()
        
    # Water Balance Check
    p_mult = model.precip_mult.item()
    precip_in = forcing[:, 0].mean().item() * p_mult
    runoff_out = final_runoff.mean()
    et_approx = precip_in - runoff_out
    
    print(f"\nFinal Water Balance:")
    print(f"  Input Precip: {precip_in:.2f} mm/day (Mult: {p_mult:.2f})")
    print(f"  Runoff:       {runoff_out:.2f} mm/day")
    print(f"  Approx ET:    {et_approx:.2f} mm/day")
    
    # Plot
    plot_obs = obs.cpu().numpy()[WARMUP_DAYS:]
    plot_sim = final_runoff[WARMUP_DAYS:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_obs, color='grey', label='Observed', alpha=0.6)
    plt.plot(plot_sim, color='#ff7f0e', label=f'dFUSE Best ({best_run["name"]})')
    plt.legend()
    plt.title(f"dFUSE Tournament Winner: {best_run['name']} (NSE={best_run['nse']:.2f})")
    plt.tight_layout()
    plt.savefig("dfuse_calibration_final.png")
    print("Saved dfuse_calibration_final.png")

if __name__ == "__main__":
    main()