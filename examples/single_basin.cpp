/**
 * @file single_basin.cpp
 * @brief Example: Single basin simulation with cFUSE
 * 
 * This example demonstrates:
 * 1. Setting up model configuration
 * 2. Initializing states and parameters
 * 3. Running a year-long simulation
 * 4. Computing performance metrics
 */

#include <cfuse/dfuse.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace dfuse;

/**
 * @brief Generate synthetic forcing data
 */
std::vector<Forcing> generate_forcing(int n_days, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<Real> temp_noise(0, 3);
    std::exponential_distribution<Real> storm_amount(0.07);
    std::bernoulli_distribution storm_event(0.15);
    
    std::vector<Forcing> forcing(n_days);
    
    for (int d = 0; d < n_days; ++d) {
        Real phase = 2.0 * M_PI * d / 365.0;
        
        // Temperature: seasonal cycle with noise
        Real temp = 15.0 + 12.0 * std::sin(phase - M_PI/2) + temp_noise(gen);
        
        // PET: follows temperature
        Real pet = 2.0 + 3.0 / (1.0 + std::exp(-(temp - 10) / 5));
        
        // Precipitation: seasonal base + storms
        Real precip = 3.0 + 2.0 * std::sin(phase + M_PI);
        if (storm_event(gen)) {
            precip += storm_amount(gen);
        }
        precip = std::max(Real(0), precip);
        
        forcing[d] = Forcing(precip, pet, temp);
    }
    
    return forcing;
}

/**
 * @brief Compute Nash-Sutcliffe Efficiency
 */
Real compute_nse(const std::vector<Real>& sim, const std::vector<Real>& obs) {
    Real mean_obs = 0;
    for (Real o : obs) mean_obs += o;
    mean_obs /= obs.size();
    
    Real ss_res = 0, ss_tot = 0;
    for (size_t i = 0; i < obs.size(); ++i) {
        ss_res += (sim[i] - obs[i]) * (sim[i] - obs[i]);
        ss_tot += (obs[i] - mean_obs) * (obs[i] - mean_obs);
    }
    
    return 1.0 - ss_res / (ss_tot + 1e-10);
}

/**
 * @brief Run simulation and print diagnostics
 */
void run_simulation(
    const char* name,
    const ModelConfig& config,
    const std::vector<Forcing>& forcing,
    const Parameters& params
) {
    printf("\n%s\n", name);
    printf("====================================\n");
    
    int n_days = forcing.size();
    
    // Initialize state
    State state;
    state.S1 = 150.0;
    state.S2 = 800.0;
    state.SWE = 0.0;
    state.sync_derived(config);
    
    // Storage for outputs
    std::vector<Real> runoff(n_days);
    std::vector<Real> et(n_days);
    
    // Accumulate totals
    Real total_precip = 0, total_runoff = 0, total_et = 0;
    Real max_runoff = 0;
    int peak_day = 0;
    
    // Run simulation
    Flux flux;
    for (int d = 0; d < n_days; ++d) {
        fuse_step(state, forcing[d], params, config, 1.0, flux);
        
        runoff[d] = flux.q_total;
        et[d] = flux.e_total;
        
        total_precip += forcing[d].precip;
        total_runoff += flux.q_total;
        total_et += flux.e_total;
        
        if (flux.q_total > max_runoff) {
            max_runoff = flux.q_total;
            peak_day = d;
        }
    }
    
    // Compute final storage
    Real final_storage = state.S1 + state.S2 + state.SWE;
    Real initial_storage = 150.0 + 800.0 + 0.0;
    Real storage_change = final_storage - initial_storage;
    
    // Water balance
    Real balance_error = total_precip - total_runoff - total_et - storage_change;
    
    // Print results
    printf("Water Balance (mm/year):\n");
    printf("  Precipitation:    %7.1f\n", total_precip);
    printf("  Total runoff:     %7.1f (%.0f%%)\n", total_runoff, 100*total_runoff/total_precip);
    printf("  Evapotranspiration: %5.1f (%.0f%%)\n", total_et, 100*total_et/total_precip);
    printf("  Storage change:   %7.1f\n", storage_change);
    printf("  Balance error:    %7.2f (%.2f%%)\n", balance_error, 100*std::abs(balance_error)/total_precip);
    printf("\n");
    printf("Peak flow: %.1f mm/day on day %d\n", max_runoff, peak_day);
    printf("Runoff ratio: %.2f\n", total_runoff / total_precip);
    printf("\n");
    printf("Final states:\n");
    printf("  Upper layer (S1): %.1f mm\n", state.S1);
    printf("  Lower layer (S2): %.1f mm\n", state.S2);
    printf("  Snow (SWE):       %.1f mm\n", state.SWE);
}

int main() {
    printf("========================================\n");
    printf("cFUSE Single Basin Example\n");
    printf("========================================\n");
    
    // Generate forcing
    int n_days = 365;
    auto forcing = generate_forcing(n_days);
    
    printf("\nForcing summary (%d days):\n", n_days);
    Real sum_p = 0, sum_pet = 0, sum_t = 0;
    for (const auto& f : forcing) {
        sum_p += f.precip;
        sum_pet += f.pet;
        sum_t += f.temp;
    }
    printf("  Total precipitation: %.0f mm\n", sum_p);
    printf("  Total PET: %.0f mm\n", sum_pet);
    printf("  Mean temperature: %.1f °C\n", sum_t / n_days);
    
    // Common parameters (will use different ones for different models)
    Parameters params;
    params.S1_max = 400.0;
    params.S2_max = 1500.0;
    params.f_tens = 0.5;
    params.f_rchr = 0.5;
    params.f_base = 0.6;
    params.r1 = 0.7;
    params.ku = 12.0;
    params.c = 2.0;
    params.alpha = 50.0;
    params.psi = 2.0;
    params.kappa = 0.4;
    params.ki = 8.0;
    params.ks = 40.0;
    params.n = 2.0;
    params.v = 0.05;
    params.v_A = 0.12;
    params.v_B = 0.025;
    params.Ac_max = 0.75;
    params.b = 1.5;
    params.lambda = 7.0;
    params.chi = 3.0;
    params.mu_t = 1.0;
    params.T_rain = 1.5;
    params.T_melt = 0.0;
    params.melt_rate = 3.5;
    params.smooth_frac = 0.01;
    params.lambda_n = 10.0;
    params.compute_derived();
    
    // Run each parent model configuration
    run_simulation("VIC Model", models::VIC, forcing, params);
    run_simulation("TOPMODEL", models::TOPMODEL, forcing, params);
    run_simulation("Sacramento Model", models::SACRAMENTO, forcing, params);
    run_simulation("PRMS Model", models::PRMS, forcing, params);
    
    printf("\n========================================\n");
    printf("Example completed successfully!\n");
    printf("========================================\n");
    
    return 0;
}
