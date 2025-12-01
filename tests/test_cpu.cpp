/**
 * @file test_cpu.cpp
 * @brief Basic CPU tests for dFUSE
 */

#include <dfuse/dfuse.hpp>
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace dfuse;

/**
 * @brief Test VIC-style configuration
 */
void test_vic_single_step() {
    printf("Testing VIC single step... ");
    
    // Configuration
    ModelConfig config = models::VIC;
    
    // Initialize state
    State state;
    state.S1 = 200.0;   // mm in upper layer
    state.S2 = 1000.0;  // mm in lower layer
    state.SWE = 0.0;    // No snow
    state.sync_derived(config);
    
    // Parameters (reasonable defaults)
    Parameters params;
    params.S1_max = 500.0;
    params.S2_max = 2000.0;
    params.f_tens = 0.5;
    params.f_rchr = 0.5;
    params.f_base = 0.5;
    params.r1 = 0.7;
    params.ku = 10.0;
    params.c = 2.0;
    params.alpha = 10.0;
    params.psi = 1.5;
    params.kappa = 0.5;
    params.ki = 5.0;
    params.ks = 50.0;
    params.n = 2.0;
    params.v = 0.05;
    params.v_A = 0.1;
    params.v_B = 0.02;
    params.Ac_max = 0.8;
    params.b = 1.5;
    params.lambda = 7.0;
    params.chi = 3.0;
    params.mu_t = 1.0;
    params.T_rain = 1.0;
    params.T_melt = 0.0;
    params.melt_rate = 3.0;
    params.smooth_frac = 0.01;
    params.compute_derived();
    
    // Forcing (typical day)
    Forcing forcing(10.0, 4.0, 15.0);  // 10mm precip, 4mm PET, 15°C
    
    // Run single step
    Flux flux;
    Real dt = 1.0;  // 1 day
    
    fuse_step(state, forcing, params, config, dt, flux);
    
    // Basic sanity checks
    assert(flux.q_total >= 0.0);
    assert(flux.q_total < forcing.precip * 2);  // Runoff shouldn't exceed precip + storage
    assert(flux.e_total >= 0.0);
    assert(flux.e_total <= forcing.pet);  // Actual ET <= PET
    assert(flux.Ac >= 0.0 && flux.Ac <= 1.0);  // Saturated area is a fraction
    
    printf("PASSED\n");
    printf("  Runoff: %.2f mm/day\n", flux.q_total);
    printf("  ET: %.2f mm/day\n", flux.e_total);
    printf("  Sat. area: %.2f\n", flux.Ac);
}

/**
 * @brief Test TOPMODEL configuration
 */
void test_topmodel_single_step() {
    printf("Testing TOPMODEL single step... ");
    
    ModelConfig config = models::TOPMODEL;
    
    State state;
    state.S1 = 150.0;
    state.S2 = 800.0;
    state.SWE = 50.0;  // Some snow
    state.sync_derived(config);
    
    Parameters params;
    params.S1_max = 400.0;
    params.S2_max = 1500.0;
    params.f_tens = 0.6;
    params.r1 = 0.8;
    params.ku = 15.0;
    params.c = 3.0;
    params.ks = 30.0;
    params.n = 1.5;
    params.lambda = 6.5;
    params.chi = 2.5;
    params.lambda_n = 10.0;  // Pre-computed for testing
    params.mu_t = 1.5;
    params.Ac_max = 0.9;
    params.T_rain = 2.0;
    params.T_melt = 0.0;
    params.melt_rate = 4.0;
    params.smooth_frac = 0.01;
    params.compute_derived();
    
    // Cold day with snow
    Forcing forcing(8.0, 2.0, -5.0);  // Snow, low ET, cold
    
    Flux flux;
    fuse_step(state, forcing, params, config, 1.0, flux);
    
    // Should have snow accumulation, minimal melt
    assert(flux.rain < forcing.precip);  // Some should be snow
    assert(flux.melt < 10.0);  // Limited melt in cold
    assert(flux.q_total >= 0.0);
    
    printf("PASSED\n");
    printf("  Rain: %.2f, Melt: %.2f mm/day\n", flux.rain, flux.melt);
    printf("  Runoff: %.2f mm/day\n", flux.q_total);
}

/**
 * @brief Test Sacramento configuration
 */
void test_sacramento_single_step() {
    printf("Testing Sacramento single step... ");
    
    ModelConfig config = models::SACRAMENTO;
    
    State state;
    state.S1_T = 100.0;
    state.S1_F = 50.0;
    state.S2_T = 200.0;
    state.S2_FA = 300.0;
    state.S2_FB = 150.0;
    state.SWE = 0.0;
    state.sync_derived(config);
    
    Parameters params;
    params.S1_max = 300.0;
    params.S2_max = 1000.0;
    params.f_tens = 0.6;
    params.f_rchr = 0.5;
    params.f_base = 0.6;
    params.r1 = 0.7;
    params.ku = 20.0;
    params.c = 2.5;
    params.alpha = 50.0;
    params.psi = 2.0;
    params.kappa = 0.4;
    params.v_A = 0.15;
    params.v_B = 0.03;
    params.Ac_max = 0.7;
    params.T_rain = 1.0;
    params.melt_rate = 3.0;
    params.smooth_frac = 0.01;
    params.compute_derived();
    
    Forcing forcing(15.0, 5.0, 20.0);  // Wet warm day
    
    Flux flux;
    fuse_step(state, forcing, params, config, 1.0, flux);
    
    // Should have both baseflow components
    assert(flux.qb >= 0.0);
    assert(flux.q_total >= 0.0);
    
    printf("PASSED\n");
    printf("  Baseflow A: %.2f, B: %.2f mm/day\n", flux.qb_A, flux.qb_B);
    printf("  Total runoff: %.2f mm/day\n", flux.q_total);
}

/**
 * @brief Test time series simulation
 */
void test_time_series() {
    printf("Testing time series simulation... ");
    
    ModelConfig config = models::VIC;
    
    State state;
    state.S1 = 100.0;
    state.S2 = 500.0;
    state.SWE = 0.0;
    state.sync_derived(config);
    
    Parameters params;
    params.S1_max = 400.0;
    params.S2_max = 1500.0;
    params.f_tens = 0.5;
    params.r1 = 0.7;
    params.ku = 10.0;
    params.c = 2.0;
    params.ks = 40.0;
    params.n = 2.0;
    params.b = 1.5;
    params.Ac_max = 0.8;
    params.T_rain = 1.0;
    params.melt_rate = 3.0;
    params.smooth_frac = 0.01;
    params.compute_derived();
    
    // Simulate 365 days
    int n_days = 365;
    Real total_precip = 0.0;
    Real total_runoff = 0.0;
    Real total_et = 0.0;
    
    Flux flux;
    for (int d = 0; d < n_days; ++d) {
        // Sinusoidal forcing (seasonal pattern)
        Real phase = 2.0 * M_PI * d / 365.0;
        Real precip = 5.0 + 4.0 * std::sin(phase);  // 1-9 mm/day
        Real pet = 3.0 + 2.0 * std::sin(phase);     // 1-5 mm/day
        Real temp = 15.0 + 10.0 * std::sin(phase);  // 5-25 °C
        
        Forcing forcing(precip, pet, temp);
        fuse_step(state, forcing, params, config, 1.0, flux);
        
        total_precip += precip;
        total_runoff += flux.q_total;
        total_et += flux.e_total;
    }
    
    // Water balance check
    Real storage_change = (state.S1 + state.S2 + state.SWE) - (100.0 + 500.0 + 0.0);
    Real balance_error = std::abs(total_precip - total_runoff - total_et - storage_change);
    Real balance_error_pct = 100.0 * balance_error / total_precip;
    
    assert(balance_error_pct < 5.0);  // Less than 5% error (due to smoothing)
    
    printf("PASSED\n");
    printf("  Annual precip: %.0f mm\n", total_precip);
    printf("  Annual runoff: %.0f mm (%.0f%%)\n", total_runoff, 100*total_runoff/total_precip);
    printf("  Annual ET: %.0f mm (%.0f%%)\n", total_et, 100*total_et/total_precip);
    printf("  Storage change: %.0f mm\n", storage_change);
    printf("  Balance error: %.1f%%\n", balance_error_pct);
}

/**
 * @brief Test mass conservation
 */
void test_mass_conservation() {
    printf("Testing mass conservation... ");
    
    ModelConfig config = models::VIC;
    
    // Start with known storage
    State state;
    state.S1 = 200.0;
    state.S2 = 800.0;
    state.SWE = 100.0;
    state.sync_derived(config);
    
    Real initial_storage = state.S1 + state.S2 + state.SWE;
    
    Parameters params;
    params.S1_max = 500.0;
    params.S2_max = 2000.0;
    params.f_tens = 0.5;
    params.r1 = 0.7;
    params.ku = 10.0;
    params.c = 2.0;
    params.ks = 40.0;
    params.n = 2.0;
    params.b = 1.5;
    params.Ac_max = 0.8;
    params.T_rain = 1.0;
    params.T_melt = 0.0;
    params.melt_rate = 3.0;
    params.smooth_frac = 0.01;
    params.compute_derived();
    
    // Run for 100 steps
    Real total_in = 0.0;
    Real total_out = 0.0;
    
    Flux flux;
    for (int i = 0; i < 100; ++i) {
        Forcing forcing(5.0, 3.0, 10.0);
        fuse_step(state, forcing, params, config, 1.0, flux);
        
        total_in += forcing.precip;
        total_out += flux.q_total + flux.e_total;
    }
    
    Real final_storage = state.S1 + state.S2 + state.SWE;
    Real storage_change = final_storage - initial_storage;
    
    Real mass_error = std::abs(total_in - total_out - storage_change);
    Real relative_error = mass_error / total_in;
    
    // Allow some numerical error due to smoothing
    assert(relative_error < 0.05);
    
    printf("PASSED\n");
    printf("  Mass error: %.2f mm (%.2f%%)\n", mass_error, 100*relative_error);
}

/**
 * @brief Test smooth functions
 */
void test_smooth_functions() {
    printf("Testing smooth functions... ");
    
    using namespace physics;
    
    // Test logistic overflow
    // Note: The function includes an offset (e_mult=5) to ensure S < S_max
    // So at S=S_max, the function is NOT 0.5, it's still low
    Real w = 0.01 * 100.0;  // 1% of capacity
    
    // Well below capacity: should be ~0
    Real f_below = logistic_overflow(80.0, 100.0, w);
    assert(f_below < 0.01);
    
    // At capacity: due to offset, still mostly 0
    Real f_at = logistic_overflow(100.0, 100.0, w);
    assert(f_at < 0.1);  // Still low due to offset
    
    // Well above capacity: should approach 1
    Real f_above = logistic_overflow(110.0, 100.0, w);
    assert(f_above > 0.5);  // Getting higher
    
    // Test smooth_max
    Real sm = smooth_max(5.0, 3.0);
    assert(std::abs(sm - 5.0) < 0.1);  // Should be close to max(5, 3) = 5
    
    // Test smooth_min
    Real smi = smooth_min(5.0, 3.0);
    assert(std::abs(smi - 3.0) < 0.1);  // Should be close to min(5, 3) = 3
    
    printf("PASSED\n");
}

int main() {
    printf("\n");
    printf("========================================\n");
    printf("dFUSE CPU Tests\n");
    printf("========================================\n");
    dfuse::print_info();
    printf("========================================\n\n");
    
    test_smooth_functions();
    test_vic_single_step();
    test_topmodel_single_step();
    test_sacramento_single_step();
    test_time_series();
    test_mass_conservation();
    
    printf("\n========================================\n");
    printf("All tests PASSED!\n");
    printf("========================================\n\n");
    
    return 0;
}
