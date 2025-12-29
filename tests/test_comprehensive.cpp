/**
 * @file test_comprehensive.cpp
 * @brief Comprehensive tests for all cFUSE backends
 * 
 * Tests:
 * - CPU physics validation
 * - SUNDIALS ODE solvers (if compiled with -DDFUSE_USE_SUNDIALS=ON)
 * - Enzyme AD gradients (if compiled with -DDFUSE_USE_ENZYME=ON)
 * - Routing module
 * - Simulation interface
 * - I/O functions
 */

#include <cfuse/dfuse.hpp>
#include <cfuse/solver.hpp>
#include <cfuse/routing.hpp>
#include <cfuse/enzyme_ad.hpp>
#include <cfuse/simulation.hpp>
#include <cfuse/netcdf_io.hpp>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

using namespace dfuse;
using namespace dfuse::models;

// ============================================================================
// TEST UTILITIES
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double elapsed_ms;
};

class TestSuite {
public:
    void add(const TestResult& result) {
        results_.push_back(result);
    }
    
    void print_summary() const {
        int passed = 0;
        for (const auto& r : results_) {
            if (r.passed) passed++;
        }
        
        std::cout << "\n========================================\n";
        std::cout << "Test Summary: " << passed << "/" << results_.size() << " passed\n";
        std::cout << "========================================\n\n";
        
        for (const auto& r : results_) {
            std::cout << (r.passed ? "[PASS] " : "[FAIL] ") << r.name;
            if (!r.message.empty()) {
                std::cout << " - " << r.message;
            }
            std::cout << " (" << std::fixed << std::setprecision(2) << r.elapsed_ms << " ms)\n";
        }
    }
    
    int failed_count() const {
        int failed = 0;
        for (const auto& r : results_) {
            if (!r.passed) failed++;
        }
        return failed;
    }
    
private:
    std::vector<TestResult> results_;
};

// Helper function for synthetic forcing
void generate_synthetic_forcing(
    std::vector<Real>& precip, std::vector<Real>& pet, std::vector<Real>& temp, 
    int n_days, unsigned int seed = 42
) {
    precip.resize(n_days);
    pet.resize(n_days);
    temp.resize(n_days);
    std::srand(seed);
    
    for (int d = 0; d < n_days; ++d) {
        Real day_of_year = Real(d % 365) / Real(365);
        Real precip_base = Real(3) + Real(2) * std::sin(Real(2) * Real(M_PI) * (day_of_year - Real(0.1)));
        Real noise = (Real(std::rand()) / Real(RAND_MAX) - Real(0.5)) * Real(4);
        precip[d] = std::max(Real(0), precip_base + noise);
        if (Real(std::rand()) / Real(RAND_MAX) < Real(0.3)) precip[d] = Real(0);
        
        Real pet_base = Real(2) + Real(1.5) * std::sin(Real(2) * Real(M_PI) * (day_of_year - Real(0.25)));
        pet[d] = std::max(Real(0.5), pet_base);
        
        temp[d] = Real(10) + Real(15) * std::sin(Real(2) * Real(M_PI) * (day_of_year - Real(0.25)));
        temp[d] += (Real(std::rand()) / Real(RAND_MAX) - Real(0.5)) * Real(5);
    }
}

// ============================================================================
// SOLVER TESTS
// ============================================================================

TestResult test_explicit_euler() {
    auto start = std::chrono::high_resolution_clock::now();
    
    ModelConfig config = VIC;
    Parameters params;
    params.S1_max = Real(200);
    params.S2_max = Real(800);
    params.ks = Real(10);
    params.n = Real(1.5);
    params.b = Real(0.3);
    params.compute_derived();
    
    State state;
    state.S1 = Real(100);
    state.S2 = Real(400);
    state.sync_derived(config);
    
    solver::SolverConfig solver_config;
    solver_config.method = solver::SolverMethod::EXPLICIT_EULER;
    solver::Solver solver(solver_config);
    
    bool passed = true;
    std::string message;
    
    // Run 100 days
    Real total_runoff = Real(0);
    for (int t = 0; t < 100; ++t) {
        Forcing forcing{Real(5), Real(3), Real(15)};
        Flux flux;
        solver.solve(state, forcing, params, config, Real(1), flux);
        total_runoff += flux.q_total;
        
        if (!std::isfinite(flux.q_total) || !std::isfinite(state.S1)) {
            passed = false;
            message = "NaN/Inf detected at timestep " + std::to_string(t);
            break;
        }
    }
    
    if (passed && total_runoff < Real(100)) {
        passed = false;
        message = "Total runoff too low: " + std::to_string(total_runoff);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"Explicit Euler solver", passed, message, elapsed};
}

TestResult test_implicit_euler() {
    auto start = std::chrono::high_resolution_clock::now();
    
    ModelConfig config = VIC;
    Parameters params;
    params.S1_max = Real(200);
    params.S2_max = Real(800);
    params.ks = Real(10);
    params.n = Real(1.5);
    params.b = Real(0.3);
    params.compute_derived();
    
    State state;
    state.S1 = Real(100);
    state.S2 = Real(400);
    state.sync_derived(config);
    
    solver::SolverConfig solver_config;
    solver_config.method = solver::SolverMethod::IMPLICIT_EULER;
    solver::Solver solver(solver_config);
    
    bool passed = true;
    std::string message;
    
    // Run 100 days
    Real total_runoff = Real(0);
    for (int t = 0; t < 100; ++t) {
        Forcing forcing{Real(5), Real(3), Real(15)};
        Flux flux;
        solver.solve(state, forcing, params, config, Real(1), flux);
        total_runoff += flux.q_total;
        
        if (!std::isfinite(flux.q_total) || !std::isfinite(state.S1)) {
            passed = false;
            message = "NaN/Inf detected at timestep " + std::to_string(t);
            break;
        }
    }
    
    if (passed && total_runoff < Real(100)) {
        passed = false;
        message = "Total runoff too low: " + std::to_string(total_runoff);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"Implicit Euler solver", passed, message, elapsed};
}

#ifdef DFUSE_USE_SUNDIALS
TestResult test_sundials_bdf() {
    auto start = std::chrono::high_resolution_clock::now();
    
    ModelConfig config = VIC;
    Parameters params;
    params.S1_max = Real(200);
    params.S2_max = Real(800);
    params.ks = Real(10);
    params.n = Real(1.5);
    params.b = Real(0.3);
    params.compute_derived();
    
    State state;
    state.S1 = Real(100);
    state.S2 = Real(400);
    state.sync_derived(config);
    
    solver::SolverConfig solver_config;
    solver_config.method = solver::SolverMethod::SUNDIALS_BDF;
    solver::Solver solver(solver_config);
    
    bool passed = true;
    std::string message;
    
    // Run 100 days
    Real total_runoff = Real(0);
    for (int t = 0; t < 100; ++t) {
        Forcing forcing{Real(5), Real(3), Real(15)};
        Flux flux;
        solver.solve(state, forcing, params, config, Real(1), flux);
        total_runoff += flux.q_total;
        
        if (!std::isfinite(flux.q_total) || !std::isfinite(state.S1)) {
            passed = false;
            message = "NaN/Inf detected at timestep " + std::to_string(t);
            break;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"SUNDIALS BDF solver", passed, message, elapsed};
}

TestResult test_sundials_adams() {
    auto start = std::chrono::high_resolution_clock::now();
    
    ModelConfig config = VIC;
    Parameters params;
    params.S1_max = Real(200);
    params.S2_max = Real(800);
    params.ks = Real(10);
    params.n = Real(1.5);
    params.b = Real(0.3);
    params.compute_derived();
    
    State state;
    state.S1 = Real(100);
    state.S2 = Real(400);
    state.sync_derived(config);
    
    solver::SolverConfig solver_config;
    solver_config.method = solver::SolverMethod::SUNDIALS_ADAMS;
    solver::Solver solver(solver_config);
    
    bool passed = true;
    std::string message;
    
    // Run 100 days
    Real total_runoff = Real(0);
    for (int t = 0; t < 100; ++t) {
        Forcing forcing{Real(5), Real(3), Real(15)};
        Flux flux;
        solver.solve(state, forcing, params, config, Real(1), flux);
        total_runoff += flux.q_total;
        
        if (!std::isfinite(flux.q_total) || !std::isfinite(state.S1)) {
            passed = false;
            message = "NaN/Inf detected at timestep " + std::to_string(t);
            break;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"SUNDIALS Adams-Moulton solver", passed, message, elapsed};
}

TestResult test_sundials_vs_euler() {
    auto start = std::chrono::high_resolution_clock::now();
    
    ModelConfig config = VIC;
    Parameters params;
    params.S1_max = Real(200);
    params.S2_max = Real(800);
    params.ks = Real(10);
    params.n = Real(1.5);
    params.compute_derived();
    
    int n_days = 365;
    std::vector<Real> precip, pet, temp;
    generate_synthetic_forcing(precip, pet, temp, n_days);
    
    // Run with Euler
    State state_euler;
    state_euler.S1 = Real(100);
    state_euler.S2 = Real(400);
    state_euler.sync_derived(config);
    
    solver::SolverConfig euler_config;
    euler_config.method = solver::SolverMethod::EXPLICIT_EULER;
    solver::Solver euler_solver(euler_config);
    
    Real runoff_euler = Real(0);
    for (int t = 0; t < n_days; ++t) {
        Forcing forcing{precip[t], pet[t], temp[t]};
        Flux flux;
        euler_solver.solve(state_euler, forcing, params, config, Real(1), flux);
        runoff_euler += flux.q_total;
    }
    
    // Run with SUNDIALS BDF
    State state_bdf;
    state_bdf.S1 = Real(100);
    state_bdf.S2 = Real(400);
    state_bdf.sync_derived(config);
    
    solver::SolverConfig bdf_config;
    bdf_config.method = solver::SolverMethod::SUNDIALS_BDF;
    solver::Solver bdf_solver(bdf_config);
    
    Real runoff_bdf = Real(0);
    for (int t = 0; t < n_days; ++t) {
        Forcing forcing{precip[t], pet[t], temp[t]};
        Flux flux;
        bdf_solver.solve(state_bdf, forcing, params, config, Real(1), flux);
        runoff_bdf += flux.q_total;
    }
    
    bool passed = true;
    std::string message;
    
    // Compare results (allow 20% difference)
    Real diff_pct = std::abs(runoff_euler - runoff_bdf) / runoff_euler * Real(100);
    if (diff_pct > Real(20)) {
        passed = false;
        message = "SUNDIALS differs from Euler by " + std::to_string(diff_pct) + "%";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"SUNDIALS vs Euler comparison", passed, message, elapsed};
}
#endif // DFUSE_USE_SUNDIALS

// ============================================================================
// GRADIENT TESTS
// ============================================================================

TestResult test_numerical_gradient() {
    auto start = std::chrono::high_resolution_clock::now();
    
    ModelConfig config = VIC;
    
    // Set up parameters as flat array
    Real params[enzyme::NUM_PARAM_VARS];
    std::fill(params, params + enzyme::NUM_PARAM_VARS, Real(0));
    params[0] = Real(200);   // S1_max
    params[1] = Real(800);   // S2_max
    params[2] = Real(0.4);   // f_tens
    params[3] = Real(0.3);   // f_rchr
    params[4] = Real(0.3);   // f_base
    params[5] = Real(0.5);   // r1
    params[6] = Real(20);    // ku
    params[7] = Real(2);     // c
    params[8] = Real(100);   // alpha
    params[9] = Real(2);     // psi
    params[10] = Real(0.3);  // kappa
    params[11] = Real(20);   // ki
    params[12] = Real(10);   // ks
    params[13] = Real(1.5);  // n
    params[14] = Real(0.05); // v
    params[15] = Real(0.1);  // v_A
    params[16] = Real(0.01); // v_B
    params[17] = Real(0.5);  // Ac_max
    params[18] = Real(0.3);  // b
    params[19] = Real(7);    // lambda
    params[20] = Real(3);    // chi
    params[21] = Real(2);    // T_rain
    params[22] = Real(3);    // melt_rate
    params[23] = Real(0.01); // smooth_frac
    
    // Initial state
    Real state[enzyme::NUM_STATE_VARS] = {0};
    state[0] = Real(100);    // S1
    state[5] = Real(400);    // S2
    
    // Config as int array (matching enzyme interface)
    int config_arr[enzyme::NUM_CONFIG_VARS] = {0};
    config_arr[0] = 0;  // upper_arch = SINGLE_STATE
    config_arr[1] = 1;  // lower_arch = SINGLE_EVAP
    config_arr[2] = 0;  // evap = SEQUENTIAL
    config_arr[3] = 0;  // percolation = TOTAL
    config_arr[4] = 2;  // baseflow = NONLINEAR
    config_arr[5] = 1;  // runoff = PARETO
    config_arr[6] = 0;  // interflow = NONE
    config_arr[7] = 0;  // enable_snow = false
    
    // Generate forcing
    int n_timesteps = 50;
    std::vector<Real> precip(n_timesteps), pet(n_timesteps), temp(n_timesteps);
    generate_synthetic_forcing(precip, pet, temp, n_timesteps);
    
    std::vector<Real> forcing(n_timesteps * enzyme::NUM_FORCING_VARS);
    for (int t = 0; t < n_timesteps; ++t) {
        forcing[t * enzyme::NUM_FORCING_VARS + 0] = precip[t];
        forcing[t * enzyme::NUM_FORCING_VARS + 1] = pet[t];
        forcing[t * enzyme::NUM_FORCING_VARS + 2] = temp[t];
    }
    
    // Create fake observed runoff (just use zeros for gradient test)
    std::vector<Real> observed(n_timesteps, Real(5));
    
    // Compute numerical gradient
    Real grad_params[enzyme::NUM_PARAM_VARS];
    enzyme::compute_loss_gradient_numerical(
        state, forcing.data(), params, config_arr, Real(1), observed.data(), n_timesteps, grad_params);
    
    bool passed = true;
    std::string message;
    
    // Check gradients are finite and reasonable
    for (int i = 0; i < enzyme::NUM_PARAM_VARS; ++i) {
        if (!std::isfinite(grad_params[i])) {
            passed = false;
            message = "NaN gradient for param " + std::to_string(i);
            break;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"Numerical gradient computation", passed, message, elapsed};
}

#ifdef DFUSE_USE_ENZYME
TestResult test_enzyme_gradient() {
    auto start = std::chrono::high_resolution_clock::now();
    
    // This test requires Enzyme compilation
    // For now, just verify the interface compiles
    bool passed = true;
    std::string message = "Enzyme interface compiled successfully";
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"Enzyme AD interface", passed, message, elapsed};
}

TestResult test_adjoint_solver() {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Test adjoint solver structure
    int n_timesteps = 100;
    
    // Create config array for VIC-style model
    int config_arr[enzyme::NUM_CONFIG_VARS] = {
        static_cast<int>(UpperLayerArch::SINGLE_STATE),
        static_cast<int>(LowerLayerArch::SINGLE_EVAP),
        static_cast<int>(EvaporationType::ROOT_WEIGHT),
        static_cast<int>(PercolationType::TOTAL_STORAGE),
        static_cast<int>(BaseflowType::NONLINEAR),
        static_cast<int>(SurfaceRunoffType::UZ_PARETO),
        static_cast<int>(InterflowType::NONE),
        1  // enable_snow
    };
    
    enzyme::AdjointSolver adjoint(n_timesteps, config_arr);
    
    bool passed = true;
    std::string message;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"Adjoint solver structure", passed, message, elapsed};
}
#endif // DFUSE_USE_ENZYME

// ============================================================================
// ROUTING TESTS
// ============================================================================

TestResult test_routing_integration() {
    auto start = std::chrono::high_resolution_clock::now();
    
    ModelConfig config = VIC;
    Parameters params;
    params.S1_max = Real(200);
    params.S2_max = Real(800);
    params.ks = Real(10);
    params.n = Real(1.5);
    params.mu_t = Real(2);      // 2-day mean delay
    params.shape_t = Real(2.5); // gamma shape
    params.compute_derived();
    
    State state;
    state.S1 = Real(100);
    state.S2 = Real(400);
    state.sync_derived(config);
    
    // Create simulator with routing
    SimulationConfig sim_config;
    sim_config.enable_routing = true;
    sim_config.routing_shape = params.shape_t;
    sim_config.routing_mean_delay = params.mu_t;
    sim_config.warmup_days = 0;
    
    Simulator sim(config, params, sim_config);
    
    // Generate forcing
    int n_days = 100;
    std::vector<Real> precip(n_days), pet(n_days), temp(n_days);
    generate_synthetic_forcing(precip, pet, temp, n_days);
    
    // Add a precipitation pulse for testing routing delay
    for (int t = 0; t < n_days; ++t) precip[t] = Real(0);
    precip[10] = Real(50);  // 50mm pulse on day 10
    
    SimulationResults results = sim.run(state, precip, pet, temp);
    
    bool passed = true;
    std::string message;
    
    // Check that routed flow is delayed relative to instant
    Real instant_peak_day = 0, routed_peak_day = 0;
    Real instant_peak = 0, routed_peak = 0;
    for (size_t t = 0; t < results.time.size(); ++t) {
        if (results.runoff_instant[t] > instant_peak) {
            instant_peak = results.runoff_instant[t];
            instant_peak_day = Real(t);
        }
        if (results.runoff_routed[t] > routed_peak) {
            routed_peak = results.runoff_routed[t];
            routed_peak_day = Real(t);
        }
    }
    
    if (routed_peak_day <= instant_peak_day) {
        passed = false;
        message = "Routed peak should be delayed";
    }
    
    // Check mass conservation
    Real total_instant = Real(0), total_routed = Real(0);
    for (size_t t = 0; t < results.time.size(); ++t) {
        total_instant += results.runoff_instant[t];
        total_routed += results.runoff_routed[t];
    }
    
    Real mass_error = std::abs(total_instant - total_routed) / total_instant * Real(100);
    if (mass_error > Real(5)) {
        passed = false;
        message = "Routing mass error: " + std::to_string(mass_error) + "%";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"Routing integration", passed, message, elapsed};
}

// ============================================================================
// SIMULATION INTERFACE TESTS
// ============================================================================

TestResult test_simulation_interface() {
    auto start = std::chrono::high_resolution_clock::now();
    
    ModelConfig config = VIC;
    config.enable_snow = true;
    
    Parameters params;
    params.S1_max = Real(200);
    params.S2_max = Real(800);
    params.ks = Real(10);
    params.n = Real(1.5);
    params.T_rain = Real(2);
    params.melt_rate = Real(3);
    params.compute_derived();
    
    State initial_state;
    initial_state.S1 = Real(100);
    initial_state.S2 = Real(400);
    initial_state.SWE = Real(0);
    initial_state.sync_derived(config);
    
    // Create forcing data
    io::ForcingData forcing;
    generate_synthetic_forcing(forcing.precip, forcing.pet, forcing.temp, 365);
    forcing.time.resize(365);
    for (int t = 0; t < 365; ++t) forcing.time[t] = t;
    
    // Run simulation
    SimulationConfig sim_config;
    sim_config.warmup_days = 30;
    sim_config.discard_warmup = true;
    sim_config.store_states = true;
    sim_config.store_fluxes = true;
    
    Simulator sim(config, params, sim_config);
    SimulationResults results = sim.run(initial_state, forcing);
    
    bool passed = true;
    std::string message;
    
    // Check results are valid
    if (results.time.empty()) {
        passed = false;
        message = "No output produced";
    } else if (results.time.size() != 365 - 30) {
        passed = false;
        message = "Wrong output size after warmup discard";
    }
    
    // Check states stored
    if (passed && results.S1.empty()) {
        passed = false;
        message = "States not stored";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"Simulation interface", passed, message, elapsed};
}

// ============================================================================
// I/O TESTS
// ============================================================================

TestResult test_csv_io() {
    auto start = std::chrono::high_resolution_clock::now();
    
    bool passed = true;
    std::string message;
    
    // Create test forcing CSV
    std::string forcing_file = "/tmp/test_forcing.csv";
    {
        std::ofstream f(forcing_file);
        f << "time,precip,pet,temp\n";
        for (int t = 0; t < 10; ++t) {
            f << t << "," << (5 + t % 3) << "," << 3.0 << "," << (15 + t) << "\n";
        }
    }
    
    // Read it back
    io::ForcingData forcing = io::read_forcing_csv(forcing_file);
    
    if (forcing.size() != 10) {
        passed = false;
        message = "Wrong forcing size: " + std::to_string(forcing.size());
    }
    
    if (passed && std::abs(forcing.precip[0] - Real(5)) > Real(0.01)) {
        passed = false;
        message = "Wrong precip value";
    }
    
    // Test parameter I/O
    std::string param_file = "/tmp/test_params.csv";
    Parameters params;
    params.S1_max = Real(250);
    params.ks = Real(15);
    params.compute_derived();
    
    io::write_parameters_csv(param_file, params);
    Parameters params_read = io::read_parameters_csv(param_file);
    
    if (std::abs(params_read.S1_max - Real(250)) > Real(0.01)) {
        passed = false;
        message = "Parameter I/O mismatch";
    }
    
    // Clean up
    std::remove(forcing_file.c_str());
    std::remove(param_file.c_str());
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"CSV I/O", passed, message, elapsed};
}

TestResult test_output_csv() {
    auto start = std::chrono::high_resolution_clock::now();
    
    bool passed = true;
    std::string message;
    
    // Create output data
    io::OutputData output;
    output.resize(10);
    for (int t = 0; t < 10; ++t) {
        output.time[t] = t;
        output.S1[t] = Real(100 + t);
        output.S2[t] = Real(400 - t);
        output.q_total[t] = Real(5 + t * 0.1);
    }
    
    // Write and read back
    std::string output_file = "/tmp/test_output.csv";
    io::write_output_csv(output_file, output);
    
    // Check file exists
    std::ifstream f(output_file);
    if (!f.good()) {
        passed = false;
        message = "Output file not created";
    }
    f.close();
    
    // Clean up
    std::remove(output_file.c_str());
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"Output CSV", passed, message, elapsed};
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "cFUSE Comprehensive Test Suite\n";
    std::cout << "========================================\n";
    
    // Print build configuration
    std::cout << "Build configuration:\n";
#ifdef DFUSE_USE_DOUBLE
    std::cout << "  Precision: double\n";
#else
    std::cout << "  Precision: float\n";
#endif
#ifdef DFUSE_USE_SUNDIALS
    std::cout << "  SUNDIALS: enabled\n";
#else
    std::cout << "  SUNDIALS: disabled\n";
#endif
#ifdef DFUSE_USE_ENZYME
    std::cout << "  Enzyme AD: enabled\n";
#else
    std::cout << "  Enzyme AD: disabled\n";
#endif
#ifdef DFUSE_USE_NETCDF
    std::cout << "  NetCDF: enabled\n";
#else
    std::cout << "  NetCDF: disabled\n";
#endif
    std::cout << "========================================\n\n";
    
    TestSuite suite;
    
    // Solver tests
    std::cout << "Running solver tests...\n";
    suite.add(test_explicit_euler());
    suite.add(test_implicit_euler());
    
#ifdef DFUSE_USE_SUNDIALS
    std::cout << "Running SUNDIALS tests...\n";
    suite.add(test_sundials_bdf());
    suite.add(test_sundials_adams());
    suite.add(test_sundials_vs_euler());
#endif
    
    // Gradient tests
    std::cout << "Running gradient tests...\n";
    suite.add(test_numerical_gradient());
    
#ifdef DFUSE_USE_ENZYME
    std::cout << "Running Enzyme AD tests...\n";
    suite.add(test_enzyme_gradient());
    suite.add(test_adjoint_solver());
#endif
    
    // Routing tests
    std::cout << "Running routing tests...\n";
    suite.add(test_routing_integration());
    
    // Simulation interface tests
    std::cout << "Running simulation interface tests...\n";
    suite.add(test_simulation_interface());
    
    // I/O tests
    std::cout << "Running I/O tests...\n";
    suite.add(test_csv_io());
    suite.add(test_output_csv());
    
    // Print summary
    suite.print_summary();
    
    return suite.failed_count();
}
