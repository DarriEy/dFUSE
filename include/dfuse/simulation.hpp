/**
 * @file simulation.hpp
 * @brief High-level simulation interface with integrated routing
 * 
 * Provides a unified interface for running FUSE simulations with:
 * - Multiple solver backends
 * - Gamma distribution routing
 * - State/flux trajectory storage
 * - Batch simulation support
 */

#ifndef DFUSE_SIMULATION_HPP
#define DFUSE_SIMULATION_HPP

#include "state.hpp"
#include "config.hpp"
#include "solver.hpp"
#include "routing.hpp"
#include "netcdf_io.hpp"

#include <vector>
#include <memory>
#include <functional>
#include <chrono>

namespace dfuse {

// ============================================================================
// SIMULATION CONFIGURATION
// ============================================================================

/**
 * @brief Configuration for simulation runs
 */
struct SimulationConfig {
    // Time stepping
    Real dt = Real(1.0);                        ///< Timestep (days)
    
    // Solver settings
    solver::SolverMethod solver_method = solver::SolverMethod::EXPLICIT_EULER;
    Real solver_rel_tol = Real(0.1);
    Real solver_abs_tol = Real(1e-6);
    
    // Routing settings
    bool enable_routing = true;
    Real routing_shape = Real(2.5);             ///< Gamma shape parameter
    Real routing_mean_delay = Real(1.0);        ///< Mean delay (days)
    
    // Output settings
    bool store_states = true;                   ///< Store state trajectory
    bool store_fluxes = true;                   ///< Store flux components
    int output_interval = 1;                    ///< Output every N timesteps
    
    // Warmup
    int warmup_days = 365;                      ///< Spin-up period (days)
    bool discard_warmup = true;                 ///< Don't store warmup outputs
};

// ============================================================================
// SIMULATION RESULTS
// ============================================================================

/**
 * @brief Results from a simulation run
 */
struct SimulationResults {
    // Time series
    std::vector<double> time;
    
    // Runoff (instantaneous and routed)
    std::vector<Real> runoff_instant;           ///< Unrouted runoff
    std::vector<Real> runoff_routed;            ///< Routed streamflow
    
    // State trajectories (if stored)
    std::vector<Real> S1;
    std::vector<Real> S2;
    std::vector<Real> SWE;
    
    // Flux components (if stored)
    std::vector<Real> qsx;                      ///< Surface runoff
    std::vector<Real> qif;                      ///< Interflow
    std::vector<Real> qb;                       ///< Baseflow
    std::vector<Real> q12;                      ///< Percolation
    std::vector<Real> e_total;                  ///< Total ET
    std::vector<Real> rain;                     ///< Rainfall
    std::vector<Real> melt;                     ///< Snowmelt
    
    // Final state
    State final_state;
    
    // Diagnostics
    int n_timesteps = 0;
    int warmup_steps = 0;
    double runtime_ms = 0;
    
    // Convert to OutputData for I/O
    io::OutputData to_output_data() const {
        io::OutputData data;
        data.time = time;
        data.S1 = S1;
        data.S2 = S2;
        data.SWE = SWE;
        data.q_total = runoff_instant;
        data.qsx = qsx;
        data.qif = qif;
        data.qb = qb;
        data.q12 = q12;
        data.e_total = e_total;
        data.rain = rain;
        data.melt = melt;
        data.streamflow = runoff_routed;
        return data;
    }
};

// ============================================================================
// SIMULATOR CLASS
// ============================================================================

/**
 * @brief Main simulation engine
 * 
 * Handles time stepping, solver dispatch, and routing.
 */
class Simulator {
public:
    Simulator(const ModelConfig& model_config,
              const Parameters& params,
              const SimulationConfig& sim_config = SimulationConfig())
        : model_config_(model_config),
          params_(params),
          sim_config_(sim_config)
    {
        // Initialize solver
        solver::SolverConfig solver_config;
        solver_config.method = sim_config.solver_method;
        solver_config.rel_tol = sim_config.solver_rel_tol;
        solver_config.abs_tol = sim_config.solver_abs_tol;
        solver_ = std::make_unique<solver::Solver>(solver_config);
        
        // Initialize routing buffer
        if (sim_config.enable_routing) {
            routing_buffer_ = std::make_unique<routing::RoutingBuffer>(
                sim_config.routing_shape,
                sim_config.routing_mean_delay,
                sim_config.dt
            );
        }
    }
    
    /**
     * @brief Run simulation with forcing time series
     */
    SimulationResults run(State initial_state, const io::ForcingData& forcing) {
        return run(initial_state, forcing.precip, forcing.pet, forcing.temp, 
                   forcing.time.empty() ? std::vector<double>() : forcing.time);
    }
    
    /**
     * @brief Run simulation with separate forcing vectors
     */
    SimulationResults run(
        State initial_state,
        const std::vector<Real>& precip,
        const std::vector<Real>& pet,
        const std::vector<Real>& temp,
        const std::vector<double>& time = std::vector<double>()
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        size_t n_steps = precip.size();
        if (pet.size() != n_steps || temp.size() != n_steps) {
            throw std::runtime_error("Forcing vectors must have same length");
        }
        
        SimulationResults results;
        results.n_timesteps = static_cast<int>(n_steps);
        results.warmup_steps = std::min(sim_config_.warmup_days, static_cast<int>(n_steps));
        
        // Pre-allocate output arrays
        size_t output_steps = sim_config_.discard_warmup ? 
                              (n_steps - results.warmup_steps) : n_steps;
        output_steps = (output_steps + sim_config_.output_interval - 1) / sim_config_.output_interval;
        
        results.time.reserve(output_steps);
        results.runoff_instant.reserve(output_steps);
        results.runoff_routed.reserve(output_steps);
        
        if (sim_config_.store_states) {
            results.S1.reserve(output_steps);
            results.S2.reserve(output_steps);
            results.SWE.reserve(output_steps);
        }
        if (sim_config_.store_fluxes) {
            results.qsx.reserve(output_steps);
            results.qif.reserve(output_steps);
            results.qb.reserve(output_steps);
            results.q12.reserve(output_steps);
            results.e_total.reserve(output_steps);
            results.rain.reserve(output_steps);
            results.melt.reserve(output_steps);
        }
        
        // Reset routing buffer
        if (routing_buffer_) {
            routing_buffer_->reset();
        }
        
        // Initialize state
        State state = initial_state;
        state.sync_derived(model_config_);
        
        // Main time loop
        for (size_t t = 0; t < n_steps; ++t) {
            // Create forcing
            Forcing forcing{precip[t], pet[t], temp[t]};
            Flux flux;
            
            // Solve one timestep
            solver_->solve(state, forcing, params_, model_config_, sim_config_.dt, flux);
            
            // Route runoff
            Real routed_runoff = flux.q_total;
            if (routing_buffer_) {
                routed_runoff = routing_buffer_->route(flux.q_total);
            }
            
            // Store outputs (skip warmup if configured)
            bool in_warmup = static_cast<int>(t) < results.warmup_steps;
            bool should_store = !in_warmup || !sim_config_.discard_warmup;
            bool at_output_interval = (t % sim_config_.output_interval) == 0;
            
            if (should_store && at_output_interval) {
                double t_val = time.empty() ? static_cast<double>(t) * sim_config_.dt : time[t];
                results.time.push_back(t_val);
                results.runoff_instant.push_back(flux.q_total);
                results.runoff_routed.push_back(routed_runoff);
                
                if (sim_config_.store_states) {
                    results.S1.push_back(state.S1);
                    results.S2.push_back(state.S2);
                    results.SWE.push_back(state.SWE);
                }
                if (sim_config_.store_fluxes) {
                    results.qsx.push_back(flux.qsx);
                    results.qif.push_back(flux.qif);
                    results.qb.push_back(flux.qb);
                    results.q12.push_back(flux.q12);
                    results.e_total.push_back(flux.e_total);
                    results.rain.push_back(flux.rain);
                    results.melt.push_back(flux.melt);
                }
            }
        }
        
        // Store final state
        results.final_state = state;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        results.runtime_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();
        
        return results;
    }
    
    /**
     * @brief Run simulation with callback for each timestep
     */
    void run_with_callback(
        State& state,
        const std::vector<Real>& precip,
        const std::vector<Real>& pet,
        const std::vector<Real>& temp,
        std::function<void(int, const State&, const Flux&, Real)> callback
    ) {
        size_t n_steps = precip.size();
        
        if (routing_buffer_) {
            routing_buffer_->reset();
        }
        
        state.sync_derived(model_config_);
        
        for (size_t t = 0; t < n_steps; ++t) {
            Forcing forcing{precip[t], pet[t], temp[t]};
            Flux flux;
            
            solver_->solve(state, forcing, params_, model_config_, sim_config_.dt, flux);
            
            Real routed = flux.q_total;
            if (routing_buffer_) {
                routed = routing_buffer_->route(flux.q_total);
            }
            
            callback(static_cast<int>(t), state, flux, routed);
        }
    }
    
    // Accessors
    const ModelConfig& model_config() const { return model_config_; }
    const Parameters& params() const { return params_; }
    const SimulationConfig& sim_config() const { return sim_config_; }
    
    void set_params(const Parameters& params) { params_ = params; }
    void set_routing(Real shape, Real mean_delay) {
        routing_buffer_ = std::make_unique<routing::RoutingBuffer>(
            shape, mean_delay, sim_config_.dt);
    }
    
private:
    ModelConfig model_config_;
    Parameters params_;
    SimulationConfig sim_config_;
    std::unique_ptr<solver::Solver> solver_;
    std::unique_ptr<routing::RoutingBuffer> routing_buffer_;
};

// ============================================================================
// BATCH SIMULATION
// ============================================================================

/**
 * @brief Run multiple simulations with different parameters (CPU parallel)
 */
inline std::vector<SimulationResults> batch_simulate(
    const ModelConfig& model_config,
    const std::vector<Parameters>& param_sets,
    const State& initial_state,
    const io::ForcingData& forcing,
    const SimulationConfig& sim_config = SimulationConfig()
) {
    std::vector<SimulationResults> results(param_sets.size());
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < param_sets.size(); ++i) {
        Simulator sim(model_config, param_sets[i], sim_config);
        results[i] = sim.run(initial_state, forcing);
    }
    
    return results;
}

/**
 * @brief Run same simulation for multiple basins (CPU parallel)
 */
inline std::vector<SimulationResults> batch_simulate_basins(
    const ModelConfig& model_config,
    const Parameters& params,
    const std::vector<State>& initial_states,
    const std::vector<io::ForcingData>& forcing_data,
    const SimulationConfig& sim_config = SimulationConfig()
) {
    if (initial_states.size() != forcing_data.size()) {
        throw std::runtime_error("Number of initial states must match forcing data");
    }
    
    std::vector<SimulationResults> results(initial_states.size());
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < initial_states.size(); ++i) {
        Simulator sim(model_config, params, sim_config);
        results[i] = sim.run(initial_states[i], forcing_data[i]);
    }
    
    return results;
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * @brief Quick single simulation
 */
inline SimulationResults simulate(
    const ModelConfig& config,
    const Parameters& params,
    const State& initial_state,
    const std::vector<Real>& precip,
    const std::vector<Real>& pet,
    const std::vector<Real>& temp,
    bool enable_routing = true
) {
    SimulationConfig sim_config;
    sim_config.enable_routing = enable_routing;
    sim_config.routing_shape = params.shape_t;
    sim_config.routing_mean_delay = params.mu_t;
    
    Simulator sim(config, params, sim_config);
    return sim.run(initial_state, precip, pet, temp);
}

/**
 * @brief Run simulation from forcing file
 */
inline SimulationResults simulate_from_file(
    const std::string& forcing_file,
    const ModelConfig& config,
    const Parameters& params,
    const State& initial_state,
    const SimulationConfig& sim_config = SimulationConfig()
) {
    io::ForcingData forcing = io::read_forcing(forcing_file);
    Simulator sim(config, params, sim_config);
    return sim.run(initial_state, forcing);
}

/**
 * @brief Run simulation and write output to file
 */
inline void simulate_to_file(
    const std::string& forcing_file,
    const std::string& output_file,
    const ModelConfig& config,
    const Parameters& params,
    const State& initial_state,
    const SimulationConfig& sim_config = SimulationConfig()
) {
    SimulationResults results = simulate_from_file(
        forcing_file, config, params, initial_state, sim_config);
    io::write_output(output_file, results.to_output_data());
}

} // namespace dfuse

#endif // DFUSE_SIMULATION_HPP
