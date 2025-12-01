/**
 * @file test_cuda.cu
 * @brief CUDA tests for dFUSE GPU kernels
 */

#include <dfuse/dfuse.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace dfuse;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Allocate and initialize StateBatch on GPU
 */
StateBatch allocate_state_batch_gpu(int n) {
    StateBatch batch;
    batch.n = n;
    
    CUDA_CHECK(cudaMalloc(&batch.S1, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S1_T, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S1_TA, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S1_TB, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S1_F, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S2, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S2_T, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S2_FA, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S2_FB, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.SWE, n * sizeof(Real)));
    
    return batch;
}

void free_state_batch_gpu(StateBatch& batch) {
    cudaFree(batch.S1);
    cudaFree(batch.S1_T);
    cudaFree(batch.S1_TA);
    cudaFree(batch.S1_TB);
    cudaFree(batch.S1_F);
    cudaFree(batch.S2);
    cudaFree(batch.S2_T);
    cudaFree(batch.S2_FA);
    cudaFree(batch.S2_FB);
    cudaFree(batch.SWE);
}

/**
 * @brief Allocate ForcingBatch on GPU
 */
ForcingBatch allocate_forcing_batch_gpu(int n, int nt) {
    ForcingBatch batch;
    batch.n = n;
    batch.nt = nt;
    
    size_t size = n * nt * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&batch.precip, size));
    CUDA_CHECK(cudaMalloc(&batch.pet, size));
    CUDA_CHECK(cudaMalloc(&batch.temp, size));
    
    return batch;
}

void free_forcing_batch_gpu(ForcingBatch& batch) {
    cudaFree(batch.precip);
    cudaFree(batch.pet);
    cudaFree(batch.temp);
}

/**
 * @brief Allocate ParameterBatch on GPU
 */
ParameterBatch allocate_param_batch_gpu(int n) {
    ParameterBatch batch;
    batch.n = n;
    
    CUDA_CHECK(cudaMalloc(&batch.S1_max, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.S2_max, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.f_tens, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.f_rchr, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.f_base, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.r1, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.ku, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.c, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.alpha, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.psi, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.kappa, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.ki, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.ks, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.n_exp, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.v, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.v_A, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.v_B, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.Ac_max, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.b, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.lambda, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.chi, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.mu_t, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.T_rain, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.melt_rate, n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&batch.smooth_frac, n * sizeof(Real)));
    
    return batch;
}

void free_param_batch_gpu(ParameterBatch& batch) {
    cudaFree(batch.S1_max);
    cudaFree(batch.S2_max);
    cudaFree(batch.f_tens);
    cudaFree(batch.f_rchr);
    cudaFree(batch.f_base);
    cudaFree(batch.r1);
    cudaFree(batch.ku);
    cudaFree(batch.c);
    cudaFree(batch.alpha);
    cudaFree(batch.psi);
    cudaFree(batch.kappa);
    cudaFree(batch.ki);
    cudaFree(batch.ks);
    cudaFree(batch.n_exp);
    cudaFree(batch.v);
    cudaFree(batch.v_A);
    cudaFree(batch.v_B);
    cudaFree(batch.Ac_max);
    cudaFree(batch.b);
    cudaFree(batch.lambda);
    cudaFree(batch.chi);
    cudaFree(batch.mu_t);
    cudaFree(batch.T_rain);
    cudaFree(batch.melt_rate);
    cudaFree(batch.smooth_frac);
}

// ============================================================================
// SIMPLE GPU KERNEL TEST
// ============================================================================

/**
 * @brief Simple kernel to test single-step execution on GPU
 */
__global__ void test_fuse_step_kernel(
    State* states,
    Forcing* forcing,
    Parameters* params,
    ModelConfig config,
    Real dt,
    Flux* fluxes,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    fuse_step(states[idx], forcing[idx], params[idx], config, dt, fluxes[idx]);
}

/**
 * @brief Test single step on GPU
 */
void test_gpu_single_step() {
    printf("Testing GPU single step... ");
    
    const int n = 1024;  // Test with 1024 HRUs
    
    // Allocate host data
    std::vector<State> h_states(n);
    std::vector<Forcing> h_forcing(n);
    std::vector<Parameters> h_params(n);
    std::vector<Flux> h_fluxes(n);
    
    ModelConfig config = models::VIC;
    
    // Initialize host data
    for (int i = 0; i < n; ++i) {
        h_states[i].S1 = 100.0 + 100.0 * (i % 10);
        h_states[i].S2 = 500.0 + 200.0 * (i % 5);
        h_states[i].SWE = 0.0;
        h_states[i].sync_derived(config);
        
        h_forcing[i] = Forcing(5.0 + 5.0 * (i % 3), 3.0, 15.0);
        
        h_params[i].S1_max = 500.0;
        h_params[i].S2_max = 2000.0;
        h_params[i].f_tens = 0.5;
        h_params[i].r1 = 0.7;
        h_params[i].ku = 10.0;
        h_params[i].c = 2.0;
        h_params[i].ks = 40.0;
        h_params[i].n = 2.0;
        h_params[i].b = 1.5;
        h_params[i].Ac_max = 0.8;
        h_params[i].T_rain = 1.0;
        h_params[i].T_melt = 0.0;
        h_params[i].melt_rate = 3.0;
        h_params[i].smooth_frac = 0.01;
        h_params[i].compute_derived();
    }
    
    // Allocate device data
    State* d_states;
    Forcing* d_forcing;
    Parameters* d_params;
    Flux* d_fluxes;
    
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_forcing, n * sizeof(Forcing)));
    CUDA_CHECK(cudaMalloc(&d_params, n * sizeof(Parameters)));
    CUDA_CHECK(cudaMalloc(&d_fluxes, n * sizeof(Flux)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_states, h_states.data(), n * sizeof(State), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_forcing, h_forcing.data(), n * sizeof(Forcing), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_params, h_params.data(), n * sizeof(Parameters), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    test_fuse_step_kernel<<<grid_size, block_size>>>(
        d_states, d_forcing, d_params, config, 1.0, d_fluxes, n
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_states.data(), d_states, n * sizeof(State), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fluxes.data(), d_fluxes, n * sizeof(Flux), cudaMemcpyDeviceToHost));
    
    // Verify results
    bool all_valid = true;
    for (int i = 0; i < n; ++i) {
        if (h_fluxes[i].q_total < 0 || h_fluxes[i].q_total > 100 ||
            std::isnan(h_fluxes[i].q_total)) {
            all_valid = false;
            printf("\nInvalid result at index %d: q_total = %f\n", i, h_fluxes[i].q_total);
            break;
        }
    }
    
    // Cleanup
    cudaFree(d_states);
    cudaFree(d_forcing);
    cudaFree(d_params);
    cudaFree(d_fluxes);
    
    if (all_valid) {
        printf("PASSED\n");
        printf("  Processed %d HRUs\n", n);
        printf("  Sample runoff[0]: %.2f mm/day\n", h_fluxes[0].q_total);
        printf("  Sample runoff[%d]: %.2f mm/day\n", n-1, h_fluxes[n-1].q_total);
    } else {
        printf("FAILED\n");
        exit(1);
    }
}

// ============================================================================
// TIME SERIES ON GPU
// ============================================================================

/**
 * @brief Kernel for multi-timestep simulation
 */
__global__ void fuse_timeseries_kernel(
    State* states,
    const Real* precip,    // [nt, n] column major
    const Real* pet,
    const Real* temp,
    Parameters* params,
    ModelConfig config,
    int n,
    int nt,
    Real dt,
    Real* runoff          // [nt, n] output
) {
    int hru = blockIdx.x * blockDim.x + threadIdx.x;
    if (hru >= n) return;
    
    State state = states[hru];
    Parameters param = params[hru];
    Flux flux;
    
    for (int t = 0; t < nt; ++t) {
        int idx = t * n + hru;
        Forcing forcing(precip[idx], pet[idx], temp[idx]);
        
        fuse_step(state, forcing, param, config, dt, flux);
        
        runoff[idx] = flux.q_total;
    }
    
    states[hru] = state;
}

/**
 * @brief Test time series simulation on GPU
 */
void test_gpu_timeseries() {
    printf("Testing GPU time series... ");
    
    const int n_hru = 1000;
    const int n_timesteps = 365;
    
    ModelConfig config = models::VIC;
    
    // Allocate host data
    std::vector<State> h_states(n_hru);
    std::vector<Parameters> h_params(n_hru);
    std::vector<Real> h_precip(n_hru * n_timesteps);
    std::vector<Real> h_pet(n_hru * n_timesteps);
    std::vector<Real> h_temp(n_hru * n_timesteps);
    std::vector<Real> h_runoff(n_hru * n_timesteps);
    
    // Initialize
    for (int i = 0; i < n_hru; ++i) {
        h_states[i].S1 = 150.0;
        h_states[i].S2 = 700.0;
        h_states[i].SWE = 0.0;
        h_states[i].sync_derived(config);
        
        h_params[i].S1_max = 400.0 + 100.0 * (i % 5);
        h_params[i].S2_max = 1500.0 + 500.0 * (i % 3);
        h_params[i].f_tens = 0.5;
        h_params[i].r1 = 0.7;
        h_params[i].ku = 10.0;
        h_params[i].c = 2.0;
        h_params[i].ks = 40.0;
        h_params[i].n = 2.0;
        h_params[i].b = 1.5;
        h_params[i].Ac_max = 0.8;
        h_params[i].T_rain = 1.0;
        h_params[i].T_melt = 0.0;
        h_params[i].melt_rate = 3.0;
        h_params[i].smooth_frac = 0.01;
        h_params[i].compute_derived();
    }
    
    // Generate forcing (seasonal pattern)
    for (int t = 0; t < n_timesteps; ++t) {
        Real phase = 2.0 * M_PI * t / 365.0;
        for (int i = 0; i < n_hru; ++i) {
            int idx = t * n_hru + i;
            h_precip[idx] = 5.0 + 4.0 * std::sin(phase) + 0.5 * (i % 5);
            h_pet[idx] = 3.0 + 2.0 * std::sin(phase);
            h_temp[idx] = 15.0 + 10.0 * std::sin(phase);
        }
    }
    
    // Allocate device data
    State* d_states;
    Parameters* d_params;
    Real* d_precip;
    Real* d_pet;
    Real* d_temp;
    Real* d_runoff;
    
    size_t forcing_size = n_hru * n_timesteps * sizeof(Real);
    
    CUDA_CHECK(cudaMalloc(&d_states, n_hru * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_params, n_hru * sizeof(Parameters)));
    CUDA_CHECK(cudaMalloc(&d_precip, forcing_size));
    CUDA_CHECK(cudaMalloc(&d_pet, forcing_size));
    CUDA_CHECK(cudaMalloc(&d_temp, forcing_size));
    CUDA_CHECK(cudaMalloc(&d_runoff, forcing_size));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_states, h_states.data(), n_hru * sizeof(State), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_params, h_params.data(), n_hru * sizeof(Parameters), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_precip, h_precip.data(), forcing_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pet, h_pet.data(), forcing_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp, h_temp.data(), forcing_size, cudaMemcpyHostToDevice));
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (n_hru + block_size - 1) / block_size;
    
    fuse_timeseries_kernel<<<grid_size, block_size>>>(
        d_states, d_precip, d_pet, d_temp, d_params,
        config, n_hru, n_timesteps, 1.0, d_runoff
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_runoff.data(), d_runoff, forcing_size, cudaMemcpyDeviceToHost));
    
    // Verify and compute statistics
    Real total_runoff = 0;
    bool all_valid = true;
    for (int i = 0; i < n_hru * n_timesteps; ++i) {
        if (std::isnan(h_runoff[i]) || h_runoff[i] < 0) {
            all_valid = false;
            break;
        }
        total_runoff += h_runoff[i];
    }
    
    Real mean_runoff = total_runoff / (n_hru * n_timesteps);
    
    // Performance metrics
    double total_hru_days = (double)n_hru * n_timesteps;
    double throughput = total_hru_days / (milliseconds / 1000.0);
    
    // Cleanup
    cudaFree(d_states);
    cudaFree(d_params);
    cudaFree(d_precip);
    cudaFree(d_pet);
    cudaFree(d_temp);
    cudaFree(d_runoff);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (all_valid) {
        printf("PASSED\n");
        printf("  HRUs: %d, Timesteps: %d\n", n_hru, n_timesteps);
        printf("  Time: %.2f ms\n", milliseconds);
        printf("  Throughput: %.2e HRU-days/sec\n", throughput);
        printf("  Mean runoff: %.2f mm/day\n", mean_runoff);
    } else {
        printf("FAILED - invalid results\n");
        exit(1);
    }
}

// ============================================================================
// BENCHMARK
// ============================================================================

void benchmark_gpu() {
    printf("\nRunning GPU benchmark...\n");
    
    ModelConfig config = models::VIC;
    
    // Test different sizes
    std::vector<int> hru_counts = {100, 1000, 10000, 100000};
    int n_timesteps = 365;
    
    for (int n_hru : hru_counts) {
        // Allocate
        std::vector<State> h_states(n_hru);
        std::vector<Parameters> h_params(n_hru);
        
        for (int i = 0; i < n_hru; ++i) {
            h_states[i].S1 = 150.0;
            h_states[i].S2 = 700.0;
            h_states[i].SWE = 0.0;
            h_states[i].sync_derived(config);
            
            h_params[i].S1_max = 400.0;
            h_params[i].S2_max = 1500.0;
            h_params[i].f_tens = 0.5;
            h_params[i].r1 = 0.7;
            h_params[i].ku = 10.0;
            h_params[i].c = 2.0;
            h_params[i].ks = 40.0;
            h_params[i].n = 2.0;
            h_params[i].b = 1.5;
            h_params[i].Ac_max = 0.8;
            h_params[i].smooth_frac = 0.01;
            h_params[i].compute_derived();
        }
        
        size_t forcing_size = n_hru * n_timesteps * sizeof(Real);
        std::vector<Real> h_forcing(n_hru * n_timesteps, 5.0);
        
        // Device allocation
        State* d_states;
        Parameters* d_params;
        Real* d_precip, *d_pet, *d_temp, *d_runoff;
        
        CUDA_CHECK(cudaMalloc(&d_states, n_hru * sizeof(State)));
        CUDA_CHECK(cudaMalloc(&d_params, n_hru * sizeof(Parameters)));
        CUDA_CHECK(cudaMalloc(&d_precip, forcing_size));
        CUDA_CHECK(cudaMalloc(&d_pet, forcing_size));
        CUDA_CHECK(cudaMalloc(&d_temp, forcing_size));
        CUDA_CHECK(cudaMalloc(&d_runoff, forcing_size));
        
        CUDA_CHECK(cudaMemcpy(d_states, h_states.data(), n_hru * sizeof(State), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_params, h_params.data(), n_hru * sizeof(Parameters), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_precip, h_forcing.data(), forcing_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pet, h_forcing.data(), forcing_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_temp, h_forcing.data(), forcing_size, cudaMemcpyHostToDevice));
        
        // Warmup
        int block_size = 256;
        int grid_size = (n_hru + block_size - 1) / block_size;
        fuse_timeseries_kernel<<<grid_size, block_size>>>(
            d_states, d_precip, d_pet, d_temp, d_params,
            config, n_hru, n_timesteps, 1.0, d_runoff
        );
        cudaDeviceSynchronize();
        
        // Reset states
        CUDA_CHECK(cudaMemcpy(d_states, h_states.data(), n_hru * sizeof(State), cudaMemcpyHostToDevice));
        
        // Timed run
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        fuse_timeseries_kernel<<<grid_size, block_size>>>(
            d_states, d_precip, d_pet, d_temp, d_params,
            config, n_hru, n_timesteps, 1.0, d_runoff
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        double total_hru_days = (double)n_hru * n_timesteps;
        double throughput = total_hru_days / (ms / 1000.0);
        
        printf("  %6d HRUs x %3d days: %8.2f ms -> %.2e HRU-days/sec\n",
               n_hru, n_timesteps, ms, throughput);
        
        cudaFree(d_states);
        cudaFree(d_params);
        cudaFree(d_precip);
        cudaFree(d_pet);
        cudaFree(d_temp);
        cudaFree(d_runoff);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("\n");
    printf("========================================\n");
    printf("dFUSE CUDA Tests\n");
    printf("========================================\n");
    
    // Print device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("Device: %s\n", props.name);
    printf("Compute: %d.%d\n", props.major, props.minor);
    printf("Memory: %.0f MB\n", props.totalGlobalMem / 1e6);
    printf("========================================\n\n");
    
    test_gpu_single_step();
    test_gpu_timeseries();
    benchmark_gpu();
    
    printf("\n========================================\n");
    printf("All CUDA tests PASSED!\n");
    printf("========================================\n\n");
    
    return 0;
}
