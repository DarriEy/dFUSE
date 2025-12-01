/**
 * @file multi_basin_gpu.cu
 * @brief Example: Multi-basin parallel simulation on GPU
 * 
 * This example demonstrates:
 * - Running FUSE across thousands of basins in parallel
 * - Efficient memory management for large-scale simulations
 * - GPU performance optimization techniques
 */

#include <dfuse/dfuse.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>

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
// GPU KERNEL
// ============================================================================

/**
 * @brief Main simulation kernel - each thread handles one basin
 */
__global__ void multi_basin_kernel(
    // States [n_basins]
    Real* S1, Real* S2, Real* SWE,
    // Forcing [n_timesteps, n_basins] - column major
    const Real* precip, const Real* pet, const Real* temp,
    // Parameters [n_basins]
    const Real* S1_max, const Real* S2_max,
    const Real* ku, const Real* c, 
    const Real* ks, const Real* n_exp,
    const Real* b, const Real* Ac_max,
    const Real* T_rain, const Real* melt_rate,
    // Dimensions
    int n_basins, int n_timesteps,
    // Output runoff [n_timesteps, n_basins]
    Real* runoff
) {
    int basin = blockIdx.x * blockDim.x + threadIdx.x;
    if (basin >= n_basins) return;
    
    // Load initial state
    Real s1 = S1[basin];
    Real s2 = S2[basin];
    Real swe = SWE[basin];
    
    // Load parameters
    Real p_S1_max = S1_max[basin];
    Real p_S2_max = S2_max[basin];
    Real p_ku = ku[basin];
    Real p_c = c[basin];
    Real p_ks = ks[basin];
    Real p_n = n_exp[basin];
    Real p_b = b[basin];
    Real p_Ac_max = Ac_max[basin];
    Real p_T_rain = T_rain[basin];
    Real p_melt_rate = melt_rate[basin];
    
    // Time loop
    for (int t = 0; t < n_timesteps; ++t) {
        int idx = t * n_basins + basin;
        
        Real p = precip[idx];
        Real pe = pet[idx];
        Real tm = temp[idx];
        
        // === Snow module ===
        Real snow_frac = Real(1) / (Real(1) + expf(tm - p_T_rain));
        Real snow = p * snow_frac;
        Real rain = p * (Real(1) - snow_frac);
        Real pot_melt = fmaxf(p_melt_rate * (tm - p_T_rain), Real(0));
        Real swe_after = swe + snow;
        Real melt = fminf(pot_melt, swe_after);
        swe = fmaxf(swe_after - melt, Real(0));
        Real throughfall = rain + melt;
        
        // === Surface runoff (VIC b-curve) ===
        Real s1_frac = fminf(s1, p_S1_max) / p_S1_max;
        Real Ac = Real(1) - powf(Real(1) - s1_frac, p_b);
        Ac = fminf(Ac, p_Ac_max);
        Real qsx = Ac * throughfall;
        Real infiltration = throughfall - qsx;
        
        // === Evaporation ===
        Real e1 = pe * s1_frac;
        
        // === Percolation ===
        Real q12 = p_ku * powf(fmaxf(s1_frac, Real(0)), p_c);
        
        // === Baseflow ===
        Real s2_frac = s2 / p_S2_max;
        Real qb = p_ks * powf(fmaxf(s2_frac, Real(0)), p_n);
        
        // === State update ===
        s1 = fmaxf(s1 + (infiltration - e1 - q12), Real(0));
        s2 = fmaxf(s2 + (q12 - qb), Real(0));
        
        // === Output ===
        runoff[idx] = qsx + qb;
    }
    
    // Store final state
    S1[basin] = s1;
    S2[basin] = s2;
    SWE[basin] = swe;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Generate random parameters with spatial variation
 */
void generate_parameters(
    std::vector<Real>& S1_max,
    std::vector<Real>& S2_max,
    std::vector<Real>& ku,
    std::vector<Real>& c,
    std::vector<Real>& ks,
    std::vector<Real>& n_exp,
    std::vector<Real>& b,
    std::vector<Real>& Ac_max,
    std::vector<Real>& T_rain,
    std::vector<Real>& melt_rate,
    int n_basins,
    unsigned int seed = 42
) {
    std::mt19937 gen(seed);
    
    // Uniform distributions for each parameter
    std::uniform_real_distribution<Real> dist_S1(200.0, 800.0);
    std::uniform_real_distribution<Real> dist_S2(800.0, 3000.0);
    std::uniform_real_distribution<Real> dist_ku(5.0, 30.0);
    std::uniform_real_distribution<Real> dist_c(1.5, 4.0);
    std::uniform_real_distribution<Real> dist_ks(20.0, 80.0);
    std::uniform_real_distribution<Real> dist_n(1.5, 3.0);
    std::uniform_real_distribution<Real> dist_b(0.5, 2.5);
    std::uniform_real_distribution<Real> dist_Ac(0.5, 0.95);
    std::uniform_real_distribution<Real> dist_Tr(-1.0, 3.0);
    std::uniform_real_distribution<Real> dist_mr(2.0, 6.0);
    
    S1_max.resize(n_basins);
    S2_max.resize(n_basins);
    ku.resize(n_basins);
    c.resize(n_basins);
    ks.resize(n_basins);
    n_exp.resize(n_basins);
    b.resize(n_basins);
    Ac_max.resize(n_basins);
    T_rain.resize(n_basins);
    melt_rate.resize(n_basins);
    
    for (int i = 0; i < n_basins; ++i) {
        S1_max[i] = dist_S1(gen);
        S2_max[i] = dist_S2(gen);
        ku[i] = dist_ku(gen);
        c[i] = dist_c(gen);
        ks[i] = dist_ks(gen);
        n_exp[i] = dist_n(gen);
        b[i] = dist_b(gen);
        Ac_max[i] = dist_Ac(gen);
        T_rain[i] = dist_Tr(gen);
        melt_rate[i] = dist_mr(gen);
    }
}

/**
 * @brief Generate forcing with spatial variation
 */
void generate_forcing(
    std::vector<Real>& precip,
    std::vector<Real>& pet,
    std::vector<Real>& temp,
    int n_basins,
    int n_timesteps,
    unsigned int seed = 123
) {
    std::mt19937 gen(seed);
    std::normal_distribution<Real> noise(0.0, 1.0);
    
    size_t size = (size_t)n_basins * n_timesteps;
    precip.resize(size);
    pet.resize(size);
    temp.resize(size);
    
    for (int t = 0; t < n_timesteps; ++t) {
        Real phase = 2.0 * M_PI * t / 365.0;
        
        for (int b = 0; b < n_basins; ++b) {
            int idx = t * n_basins + b;
            
            // Base seasonal pattern with spatial variation
            Real basin_factor = 1.0 + 0.3 * std::sin(0.01 * b);
            
            precip[idx] = std::max(Real(0),
                (5.0 + 4.0 * std::sin(phase)) * basin_factor + noise(gen));
            pet[idx] = std::max(Real(0),
                (3.0 + 2.0 * std::sin(phase)) * basin_factor + 0.3 * noise(gen));
            temp[idx] = (15.0 + 10.0 * std::sin(phase)) + 
                        5.0 * std::sin(0.02 * b) + noise(gen);
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("dFUSE Multi-Basin GPU Example\n");
    printf("========================================\n");
    
    // Parse arguments
    int n_basins = 10000;
    int n_timesteps = 365 * 3;  // 3 years
    
    if (argc > 1) n_basins = atoi(argv[1]);
    if (argc > 2) n_timesteps = atoi(argv[2]);
    
    printf("Configuration:\n");
    printf("  Basins: %d\n", n_basins);
    printf("  Timesteps: %d (%d years)\n", n_timesteps, n_timesteps / 365);
    
    // Get GPU info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("  GPU: %s\n", props.name);
    
    // Generate data
    printf("\nGenerating data...\n");
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Parameters
    std::vector<Real> h_S1_max, h_S2_max, h_ku, h_c, h_ks, h_n, h_b, h_Ac, h_Tr, h_mr;
    generate_parameters(h_S1_max, h_S2_max, h_ku, h_c, h_ks, h_n, h_b, h_Ac, h_Tr, h_mr, n_basins);
    
    // Initial states
    std::vector<Real> h_S1(n_basins, 150.0);
    std::vector<Real> h_S2(n_basins, 700.0);
    std::vector<Real> h_SWE(n_basins, 0.0);
    
    // Forcing
    std::vector<Real> h_precip, h_pet, h_temp;
    generate_forcing(h_precip, h_pet, h_temp, n_basins, n_timesteps);
    
    // Output
    size_t output_size = (size_t)n_basins * n_timesteps;
    std::vector<Real> h_runoff(output_size);
    
    auto t_gen = std::chrono::high_resolution_clock::now();
    double gen_time = std::chrono::duration<double>(t_gen - t_start).count();
    printf("  Data generation: %.2f s\n", gen_time);
    
    // Allocate GPU memory
    printf("\nAllocating GPU memory...\n");
    
    Real *d_S1, *d_S2, *d_SWE;
    Real *d_precip, *d_pet, *d_temp;
    Real *d_S1_max, *d_S2_max, *d_ku, *d_c, *d_ks, *d_n, *d_b, *d_Ac, *d_Tr, *d_mr;
    Real *d_runoff;
    
    size_t state_bytes = n_basins * sizeof(Real);
    size_t forcing_bytes = output_size * sizeof(Real);
    
    CUDA_CHECK(cudaMalloc(&d_S1, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_S2, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_SWE, state_bytes));
    
    CUDA_CHECK(cudaMalloc(&d_precip, forcing_bytes));
    CUDA_CHECK(cudaMalloc(&d_pet, forcing_bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, forcing_bytes));
    
    CUDA_CHECK(cudaMalloc(&d_S1_max, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_S2_max, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_ku, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_ks, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_n, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_Ac, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_Tr, state_bytes));
    CUDA_CHECK(cudaMalloc(&d_mr, state_bytes));
    
    CUDA_CHECK(cudaMalloc(&d_runoff, forcing_bytes));
    
    double mem_gb = (3 * state_bytes + 3 * forcing_bytes + 
                     10 * state_bytes + forcing_bytes) / 1e9;
    printf("  GPU memory used: %.2f GB\n", mem_gb);
    
    // Copy to GPU
    printf("\nCopying to GPU...\n");
    auto t_copy_start = std::chrono::high_resolution_clock::now();
    
    CUDA_CHECK(cudaMemcpy(d_S1, h_S1.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S2, h_S2.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SWE, h_SWE.data(), state_bytes, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_precip, h_precip.data(), forcing_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pet, h_pet.data(), forcing_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp, h_temp.data(), forcing_bytes, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_S1_max, h_S1_max.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S2_max, h_S2_max.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ku, h_ku.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ks, h_ks.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n, h_n.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Ac, h_Ac.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Tr, h_Tr.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mr, h_mr.data(), state_bytes, cudaMemcpyHostToDevice));
    
    auto t_copy_end = std::chrono::high_resolution_clock::now();
    double copy_time = std::chrono::duration<double>(t_copy_end - t_copy_start).count();
    printf("  H2D copy time: %.2f s\n", copy_time);
    
    // Launch kernel
    printf("\nRunning simulation...\n");
    
    int block_size = 256;
    int grid_size = (n_basins + block_size - 1) / block_size;
    
    // Warmup
    multi_basin_kernel<<<grid_size, block_size>>>(
        d_S1, d_S2, d_SWE,
        d_precip, d_pet, d_temp,
        d_S1_max, d_S2_max, d_ku, d_c, d_ks, d_n, d_b, d_Ac, d_Tr, d_mr,
        n_basins, n_timesteps, d_runoff
    );
    cudaDeviceSynchronize();
    
    // Reset states
    CUDA_CHECK(cudaMemcpy(d_S1, h_S1.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_S2, h_S2.data(), state_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SWE, h_SWE.data(), state_bytes, cudaMemcpyHostToDevice));
    
    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    multi_basin_kernel<<<grid_size, block_size>>>(
        d_S1, d_S2, d_SWE,
        d_precip, d_pet, d_temp,
        d_S1_max, d_S2_max, d_ku, d_c, d_ks, d_n, d_b, d_Ac, d_Tr, d_mr,
        n_basins, n_timesteps, d_runoff
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_runoff.data(), d_runoff, forcing_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_S1.data(), d_S1, state_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_S2.data(), d_S2, state_bytes, cudaMemcpyDeviceToHost));
    
    // Compute statistics
    printf("\nResults:\n");
    printf("  Kernel time: %.2f ms\n", kernel_ms);
    
    double total_basin_days = (double)n_basins * n_timesteps;
    double throughput = total_basin_days / (kernel_ms / 1000.0);
    printf("  Throughput: %.2e basin-days/sec\n", throughput);
    
    // Summary statistics
    Real mean_runoff = 0, max_runoff = 0;
    for (size_t i = 0; i < output_size; ++i) {
        mean_runoff += h_runoff[i];
        if (h_runoff[i] > max_runoff) max_runoff = h_runoff[i];
    }
    mean_runoff /= output_size;
    
    printf("  Mean runoff: %.2f mm/day\n", mean_runoff);
    printf("  Max runoff: %.2f mm/day\n", max_runoff);
    
    // Sample final states
    printf("\n  Sample final states (first 5 basins):\n");
    for (int i = 0; i < 5 && i < n_basins; ++i) {
        printf("    Basin %d: S1=%.1f, S2=%.1f mm\n", i, h_S1[i], h_S2[i]);
    }
    
    // Cleanup
    cudaFree(d_S1); cudaFree(d_S2); cudaFree(d_SWE);
    cudaFree(d_precip); cudaFree(d_pet); cudaFree(d_temp);
    cudaFree(d_S1_max); cudaFree(d_S2_max);
    cudaFree(d_ku); cudaFree(d_c); cudaFree(d_ks); cudaFree(d_n);
    cudaFree(d_b); cudaFree(d_Ac); cudaFree(d_Tr); cudaFree(d_mr);
    cudaFree(d_runoff);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n========================================\n");
    printf("Example complete!\n");
    printf("========================================\n");
    
    return 0;
}
