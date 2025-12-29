/**
 * @file routing.hpp
 * @brief Runoff routing module for cFUSE
 * 
 * Implements the gamma distribution unit hydrograph routing from
 * Clark et al. (2008) equation (13):
 * 
 *   P(a,x) = gamma(a,x) / Gamma(a)
 *   
 * where x = t * a / mu_t, and mu_t is the mean time delay.
 * 
 * This creates a time-delayed response that smooths the instantaneous
 * runoff generation from the bucket model.
 */

#pragma once

#include "config.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace dfuse {
namespace routing {

// ============================================================================
// GAMMA DISTRIBUTION FUNCTIONS
// ============================================================================

/**
 * @brief Compute the incomplete gamma function using series expansion
 * 
 * gamma(a, x) = integral from 0 to x of t^(a-1) * exp(-t) dt
 */
DFUSE_HOST_DEVICE inline Real incomplete_gamma(Real a, Real x, int max_iter = 100) {
    if (x < Real(0) || a <= Real(0)) return Real(0);
    if (x == Real(0)) return Real(0);
    
    // Use series expansion for small x
    if (x < a + Real(1)) {
        Real sum = Real(1) / a;
        Real term = Real(1) / a;
        for (int n = 1; n < max_iter; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < Real(1e-10) * std::abs(sum)) break;
        }
        return std::exp(-x + a * std::log(x)) * sum;
    }
    
    // Use continued fraction for large x
    Real b = x + Real(1) - a;
    Real c = Real(1) / Real(1e-30);
    Real d = Real(1) / b;
    Real h = d;
    
    for (int i = 1; i < max_iter; ++i) {
        Real an = -Real(i) * (Real(i) - a);
        b += Real(2);
        d = an * d + b;
        if (std::abs(d) < Real(1e-30)) d = Real(1e-30);
        c = b + an / c;
        if (std::abs(c) < Real(1e-30)) c = Real(1e-30);
        d = Real(1) / d;
        Real delta = d * c;
        h *= delta;
        if (std::abs(delta - Real(1)) < Real(1e-10)) break;
    }
    
    // gamma(a) using Stirling for large a
    Real log_gamma_a;
    if (a > Real(10)) {
        log_gamma_a = Real(0.5) * std::log(Real(2) * Real(M_PI) / a) + 
                      a * (std::log(a + Real(1) / (Real(12) * a - Real(1) / (Real(10) * a))) - Real(1));
    } else {
        // Direct computation for small a
        log_gamma_a = std::lgamma(a);
    }
    
    return std::exp(log_gamma_a) - std::exp(-x + a * std::log(x) - log_gamma_a) * h;
}

/**
 * @brief Compute the regularized incomplete gamma function P(a, x)
 * 
 * P(a, x) = gamma(a, x) / Gamma(a)
 * This is the CDF of the gamma distribution.
 */
DFUSE_HOST_DEVICE inline Real gamma_cdf(Real a, Real x) {
    if (x <= Real(0)) return Real(0);
    if (a <= Real(0)) return Real(0);
    
    Real log_gamma_a = std::lgamma(a);
    
    // Series expansion for x < a+1
    if (x < a + Real(1)) {
        Real sum = Real(1) / a;
        Real term = Real(1) / a;
        for (int n = 1; n < 100; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::abs(term) < Real(1e-12) * std::abs(sum)) break;
        }
        return std::exp(-x + a * std::log(x) - log_gamma_a) * sum;
    }
    
    // Continued fraction for x >= a+1
    Real b = x + Real(1) - a;
    Real c = Real(1) / Real(1e-30);
    Real d = Real(1) / b;
    Real h = d;
    
    for (int i = 1; i < 100; ++i) {
        Real an = -Real(i) * (Real(i) - a);
        b += Real(2);
        d = an * d + b;
        if (std::abs(d) < Real(1e-30)) d = Real(1e-30);
        c = b + an / c;
        if (std::abs(c) < Real(1e-30)) c = Real(1e-30);
        d = Real(1) / d;
        Real delta = d * c;
        h *= delta;
        if (std::abs(delta - Real(1)) < Real(1e-12)) break;
    }
    
    return Real(1) - std::exp(-x + a * std::log(x) - log_gamma_a) * h;
}

/**
 * @brief Compute gamma distribution PDF
 * 
 * f(x; a, mu) = (a/mu)^a * x^(a-1) * exp(-a*x/mu) / Gamma(a)
 */
DFUSE_HOST_DEVICE inline Real gamma_pdf(Real x, Real shape, Real mean) {
    if (x <= Real(0) || shape <= Real(0) || mean <= Real(0)) return Real(0);
    
    Real rate = shape / mean;
    Real log_pdf = shape * std::log(rate) + (shape - Real(1)) * std::log(x) 
                   - rate * x - std::lgamma(shape);
    return std::exp(log_pdf);
}

// ============================================================================
// UNIT HYDROGRAPH GENERATION
// ============================================================================

/**
 * @brief Generate discrete unit hydrograph from gamma distribution
 * 
 * Implements Clark et al. (2008) equation (13):
 *   h[t] = P(a, t*a/mu) - P(a, (t-1)*a/mu)
 * 
 * @param shape Shape parameter 'a' (typically 2-5, default 3)
 * @param mean_delay Mean delay time mu_t (days)
 * @param dt Time step (days)
 * @param[out] uh Unit hydrograph weights
 * @param max_length Maximum UH length
 * @return Actual length of UH
 */
inline int generate_unit_hydrograph(
    Real shape,
    Real mean_delay,
    Real dt,
    std::vector<Real>& uh,
    int max_length = 100
) {
    uh.clear();
    uh.reserve(max_length);
    
    Real cumsum = Real(0);
    Real prev_cdf = Real(0);
    
    for (int i = 0; i < max_length; ++i) {
        Real t = (i + 1) * dt;
        Real x = t * shape / mean_delay;
        Real curr_cdf = gamma_cdf(shape, x);
        
        Real weight = curr_cdf - prev_cdf;
        uh.push_back(weight);
        cumsum += weight;
        prev_cdf = curr_cdf;
        
        // Stop when remaining mass is negligible
        if (Real(1) - cumsum < Real(1e-6)) {
            break;
        }
    }
    
    // Normalize to ensure sum = 1
    Real sum = Real(0);
    for (Real w : uh) sum += w;
    if (sum > Real(0)) {
        for (Real& w : uh) w /= sum;
    }
    
    return static_cast<int>(uh.size());
}

// ============================================================================
// ROUTING BUFFER (Convolution State)
// ============================================================================

/**
 * @brief Routing buffer for storing delayed runoff
 * 
 * Maintains a circular buffer of past runoff values and applies
 * convolution with the unit hydrograph.
 */
class RoutingBuffer {
public:
    RoutingBuffer() : buffer_pos_(0), initialized_(false) {}
    
    /**
     * @brief Construct and initialize routing buffer
     */
    RoutingBuffer(Real shape, Real mean_delay, Real dt, int max_length = 100) 
        : buffer_pos_(0), initialized_(false) 
    {
        initialize(shape, mean_delay, dt, max_length);
    }
    
    /**
     * @brief Initialize routing buffer
     */
    void initialize(Real shape, Real mean_delay, Real dt, int max_length = 100) {
        generate_unit_hydrograph(shape, mean_delay, dt, uh_, max_length);
        buffer_.resize(uh_.size(), Real(0));
        buffer_pos_ = 0;
        initialized_ = true;
    }
    
    /**
     * @brief Apply routing to instantaneous runoff
     * 
     * @param instant_runoff Instantaneous runoff from bucket model (mm/day)
     * @return Routed runoff at basin outlet (mm/day)
     */
    Real route(Real instant_runoff) {
        if (!initialized_ || uh_.empty()) {
            return instant_runoff;
        }
        
        // Add new runoff to buffer
        buffer_[buffer_pos_] = instant_runoff;
        
        // Convolve with unit hydrograph
        Real routed = Real(0);
        int n = static_cast<int>(uh_.size());
        for (int i = 0; i < n; ++i) {
            int buf_idx = (buffer_pos_ - i + n) % n;
            routed += buffer_[buf_idx] * uh_[i];
        }
        
        // Advance buffer position
        buffer_pos_ = (buffer_pos_ + 1) % n;
        
        return routed;
    }
    
    /**
     * @brief Reset buffer (e.g., for new simulation)
     */
    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), Real(0));
        buffer_pos_ = 0;
    }
    
    /**
     * @brief Get the unit hydrograph weights
     */
    const std::vector<Real>& get_unit_hydrograph() const { return uh_; }
    
    /**
     * @brief Get UH length
     */
    int get_uh_length() const { return static_cast<int>(uh_.size()); }
    
private:
    std::vector<Real> uh_;      // Unit hydrograph weights
    std::vector<Real> buffer_;  // Circular buffer of past runoff
    int buffer_pos_;
    bool initialized_;
};

// ============================================================================
// GPU-COMPATIBLE FIXED-LENGTH ROUTING
// ============================================================================

/**
 * @brief Fixed-length routing for GPU kernels
 * 
 * Uses a statically-sized array for GPU compatibility.
 */
template<int MAX_UH_LENGTH = 32>
struct FixedRoutingBuffer {
    Real uh[MAX_UH_LENGTH];      // Unit hydrograph weights
    Real buffer[MAX_UH_LENGTH];  // Circular buffer
    int uh_length;
    int buffer_pos;
    
    DFUSE_HOST_DEVICE void initialize(Real shape, Real mean_delay, Real dt) {
        // Generate UH
        uh_length = 0;
        Real cumsum = Real(0);
        Real prev_cdf = Real(0);
        
        for (int i = 0; i < MAX_UH_LENGTH; ++i) {
            Real t = (i + 1) * dt;
            Real x = t * shape / mean_delay;
            Real curr_cdf = gamma_cdf(shape, x);
            
            uh[i] = curr_cdf - prev_cdf;
            cumsum += uh[i];
            prev_cdf = curr_cdf;
            uh_length = i + 1;
            
            if (Real(1) - cumsum < Real(1e-6)) break;
        }
        
        // Normalize
        if (cumsum > Real(0)) {
            for (int i = 0; i < uh_length; ++i) {
                uh[i] /= cumsum;
            }
        }
        
        // Clear buffer
        for (int i = 0; i < MAX_UH_LENGTH; ++i) {
            buffer[i] = Real(0);
        }
        buffer_pos = 0;
    }
    
    DFUSE_HOST_DEVICE Real route(Real instant_runoff) {
        if (uh_length == 0) return instant_runoff;
        
        // Add to buffer
        buffer[buffer_pos] = instant_runoff;
        
        // Convolve
        Real routed = Real(0);
        for (int i = 0; i < uh_length; ++i) {
            int idx = (buffer_pos - i + uh_length) % uh_length;
            routed += buffer[idx] * uh[i];
        }
        
        buffer_pos = (buffer_pos + 1) % uh_length;
        return routed;
    }
    
    DFUSE_HOST_DEVICE void reset() {
        for (int i = 0; i < MAX_UH_LENGTH; ++i) {
            buffer[i] = Real(0);
        }
        buffer_pos = 0;
    }
};

// ============================================================================
// BATCH ROUTING FOR TIME SERIES
// ============================================================================

/**
 * @brief Apply routing to entire time series (batch mode)
 * 
 * More efficient than step-by-step routing when full time series is available.
 * 
 * @param instant_runoff Input instantaneous runoff [n_timesteps]
 * @param shape Gamma shape parameter
 * @param mean_delay Mean delay time (days)
 * @param dt Time step (days)
 * @param[out] routed_runoff Output routed runoff [n_timesteps]
 * @param n_timesteps Number of timesteps
 */
inline void route_timeseries(
    const Real* instant_runoff,
    Real shape,
    Real mean_delay,
    Real dt,
    Real* routed_runoff,
    int n_timesteps
) {
    // Generate unit hydrograph
    std::vector<Real> uh;
    int uh_len = generate_unit_hydrograph(shape, mean_delay, dt, uh);
    
    // Convolution (direct implementation)
    for (int t = 0; t < n_timesteps; ++t) {
        Real sum = Real(0);
        for (int i = 0; i < uh_len && i <= t; ++i) {
            sum += instant_runoff[t - i] * uh[i];
        }
        routed_runoff[t] = sum;
    }
}

/**
 * @brief Apply routing to batch of basins
 * 
 * @param instant_runoff Input [n_timesteps, n_basins] column major
 * @param shapes Shape parameters [n_basins]
 * @param mean_delays Mean delays [n_basins]
 * @param dt Time step
 * @param routed_runoff Output [n_timesteps, n_basins]
 * @param n_timesteps Number of timesteps
 * @param n_basins Number of basins
 */
inline void route_timeseries_batch(
    const Real* instant_runoff,
    const Real* shapes,
    const Real* mean_delays,
    Real dt,
    Real* routed_runoff,
    int n_timesteps,
    int n_basins
) {
    #pragma omp parallel for
    for (int b = 0; b < n_basins; ++b) {
        std::vector<Real> uh;
        int uh_len = generate_unit_hydrograph(shapes[b], mean_delays[b], dt, uh);
        
        for (int t = 0; t < n_timesteps; ++t) {
            Real sum = Real(0);
            for (int i = 0; i < uh_len && i <= t; ++i) {
                int src_idx = (t - i) * n_basins + b;
                sum += instant_runoff[src_idx] * uh[i];
            }
            routed_runoff[t * n_basins + b] = sum;
        }
    }
}

// ============================================================================
// CUDA ROUTING KERNEL
// ============================================================================

#ifdef __CUDACC__

/**
 * @brief CUDA kernel for batch routing
 * 
 * Each thread handles one basin's entire time series.
 */
__global__ void route_timeseries_kernel(
    const Real* instant_runoff,  // [n_timesteps, n_basins]
    const Real* shapes,          // [n_basins]
    const Real* mean_delays,     // [n_basins]
    Real dt,
    Real* routed_runoff,         // [n_timesteps, n_basins]
    int n_timesteps,
    int n_basins
) {
    int basin = blockIdx.x * blockDim.x + threadIdx.x;
    if (basin >= n_basins) return;
    
    // Generate UH for this basin
    FixedRoutingBuffer<64> router;
    router.initialize(shapes[basin], mean_delays[basin], dt);
    
    // Route each timestep
    for (int t = 0; t < n_timesteps; ++t) {
        Real instant = instant_runoff[t * n_basins + basin];
        routed_runoff[t * n_basins + basin] = router.route(instant);
    }
}

#endif // __CUDACC__

// ============================================================================
// DIFFERENTIABLE ROUTING
// ============================================================================

/**
 * @brief Routing with gradient computation
 * 
 * Computes both routed runoff and gradients w.r.t. UH parameters.
 */
class DifferentiableRouter {
public:
    DifferentiableRouter(Real shape = Real(3), Real mean_delay = Real(1), Real dt = Real(1))
        : shape_(shape), mean_delay_(mean_delay), dt_(dt) {
        generate_unit_hydrograph(shape_, mean_delay_, dt_, uh_);
        buffer_.resize(uh_.size(), Real(0));
    }
    
    /**
     * @brief Forward pass
     */
    Real forward(Real instant_runoff) {
        // Store input for backward pass
        inputs_.push_back(instant_runoff);
        
        // Add to buffer
        buffer_[buffer_pos_] = instant_runoff;
        
        // Convolve
        Real routed = Real(0);
        int n = static_cast<int>(uh_.size());
        for (int i = 0; i < n; ++i) {
            int idx = (buffer_pos_ - i + n) % n;
            routed += buffer_[idx] * uh_[i];
        }
        
        buffer_pos_ = (buffer_pos_ + 1) % n;
        return routed;
    }
    
    /**
     * @brief Backward pass - compute gradient w.r.t. instant_runoff
     * 
     * @param grad_output Gradient of loss w.r.t. routed output
     * @param t Timestep index
     * @return Gradient w.r.t. instant_runoff at timestep t
     */
    Real backward_input(Real grad_output, int t) {
        // d(routed[t]) / d(instant[t-i]) = uh[i]
        // So d(loss)/d(instant[s]) = sum over t where s <= t of grad_output[t] * uh[t-s]
        
        // For single timestep, gradient flows back through all affected UH weights
        if (t < static_cast<int>(inputs_.size())) {
            Real grad_sum = Real(0);
            for (int i = 0; i < static_cast<int>(uh_.size()) && t + i < static_cast<int>(grad_outputs_.size()); ++i) {
                grad_sum += grad_outputs_[t + i] * uh_[i];
            }
            return grad_sum;
        }
        return Real(0);
    }
    
    /**
     * @brief Compute gradient w.r.t. UH parameters (shape, mean_delay)
     * 
     * Uses numerical differentiation as gamma function gradients are complex.
     */
    void backward_params(
        const std::vector<Real>& grad_outputs,
        Real& grad_shape,
        Real& grad_mean_delay
    ) {
        grad_outputs_ = grad_outputs;
        const Real eps = Real(1e-5);
        
        // Numerical gradient w.r.t. shape
        std::vector<Real> uh_plus, uh_minus;
        generate_unit_hydrograph(shape_ + eps, mean_delay_, dt_, uh_plus);
        generate_unit_hydrograph(shape_ - eps, mean_delay_, dt_, uh_minus);
        
        grad_shape = compute_param_grad(uh_plus, uh_minus, eps);
        
        // Numerical gradient w.r.t. mean_delay
        generate_unit_hydrograph(shape_, mean_delay_ + eps, dt_, uh_plus);
        generate_unit_hydrograph(shape_, mean_delay_ - eps, dt_, uh_minus);
        
        grad_mean_delay = compute_param_grad(uh_plus, uh_minus, eps);
    }
    
    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), Real(0));
        buffer_pos_ = 0;
        inputs_.clear();
        grad_outputs_.clear();
    }
    
private:
    Real shape_, mean_delay_, dt_;
    std::vector<Real> uh_;
    std::vector<Real> buffer_;
    int buffer_pos_ = 0;
    
    std::vector<Real> inputs_;
    std::vector<Real> grad_outputs_;
    
    Real compute_param_grad(
        const std::vector<Real>& uh_plus,
        const std::vector<Real>& uh_minus,
        Real eps
    ) {
        // d(loss)/d(param) = sum_t grad_output[t] * d(routed[t])/d(param)
        //                  = sum_t grad_output[t] * sum_i input[t-i] * d(uh[i])/d(param)
        
        Real grad = Real(0);
        int n_t = static_cast<int>(inputs_.size());
        int n_uh = std::min(static_cast<int>(uh_plus.size()), 
                           static_cast<int>(uh_minus.size()));
        
        for (int t = 0; t < n_t && t < static_cast<int>(grad_outputs_.size()); ++t) {
            for (int i = 0; i < n_uh && i <= t; ++i) {
                Real duh = (uh_plus[i] - uh_minus[i]) / (2 * eps);
                grad += grad_outputs_[t] * inputs_[t - i] * duh;
            }
        }
        
        return grad;
    }
};

} // namespace routing
} // namespace dfuse
