/**
 * @file state.hpp
 * @brief State variables, fluxes, and parameters for cFUSE
 * 
 * Differentiable Framework for Understanding Structural Errors (cFUSE)
 * Based on Clark et al. (2008) WRR, doi:10.1029/2007WR006735
 * 
 * This file defines the data structures for model states, fluxes, and parameters.
 * Two layouts are provided:
 * - Array-of-Structs (AoS): For single-HRU operations
 * - Struct-of-Arrays (SoA): For GPU-efficient batch operations
 */

#pragma once

#include "config.hpp"
#include <cmath>

namespace dfuse {

// ============================================================================
// CONSTANTS
// ============================================================================

constexpr int MAX_UPPER_STATES = 3;  // S1_TA, S1_TB, S1_F for PRMS
constexpr int MAX_LOWER_STATES = 3;  // S2_T, S2_FA, S2_FB for Sacramento
constexpr int MAX_TOTAL_STATES = MAX_UPPER_STATES + MAX_LOWER_STATES + 1; // +1 for snow
constexpr int NUM_PARAMETERS = 29;   // Total adjustable parameters
constexpr int NUM_FLUXES = 16;       // Number of diagnostic fluxes

// ============================================================================
// STATE VARIABLES (Array-of-Structs layout)
// ============================================================================

/**
 * @brief Model state variables for a single HRU
 * 
 * State variable naming follows Clark et al. (2008) Table 1:
 * - S1: Upper layer total storage
 * - S1_T: Upper layer tension storage  
 * - S1_TA: Primary tension storage (upper)
 * - S1_TB: Secondary tension storage (upper)
 * - S1_F: Free storage (upper)
 * - S2: Lower layer total storage
 * - S2_T: Lower layer tension storage
 * - S2_FA: Primary baseflow reservoir
 * - S2_FB: Secondary baseflow reservoir
 * - SWE: Snow water equivalent (extension)
 */
struct State {
    // Upper layer states (mm)
    Real S1 = Real(100);    ///< Total upper layer storage
    Real S1_T = Real(40);   ///< Tension storage (below field capacity)
    Real S1_TA = Real(20);  ///< Primary tension storage
    Real S1_TB = Real(20);  ///< Secondary tension storage  
    Real S1_F = Real(60);   ///< Free storage (above field capacity)
    
    // Lower layer states (mm)
    Real S2 = Real(400);    ///< Total lower layer storage
    Real S2_T = Real(160);  ///< Tension storage in lower layer
    Real S2_FA = Real(120); ///< Primary baseflow reservoir
    Real S2_FB = Real(120); ///< Secondary baseflow reservoir
    
    // Snow state (mm)
    Real SWE = Real(0);     ///< Snow water equivalent
    
    /**
     * @brief Initialize states based on model configuration
     */
    DFUSE_HOST_DEVICE void sync_derived(const ModelConfig& config) {
        switch (config.upper_arch) {
            case UpperLayerArch::SINGLE_STATE:
                // For single-state with interflow, partition S1 into tension/free
                // based on threshold (matches Fortran FUSE behavior)
                // S1_T_max should be set via params.compute_derived()
                // For now, use a simple partition: anything above threshold is "free"
                S1_T = S1;  // Track total as tension (for evap calculations)
                // S1_F computed in fuse_step using params.S1_T_max
                S1_TA = S1;
                S1_TB = Real(0);
                break;
            case UpperLayerArch::TENSION_FREE:
                // S1_T and S1_F are tracked
                S1 = S1_T + S1_F;
                S1_TA = S1_T;
                S1_TB = Real(0);
                break;
            case UpperLayerArch::TENSION2_FREE:
                // S1_TA, S1_TB, S1_F are tracked
                S1_T = S1_TA + S1_TB;
                S1 = S1_T + S1_F;
                break;
        }
        
        switch (config.lower_arch) {
            case LowerLayerArch::SINGLE_NOEVAP:
            case LowerLayerArch::SINGLE_EVAP:
                // S2 is the master state
                S2_T = S2;
                S2_FA = S2;
                S2_FB = Real(0);
                break;
            case LowerLayerArch::TENSION_2RESERV:
                // S2_T, S2_FA, S2_FB are tracked
                S2 = S2_T + S2_FA + S2_FB;
                break;
        }
    }
    
    /**
     * @brief Get state as flat array for numerical integration
     */
    DFUSE_HOST_DEVICE void to_array(Real* arr, const ModelConfig& config) const {
        int idx = 0;
        switch (config.upper_arch) {
            case UpperLayerArch::SINGLE_STATE:
                arr[idx++] = S1;
                break;
            case UpperLayerArch::TENSION_FREE:
                arr[idx++] = S1_T;
                arr[idx++] = S1_F;
                break;
            case UpperLayerArch::TENSION2_FREE:
                arr[idx++] = S1_TA;
                arr[idx++] = S1_TB;
                arr[idx++] = S1_F;
                break;
        }
        
        switch (config.lower_arch) {
            case LowerLayerArch::SINGLE_NOEVAP:
            case LowerLayerArch::SINGLE_EVAP:
                arr[idx++] = S2;
                break;
            case LowerLayerArch::TENSION_2RESERV:
                arr[idx++] = S2_T;
                arr[idx++] = S2_FA;
                arr[idx++] = S2_FB;
                break;
        }
        
        if (config.enable_snow) {
            arr[idx++] = SWE;
        }
    }
    
    /**
     * @brief Set state from flat array
     */
    DFUSE_HOST_DEVICE void from_array(const Real* arr, const ModelConfig& config) {
        int idx = 0;
        switch (config.upper_arch) {
            case UpperLayerArch::SINGLE_STATE:
                S1 = arr[idx++];
                break;
            case UpperLayerArch::TENSION_FREE:
                S1_T = arr[idx++];
                S1_F = arr[idx++];
                break;
            case UpperLayerArch::TENSION2_FREE:
                S1_TA = arr[idx++];
                S1_TB = arr[idx++];
                S1_F = arr[idx++];
                break;
        }
        
        switch (config.lower_arch) {
            case LowerLayerArch::SINGLE_NOEVAP:
            case LowerLayerArch::SINGLE_EVAP:
                S2 = arr[idx++];
                break;
            case LowerLayerArch::TENSION_2RESERV:
                S2_T = arr[idx++];
                S2_FA = arr[idx++];
                S2_FB = arr[idx++];
                break;
        }
        
        if (config.enable_snow) {
            SWE = arr[idx++];
        }
        
        sync_derived(config);
    }
};

// ============================================================================
// FLUX VARIABLES
// ============================================================================

/**
 * @brief Model fluxes for a single timestep
 * 
 * Flux naming follows Clark et al. (2008) Table 2.
 * All fluxes in mm/day.
 */
struct Flux {
    // Input fluxes
    Real rain;        ///< Rainfall (after snow partition)
    Real melt;        ///< Snowmelt
    Real throughfall; ///< Rain + melt reaching soil
    
    // Evaporation fluxes
    Real e1;          ///< Evaporation from upper layer
    Real e2;          ///< Evaporation from lower layer
    Real e1_A;        ///< Evap from primary tension store
    Real e1_B;        ///< Evap from secondary tension store
    Real e_total;     ///< Total actual evaporation
    
    // Runoff fluxes
    Real qsx;         ///< Saturation excess surface runoff
    Real qif;         ///< Interflow
    Real qb;          ///< Total baseflow
    Real qb_A;        ///< Baseflow from primary reservoir
    Real qb_B;        ///< Baseflow from secondary reservoir
    Real q_total;     ///< Total runoff (qsx + qif + qb)
    
    // Internal fluxes
    Real q12;         ///< Percolation from upper to lower
    
    // Overflow fluxes
    Real qurof;       ///< Primary tension overflow
    Real qutof;       ///< Tension storage overflow
    Real qufof;       ///< Free storage overflow (upper)
    Real qstof;       ///< Lower tension overflow
    Real qsfof;       ///< Lower free storage overflow
    Real qsfofa;      ///< Primary baseflow overflow
    Real qsfofb;      ///< Secondary baseflow overflow
    
    // Saturated area
    Real Ac;          ///< Saturated contributing area fraction
    
    DFUSE_HOST_DEVICE void compute_totals() {
        e_total = e1 + e2;
        q_total = qsx + qif + qb + qufof + qsfof;
    }
};

// ============================================================================
// MODEL PARAMETERS
// ============================================================================

/**
 * @brief Model parameters
 * 
 * Parameter naming and bounds follow Clark et al. (2008) Table 3.
 */
struct Parameters {
    // Storage parameters
    Real S1_max = Real(200);      ///< Maximum upper layer storage (mm) [50-5000]
    Real S2_max = Real(800);      ///< Maximum lower layer storage (mm) [100-10000]
    Real f_tens = Real(0.4);      ///< Fraction of storage as tension [-] [0.05-0.95]
    Real f_rchr = Real(0.3);      ///< Fraction tension in primary zone [-] [0.05-0.95]
    Real f_base = Real(0.3);      ///< Fraction free storage in primary reservoir [-] [0.05-0.95]
    
    // Evaporation parameters  
    Real r1 = Real(0.5);          ///< Root fraction in upper layer [-] [0.05-0.95]
    
    // Percolation parameters
    Real ku = Real(50);           ///< Percolation rate (mm/day) [0.01-1000]
    Real c = Real(2);             ///< Percolation exponent [-] [1-20]
    Real alpha = Real(100);       ///< Lower zone percolation multiplier [-] [1-250]
    Real psi = Real(2);           ///< Lower zone percolation exponent [-] [1-5]
    Real kappa = Real(0.3);       ///< Fraction percolation to tension storage [-] [0.05-0.95]
    
    // Interflow parameters
    Real ki = Real(20);           ///< Interflow rate (mm/day) [0.01-1000]
    
    // Baseflow parameters
    Real ks = Real(100);          ///< Baseflow rate (mm/day) [0.001-10000]
    Real n = Real(2);             ///< Baseflow exponent [-] [1-10]
    Real v = Real(0.05);          ///< Baseflow depletion rate (1/day) [0.001-0.25]
    Real v_A = Real(0.1);         ///< Primary reservoir depletion (1/day) [0.001-0.25]
    Real v_B = Real(0.01);        ///< Secondary reservoir depletion (1/day) [0.001-0.25]
    
    // Surface runoff parameters
    Real Ac_max = Real(0.5);      ///< Maximum saturated area fraction [-] [0.05-0.95]
    Real b = Real(0.3);           ///< ARNO/VIC 'b' exponent [-] [0.001-3.0]
    
    // TOPMODEL parameters
    Real lambda = Real(7);        ///< Mean log topographic index (m) [5-10]
    Real chi = Real(3);           ///< Shape parameter for topo index [-] [2-5]
    
    // Routing parameters
    Real mu_t = Real(1);          ///< Time delay in runoff (days) [0.01-5.0]
    Real shape_t = Real(3);       ///< Shape of gamma distribution [-] (typically fixed at 3)
    
    // Snow parameters (extension from Henn et al. 2015 / SNOW-17)
    Real T_rain = Real(2);        ///< Temperature threshold for rain (°C)
    Real T_melt = Real(0);        ///< Temperature threshold for melt (°C)
    Real melt_rate = Real(3);     ///< Degree-day melt factor (mm/°C/day) - used if MFMAX=MFMIN=0
    Real MFMAX = Real(4.2);       ///< Maximum melt factor on June 21 (mm/°C/day)
    Real MFMIN = Real(2.4);       ///< Minimum melt factor on Dec 21 (mm/°C/day)
    Real lapse_rate = Real(-5);   ///< Temperature lapse rate (°C/km, typically -5 to -7)
    Real opg = Real(0.5);         ///< Orographic precipitation gradient (km-1, typically 0-0.5)
    
    // Numerical smoothing (Kavetski & Kuczera 2007)
    Real smooth_frac = Real(0.01); ///< Smoothing fraction for thresholds [0.001-0.1]
    
    // ========================================================================
    // DERIVED PARAMETERS (computed from adjustable parameters)
    // ========================================================================
    
    // Derived storage capacities
    Real S1_T_max = Real(80);    ///< Max tension storage upper
    Real S2_T_max = Real(320);   ///< Max tension storage lower
    Real S1_F_max = Real(120);   ///< Max free storage upper
    Real S2_F_max = Real(480);   ///< Max free storage lower
    Real S1_TA_max = Real(24);   ///< Max primary tension upper
    Real S1_TB_max = Real(56);   ///< Max secondary tension upper
    Real S2_FA_max = Real(144);  ///< Max primary baseflow reservoir
    Real S2_FB_max = Real(336);  ///< Max secondary baseflow reservoir
    
    // Derived TOPMODEL parameters
    Real lambda_n = Real(7);     ///< Mean of power-transformed topo index
    Real m = Real(400);          ///< Subsurface depth scaling (S2_max/n)
    
    /**
     * @brief Compute derived parameters from adjustable parameters
     * FIXED: Use branch-free smooth operations for Enzyme AD compatibility
     */
    DFUSE_HOST_DEVICE void compute_derived() {
        // Storage partitioning (Clark et al. 2008 Table 4)
        S1_T_max = f_tens * S1_max;
        S2_T_max = f_tens * S2_max;
        S1_F_max = (Real(1) - f_tens) * S1_max;
        S2_F_max = (Real(1) - f_tens) * S2_max;
        
        S1_TA_max = f_rchr * S1_T_max;
        S1_TB_max = (Real(1) - f_rchr) * S1_T_max;
        S2_FA_max = f_base * S2_F_max;
        S2_FB_max = (Real(1) - f_base) * S2_F_max;
        
        // TOPMODEL derived - Branch-free smooth lower bound
        // smooth_max(n, eps) = 0.5*(n + eps + sqrt((n-eps)^2 + 4*k^2))
        constexpr Real eps_n = Real(0.1);
        constexpr Real k_smooth = Real(0.01);
        Real diff = n - eps_n;
        Real smooth_abs = std::sqrt(diff * diff + Real(4) * k_smooth * k_smooth);
        Real safe_n = Real(0.5) * (n + eps_n + smooth_abs);
        m = S2_max / safe_n;
        // lambda_n requires integration - computed separately
    }
    
    /**
     * @brief Get parameters as flat array for optimization
     */
    DFUSE_HOST_DEVICE void to_array(Real* arr) const {
        arr[0] = S1_max;
        arr[1] = S2_max;
        arr[2] = f_tens;
        arr[3] = f_rchr;
        arr[4] = f_base;
        arr[5] = r1;
        arr[6] = ku;
        arr[7] = c;
        arr[8] = alpha;
        arr[9] = psi;
        arr[10] = kappa;
        arr[11] = ki;
        arr[12] = ks;
        arr[13] = n;
        arr[14] = v;
        arr[15] = v_A;
        arr[16] = v_B;
        arr[17] = Ac_max;
        arr[18] = b;
        arr[19] = lambda;
        arr[20] = chi;
        arr[21] = mu_t;
        arr[22] = T_rain;
        arr[23] = T_melt;
        arr[24] = melt_rate;
        arr[25] = lapse_rate;
        arr[26] = opg;
        arr[27] = MFMAX;
        arr[28] = MFMIN;
    }
    
    /**
     * @brief Set parameters from flat array
     */
    DFUSE_HOST_DEVICE void from_array(const Real* arr) {
        S1_max = arr[0];
        S2_max = arr[1];
        f_tens = arr[2];
        f_rchr = arr[3];
        f_base = arr[4];
        r1 = arr[5];
        ku = arr[6];
        c = arr[7];
        alpha = arr[8];
        psi = arr[9];
        kappa = arr[10];
        ki = arr[11];
        ks = arr[12];
        n = arr[13];
        v = arr[14];
        v_A = arr[15];
        v_B = arr[16];
        Ac_max = arr[17];
        b = arr[18];
        lambda = arr[19];
        chi = arr[20];
        mu_t = arr[21];
        T_rain = arr[22];
        T_melt = arr[23];
        melt_rate = arr[24];
        lapse_rate = arr[25];
        opg = arr[26];
        MFMAX = arr[27];
        MFMIN = arr[28];
        
        compute_derived();
    }
};

// ============================================================================
// FORCING DATA
// ============================================================================

/**
 * @brief Meteorological forcing for a single timestep
 */
struct Forcing {
    Real precip;      ///< Total precipitation (mm/day)
    Real pet;         ///< Potential evapotranspiration (mm/day)
    Real temp;        ///< Air temperature (°C) - for snow module
    
    DFUSE_HOST_DEVICE Forcing() : precip(0), pet(0), temp(0) {}
    DFUSE_HOST_DEVICE Forcing(Real p, Real pe, Real t) : precip(p), pet(pe), temp(t) {}
};

// ============================================================================
// STRUCT-OF-ARRAYS LAYOUT FOR GPU BATCH OPERATIONS
// ============================================================================

/**
 * @brief Batch state storage in Struct-of-Arrays format
 * 
 * This layout enables coalesced memory access on GPU for parallel
 * operations across many HRUs/basins/ensemble members.
 */
struct StateBatch {
    int n;            ///< Number of elements in batch
    
    // Upper layer states (each array has n elements)
    Real* S1;
    Real* S1_T;
    Real* S1_TA;
    Real* S1_TB;
    Real* S1_F;
    
    // Lower layer states
    Real* S2;
    Real* S2_T;
    Real* S2_FA;
    Real* S2_FB;
    
    // Snow state
    Real* SWE;
    
    /**
     * @brief Get state for element i
     */
    DFUSE_HOST_DEVICE State get(int i) const {
        State s;
        s.S1 = S1[i];
        s.S1_T = S1_T[i];
        s.S1_TA = S1_TA[i];
        s.S1_TB = S1_TB[i];
        s.S1_F = S1_F[i];
        s.S2 = S2[i];
        s.S2_T = S2_T[i];
        s.S2_FA = S2_FA[i];
        s.S2_FB = S2_FB[i];
        s.SWE = SWE[i];
        return s;
    }
    
    /**
     * @brief Set state for element i
     */
    DFUSE_HOST_DEVICE void set(int i, const State& s) {
        S1[i] = s.S1;
        S1_T[i] = s.S1_T;
        S1_TA[i] = s.S1_TA;
        S1_TB[i] = s.S1_TB;
        S1_F[i] = s.S1_F;
        S2[i] = s.S2;
        S2_T[i] = s.S2_T;
        S2_FA[i] = s.S2_FA;
        S2_FB[i] = s.S2_FB;
        SWE[i] = s.SWE;
    }
};

/**
 * @brief Batch flux storage in Struct-of-Arrays format
 */
struct FluxBatch {
    int n;
    
    Real* rain;
    Real* melt;
    Real* e1;
    Real* e2;
    Real* qsx;
    Real* qif;
    Real* qb;
    Real* q12;
    Real* q_total;
    Real* Ac;
    
    DFUSE_HOST_DEVICE Flux get(int i) const;
    DFUSE_HOST_DEVICE void set(int i, const Flux& f);
};

/**
 * @brief Batch forcing storage in Struct-of-Arrays format
 */
struct ForcingBatch {
    int n;
    int nt;           ///< Number of timesteps
    
    Real* precip;     ///< [nt x n] column-major
    Real* pet;
    Real* temp;
    
    DFUSE_HOST_DEVICE Forcing get(int i, int t) const {
        int idx = t * n + i;  // Column-major for time-contiguous access
        return Forcing(precip[idx], pet[idx], temp[idx]);
    }
};

/**
 * @brief Batch parameter storage in Struct-of-Arrays format
 */
struct ParameterBatch {
    int n;
    
    Real* S1_max;
    Real* S2_max;
    Real* f_tens;
    Real* f_rchr;
    Real* f_base;
    Real* r1;
    Real* ku;
    Real* c;
    Real* alpha;
    Real* psi;
    Real* kappa;
    Real* ki;
    Real* ks;
    Real* n_exp;      // renamed to avoid conflict with n (count)
    Real* v;
    Real* v_A;
    Real* v_B;
    Real* Ac_max;
    Real* b;
    Real* lambda;
    Real* chi;
    Real* mu_t;
    Real* T_rain;
    Real* melt_rate;
    Real* smooth_frac;
    
    DFUSE_HOST_DEVICE Parameters get(int i) const;
    DFUSE_HOST_DEVICE void set(int i, const Parameters& p);
};

} // namespace dfuse
