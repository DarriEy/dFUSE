/**
 * @file config.hpp
 * @brief Physics configuration and switchboard for cFUSE
 * 
 * Differentiable Framework for Understanding Structural Errors (cFUSE)
 * Based on Clark et al. (2008) WRR, doi:10.1029/2007WR006735
 * 
 * This file defines the modeling decisions available in cFUSE, which can be
 * selected at runtime or compile-time via template specialization.
 */

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define DFUSE_HOST_DEVICE __host__ __device__
#define DFUSE_DEVICE __device__
#else
#define DFUSE_HOST_DEVICE
#define DFUSE_DEVICE
#endif

namespace dfuse {

// ============================================================================
// Precision configuration
// ============================================================================

#ifdef DFUSE_USE_DOUBLE
using Real = double;
#else
using Real = float;
#endif

// ============================================================================
// MODELING DECISIONS - Based on Clark et al. (2008) Table of decisions
// ============================================================================

/**
 * @brief Upper layer (unsaturated zone) architecture
 * 
 * Controls the state variable structure for the upper soil layer.
 * See Clark et al. (2008) equations (1a), (1b), (1c)
 */
enum class UpperLayerArch : uint8_t {
    SINGLE_STATE = 0,      ///< Single state S1 (TOPMODEL, ARNO/VIC style) - Eq 1a
    TENSION_FREE = 1,      ///< Separate tension (S1_T) and free (S1_F) storage (Sacramento) - Eq 1b
    TENSION2_FREE = 2      ///< Two tension stores + free storage (PRMS) - Eq 1c
};

/**
 * @brief Lower layer (saturated zone) architecture
 * 
 * Controls the state variable structure and base flow mechanism.
 * Tied to base flow parameterization in original FUSE.
 * See Clark et al. (2008) equations (2a), (2b), (2c)
 */
enum class LowerLayerArch : uint8_t {
    SINGLE_NOEVAP = 0,     ///< Single reservoir, infinite, no evaporation (TOPMODEL/PRMS) - Eq 2a
    SINGLE_EVAP = 1,       ///< Single reservoir with evaporation, fixed capacity (ARNO/VIC) - Eq 2b  
    TENSION_2RESERV = 2    ///< Tension + two parallel reservoirs (Sacramento) - Eq 2c
};

/**
 * @brief Base flow parameterization
 * 
 * Different functional forms for computing base flow from lower layer storage.
 * See Clark et al. (2008) equations (6a)-(6d)
 */
enum class BaseflowType : uint8_t {
    LINEAR = 0,            ///< Single linear reservoir: qb = v*S2 (PRMS) - Eq 6a
    PARALLEL_LINEAR = 1,   ///< Two parallel linear reservoirs (Sacramento) - Eq 6b
    NONLINEAR = 2,         ///< Nonlinear power function (ARNO/VIC) - Eq 6c
    TOPMODEL = 3           ///< TOPMODEL power law transmissivity - Eq 6d
};

/**
 * @brief Percolation parameterization
 * 
 * Controls vertical water movement from upper to lower layer.
 * See Clark et al. (2008) equations (4a)-(4c)
 */
enum class PercolationType : uint8_t {
    TOTAL_STORAGE = 0,     ///< Based on total upper zone storage (VIC style) - Eq 4a
    FREE_STORAGE = 1,      ///< Based on free storage above field capacity (PRMS) - Eq 4b
    LOWER_DEMAND = 2       ///< Driven by lower zone demand (Sacramento) - Eq 4c
};

/**
 * @brief Surface runoff (saturated area) parameterization
 * 
 * Controls how saturated contributing area is computed.
 * See Clark et al. (2008) equations (9a)-(9c)
 */
enum class SurfaceRunoffType : uint8_t {
    UZ_LINEAR = 0,         ///< Linear function of tension storage (PRMS) - Eq 9a
    UZ_PARETO = 1,         ///< Pareto distribution / VIC 'b' curve (ARNO/VIC) - Eq 9b
    LZ_GAMMA = 2           ///< TOPMODEL topographic index distribution - Eq 9c
};

/**
 * @brief Evaporation parameterization
 * 
 * Controls partitioning of evaporative demand between soil layers.
 * See Clark et al. (2008) equations (3a)-(3d)
 */
enum class EvaporationType : uint8_t {
    SEQUENTIAL = 0,        ///< Sequential: upper layer first, then lower - Eq 3a,3b
    ROOT_WEIGHT = 1        ///< Root weighting between layers - Eq 3c,3d
};

/**
 * @brief Interflow parameterization
 * 
 * Controls lateral subsurface flow from upper layer.
 * See Clark et al. (2008) equations (5a)-(5b)
 */
enum class InterflowType : uint8_t {
    NONE = 0,              ///< No interflow - Eq 5a
    LINEAR = 1             ///< Linear function of free storage - Eq 5b
};

// ============================================================================
// MODEL CONFIGURATION STRUCTURE
// ============================================================================

/**
 * @brief Complete model configuration specifying all physics options
 * 
 * This struct encapsulates all modeling decisions. Can be used either:
 * - At runtime with function pointer dispatch
 * - At compile-time via template specialization for maximum performance
 */
struct ModelConfig {
    UpperLayerArch upper_arch = UpperLayerArch::SINGLE_STATE;
    LowerLayerArch lower_arch = LowerLayerArch::SINGLE_NOEVAP;
    BaseflowType baseflow = BaseflowType::LINEAR;
    PercolationType percolation = PercolationType::TOTAL_STORAGE;
    SurfaceRunoffType surface_runoff = SurfaceRunoffType::UZ_LINEAR;
    EvaporationType evaporation = EvaporationType::SEQUENTIAL;
    InterflowType interflow = InterflowType::NONE;
    
    // Snow module (extension from Henn et al. 2015)
    bool enable_snow = true;
    
    // Numerical options
    bool adaptive_timestepping = true;
    Real timestep_tolerance = 1e-4;
    Real max_substep = 1.0;  // days
    
    DFUSE_HOST_DEVICE constexpr bool operator==(const ModelConfig& other) const {
        return upper_arch == other.upper_arch &&
               lower_arch == other.lower_arch &&
               baseflow == other.baseflow &&
               percolation == other.percolation &&
               surface_runoff == other.surface_runoff &&
               evaporation == other.evaporation &&
               interflow == other.interflow;
    }
    
    /**
     * @brief Get unique model ID (0-78 for valid combinations, per Clark et al. 2008)
     */
    DFUSE_HOST_DEVICE int model_id() const;
    
    /**
     * @brief Check if configuration is valid (compatible component combinations)
     */
    DFUSE_HOST_DEVICE bool is_valid() const;
};

// ============================================================================
// PREDEFINED MODEL CONFIGURATIONS (Parent models from Clark et al. 2008)
// ============================================================================

namespace models {

/// PRMS-style configuration
constexpr ModelConfig PRMS = {
    .upper_arch = UpperLayerArch::TENSION2_FREE,
    .lower_arch = LowerLayerArch::SINGLE_NOEVAP,
    .baseflow = BaseflowType::LINEAR,
    .percolation = PercolationType::FREE_STORAGE,
    .surface_runoff = SurfaceRunoffType::UZ_LINEAR,
    .evaporation = EvaporationType::SEQUENTIAL,
    .interflow = InterflowType::LINEAR
};

/// Sacramento model configuration
constexpr ModelConfig SACRAMENTO = {
    .upper_arch = UpperLayerArch::TENSION_FREE,
    .lower_arch = LowerLayerArch::TENSION_2RESERV,
    .baseflow = BaseflowType::PARALLEL_LINEAR,
    .percolation = PercolationType::LOWER_DEMAND,
    .surface_runoff = SurfaceRunoffType::UZ_LINEAR,
    .evaporation = EvaporationType::SEQUENTIAL,
    .interflow = InterflowType::NONE
};

/// TOPMODEL configuration  
constexpr ModelConfig TOPMODEL = {
    .upper_arch = UpperLayerArch::SINGLE_STATE,
    .lower_arch = LowerLayerArch::SINGLE_NOEVAP,
    .baseflow = BaseflowType::TOPMODEL,
    .percolation = PercolationType::TOTAL_STORAGE,
    .surface_runoff = SurfaceRunoffType::LZ_GAMMA,
    .evaporation = EvaporationType::SEQUENTIAL,
    .interflow = InterflowType::NONE
};

/// VIC/ARNO configuration
constexpr ModelConfig VIC = {
    .upper_arch = UpperLayerArch::SINGLE_STATE,
    .lower_arch = LowerLayerArch::SINGLE_EVAP,
    .baseflow = BaseflowType::NONLINEAR,
    .percolation = PercolationType::TOTAL_STORAGE,
    .surface_runoff = SurfaceRunoffType::UZ_PARETO,
    .evaporation = EvaporationType::ROOT_WEIGHT,
    .interflow = InterflowType::NONE
};

} // namespace models

// ============================================================================
// COMPILE-TIME MODEL CONFIGURATION TEMPLATE
// ============================================================================

/**
 * @brief Compile-time model configuration for maximum performance
 * 
 * Use this when the model structure is known at compile time.
 * Template parameters are constexpr, enabling aggressive optimization
 * and elimination of unused code paths.
 */
template<
    UpperLayerArch UPPER = UpperLayerArch::SINGLE_STATE,
    LowerLayerArch LOWER = LowerLayerArch::SINGLE_NOEVAP,
    BaseflowType BASEFLOW = BaseflowType::LINEAR,
    PercolationType PERCOL = PercolationType::TOTAL_STORAGE,
    SurfaceRunoffType SRUNOFF = SurfaceRunoffType::UZ_LINEAR,
    EvaporationType EVAP = EvaporationType::SEQUENTIAL,
    InterflowType INTERF = InterflowType::NONE
>
struct StaticConfig {
    static constexpr UpperLayerArch upper_arch = UPPER;
    static constexpr LowerLayerArch lower_arch = LOWER;
    static constexpr BaseflowType baseflow = BASEFLOW;
    static constexpr PercolationType percolation = PERCOL;
    static constexpr SurfaceRunoffType surface_runoff = SRUNOFF;
    static constexpr EvaporationType evaporation = EVAP;
    static constexpr InterflowType interflow = INTERF;
    
    // Derived properties
    static constexpr int num_upper_states = 
        (UPPER == UpperLayerArch::SINGLE_STATE) ? 1 :
        (UPPER == UpperLayerArch::TENSION_FREE) ? 2 : 3;
        
    static constexpr int num_lower_states =
        (LOWER == LowerLayerArch::SINGLE_NOEVAP) ? 1 :
        (LOWER == LowerLayerArch::SINGLE_EVAP) ? 1 : 3;
        
    static constexpr int num_states = num_upper_states + num_lower_states;
    
    static constexpr bool has_lower_evap = 
        (LOWER == LowerLayerArch::SINGLE_EVAP) || 
        (LOWER == LowerLayerArch::TENSION_2RESERV);
        
    static constexpr bool has_interflow = (INTERF == InterflowType::LINEAR);
    
    /// Convert to runtime config
    DFUSE_HOST_DEVICE static constexpr ModelConfig to_runtime() {
        return ModelConfig{UPPER, LOWER, BASEFLOW, PERCOL, SRUNOFF, EVAP, INTERF};
    }
};

// Convenience aliases for parent models
using PRMS_Config = StaticConfig<
    UpperLayerArch::TENSION2_FREE,
    LowerLayerArch::SINGLE_NOEVAP,
    BaseflowType::LINEAR,
    PercolationType::FREE_STORAGE,
    SurfaceRunoffType::UZ_LINEAR,
    EvaporationType::SEQUENTIAL,
    InterflowType::LINEAR
>;

using Sacramento_Config = StaticConfig<
    UpperLayerArch::TENSION_FREE,
    LowerLayerArch::TENSION_2RESERV,
    BaseflowType::PARALLEL_LINEAR,
    PercolationType::LOWER_DEMAND,
    SurfaceRunoffType::UZ_LINEAR,
    EvaporationType::SEQUENTIAL,
    InterflowType::NONE
>;

using TOPMODEL_Config = StaticConfig<
    UpperLayerArch::SINGLE_STATE,
    LowerLayerArch::SINGLE_NOEVAP,
    BaseflowType::TOPMODEL,
    PercolationType::TOTAL_STORAGE,
    SurfaceRunoffType::LZ_GAMMA,
    EvaporationType::SEQUENTIAL,
    InterflowType::NONE
>;

using VIC_Config = StaticConfig<
    UpperLayerArch::SINGLE_STATE,
    LowerLayerArch::SINGLE_EVAP,
    BaseflowType::NONLINEAR,
    PercolationType::TOTAL_STORAGE,
    SurfaceRunoffType::UZ_PARETO,
    EvaporationType::ROOT_WEIGHT,
    InterflowType::NONE
>;

} // namespace dfuse
