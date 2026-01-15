#!/usr/bin/env python3
"""
Comprehensive test suite for cFUSE Python bindings.

Tests:
- All preset model configurations
- Edge cases (empty storage, extreme parameters)
- Gradient verification (Enzyme vs numerical)
- Routing functionality
- Parameter array consistency
"""

import numpy as np
import pytest
from typing import Dict, Any

import cfuse_core
from cfuse import (
    FUSEConfig,
    VIC_CONFIG,
    TOPMODEL_CONFIG,
    ARNO_CONFIG,
    PRMS_CONFIG,
    SACRAMENTO_CONFIG,
    PARAM_NAMES,
    PARAM_BOUNDS,
    DEFAULT_PARAMS,
    get_default_params_array,
    HAS_ENZYME,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_params():
    """Get default parameter array."""
    return get_default_params_array()


@pytest.fixture
def simple_forcing():
    """Generate simple forcing data."""
    n_timesteps = 100
    n_hrus = 1
    forcing = np.zeros((n_timesteps, n_hrus, 3), dtype=np.float32)
    forcing[:, :, 0] = 5.0   # Precip: 5 mm/day
    forcing[:, :, 1] = 3.0   # PET: 3 mm/day
    forcing[:, :, 2] = 10.0  # Temp: 10 C
    return forcing


# =============================================================================
# PRESET CONFIGURATION TESTS
# =============================================================================

class TestPresetConfigs:
    """Test all preset model configurations run without errors."""

    @pytest.mark.parametrize("config,name", [
        (VIC_CONFIG, "VIC"),
        (TOPMODEL_CONFIG, "TOPMODEL"),
        (ARNO_CONFIG, "ARNO"),
        (PRMS_CONFIG, "PRMS"),
        (SACRAMENTO_CONFIG, "SACRAMENTO"),
    ])
    def test_config_runs(self, config, name, default_params, simple_forcing):
        """Test that each preset config runs successfully."""
        config_dict = config.to_dict()
        n_states = cfuse_core.get_num_active_states(config_dict)

        initial_states = np.zeros((1, n_states), dtype=np.float32)
        initial_states[0, 0] = 50.0  # Some initial upper storage

        final_states, runoff = cfuse_core.run_fuse_batch(
            initial_states, simple_forcing, default_params, config_dict, 1.0
        )

        assert runoff.shape == (100, 1), f"{name} runoff shape mismatch"
        assert np.all(np.isfinite(runoff)), f"{name} produced NaN/Inf"
        assert np.all(runoff >= 0), f"{name} produced negative runoff"

    @pytest.mark.parametrize("config,name", [
        (VIC_CONFIG, "VIC"),
        (TOPMODEL_CONFIG, "TOPMODEL"),
        (ARNO_CONFIG, "ARNO"),
        (PRMS_CONFIG, "PRMS"),
        (SACRAMENTO_CONFIG, "SACRAMENTO"),
    ])
    def test_config_produces_runoff(self, config, name, default_params):
        """Test that each config produces non-zero runoff with precip input."""
        config_dict = config.to_dict()
        n_states = cfuse_core.get_num_active_states(config_dict)
        n_timesteps = 365

        # Create forcing with realistic seasonal pattern
        forcing = np.zeros((n_timesteps, 1, 3), dtype=np.float32)
        forcing[:, 0, 0] = 10.0  # High precip
        forcing[:, 0, 1] = 2.0   # Low PET
        forcing[:, 0, 2] = 15.0  # Warm (no snow)

        initial_states = np.zeros((1, n_states), dtype=np.float32)
        initial_states[0, 0] = 100.0  # Start with some storage

        _, runoff = cfuse_core.run_fuse_batch(
            initial_states, forcing, default_params, config_dict, 1.0
        )

        total_runoff = np.sum(runoff)
        assert total_runoff > 0, f"{name} produced zero total runoff"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_precip(self, default_params):
        """Test behavior with zero precipitation."""
        config = VIC_CONFIG.to_dict()
        n_states = cfuse_core.get_num_active_states(config)

        forcing = np.zeros((50, 1, 3), dtype=np.float32)
        forcing[:, :, 1] = 3.0  # Only PET
        forcing[:, :, 2] = 10.0

        initial_states = np.zeros((1, n_states), dtype=np.float32)
        initial_states[0, 0] = 200.0  # Start with storage

        _, runoff = cfuse_core.run_fuse_batch(
            initial_states, forcing, default_params, config, 1.0
        )

        assert np.all(np.isfinite(runoff))
        # Total runoff should be bounded - can't produce more water than we have
        total_runoff = np.sum(runoff)
        max_possible = initial_states[0, 0]  # Can't exceed initial storage
        assert total_runoff <= max_possible * 2  # Allow some tolerance

    def test_extreme_precip(self, default_params):
        """Test behavior with extreme precipitation."""
        config = VIC_CONFIG.to_dict()
        n_states = cfuse_core.get_num_active_states(config)

        forcing = np.zeros((10, 1, 3), dtype=np.float32)
        forcing[:, :, 0] = 500.0  # Extreme precip
        forcing[:, :, 1] = 1.0
        forcing[:, :, 2] = 20.0

        initial_states = np.zeros((1, n_states), dtype=np.float32)

        _, runoff = cfuse_core.run_fuse_batch(
            initial_states, forcing, default_params, config, 1.0
        )

        assert np.all(np.isfinite(runoff))
        assert np.all(runoff >= 0)

    def test_freezing_conditions(self, default_params):
        """Test snow module with freezing temperatures."""
        config = VIC_CONFIG.to_dict()
        config['enable_snow'] = True
        n_states = cfuse_core.get_num_active_states(config)

        forcing = np.zeros((100, 1, 3), dtype=np.float32)
        forcing[:, :, 0] = 10.0  # Precip as snow
        forcing[:, :, 1] = 0.5
        forcing[:50, :, 2] = -10.0  # Cold period
        forcing[50:, :, 2] = 15.0   # Warm period (melt)

        initial_states = np.zeros((1, n_states), dtype=np.float32)

        _, runoff = cfuse_core.run_fuse_batch(
            initial_states, forcing, default_params, config, 1.0
        )

        assert np.all(np.isfinite(runoff))
        # Should see increased runoff during melt period
        cold_runoff = np.mean(runoff[:50])
        warm_runoff = np.mean(runoff[50:])
        assert warm_runoff > cold_runoff, "Expected more runoff during warm period"

    def test_multiple_hrus(self, default_params):
        """Test batch execution with multiple HRUs."""
        config = TOPMODEL_CONFIG.to_dict()
        n_states = cfuse_core.get_num_active_states(config)
        n_hrus = 10
        n_timesteps = 50

        # Different forcing per HRU
        forcing = np.random.rand(n_timesteps, n_hrus, 3).astype(np.float32)
        forcing[:, :, 0] *= 20  # Precip 0-20
        forcing[:, :, 1] *= 5   # PET 0-5
        forcing[:, :, 2] = forcing[:, :, 2] * 30 - 5  # Temp -5 to 25

        initial_states = np.zeros((n_hrus, n_states), dtype=np.float32)
        initial_states[:, 0] = 50.0

        final_states, runoff = cfuse_core.run_fuse_batch(
            initial_states, forcing, default_params, config, 1.0
        )

        assert runoff.shape == (n_timesteps, n_hrus)
        assert final_states.shape[0] == n_hrus
        assert np.all(np.isfinite(runoff))


# =============================================================================
# GRADIENT TESTS
# =============================================================================

class TestGradients:
    """Test gradient computation."""

    def test_gradient_finite(self, default_params):
        """Test that gradients are finite."""
        config = VIC_CONFIG.to_dict()
        n_states = cfuse_core.get_num_active_states(config)
        n_timesteps = 20
        n_hrus = 2

        forcing = np.ones((n_timesteps, n_hrus, 3), dtype=np.float32)
        forcing[:, :, 0] = 10.0
        forcing[:, :, 1] = 2.0
        forcing[:, :, 2] = 15.0

        initial_states = np.zeros((n_hrus, n_states), dtype=np.float32)
        initial_states[:, 0] = 50.0

        grad_runoff = np.ones((n_timesteps, n_hrus), dtype=np.float32)

        # Use Enzyme if available, otherwise numerical
        if hasattr(cfuse_core, 'run_fuse_batch_gradient'):
            grad_params = cfuse_core.run_fuse_batch_gradient(
                initial_states, forcing, default_params, grad_runoff, config, 1.0
            )
        else:
            grad_params = cfuse_core.run_fuse_batch_gradient_numerical(
                initial_states, forcing, default_params, grad_runoff, config, 1.0
            )

        assert grad_params.shape == default_params.shape
        assert np.all(np.isfinite(grad_params))

    @pytest.mark.skipif(not HAS_ENZYME, reason="Enzyme not available")
    def test_gradient_vs_numerical(self, default_params):
        """Compare Enzyme gradient with numerical gradient."""
        config = VIC_CONFIG.to_dict()
        n_states = cfuse_core.get_num_active_states(config)
        n_timesteps = 20
        n_hrus = 1

        np.random.seed(42)
        forcing = np.random.rand(n_timesteps, n_hrus, 3).astype(np.float32)
        forcing[:, :, 0] *= 15
        forcing[:, :, 1] *= 4
        forcing[:, :, 2] = forcing[:, :, 2] * 20 + 5

        initial_states = np.zeros((n_hrus, n_states), dtype=np.float32)
        initial_states[:, 0] = 50.0

        grad_runoff = np.random.rand(n_timesteps, n_hrus).astype(np.float32)

        # Enzyme gradient
        grad_enzyme = cfuse_core.run_fuse_batch_gradient(
            initial_states, forcing, default_params, grad_runoff, config, 1.0
        )

        # Numerical gradient
        grad_numerical = cfuse_core.run_fuse_batch_gradient_numerical(
            initial_states, forcing, default_params, grad_runoff, config, 1.0, 1e-5
        )

        # Compare (allow some tolerance for numerical error)
        relative_diff = np.abs(grad_enzyme - grad_numerical) / (np.abs(grad_numerical) + 1e-8)
        max_rel_diff = np.max(relative_diff)

        assert max_rel_diff < 0.15, f"Max relative gradient difference: {max_rel_diff}"


# =============================================================================
# ROUTING TESTS
# =============================================================================

class TestRouting:
    """Test gamma routing functionality."""

    def test_routing_basic(self):
        """Test basic routing functionality."""
        n_timesteps = 100
        instant_runoff = np.zeros(n_timesteps, dtype=np.float32)
        instant_runoff[10] = 100.0  # Impulse at t=10

        routed = cfuse_core.route_runoff(instant_runoff, 2.5, 1.0, 1.0)

        assert routed.shape == (n_timesteps,)
        assert np.all(np.isfinite(routed))
        assert np.all(routed >= 0)

        # Peak should be delayed
        peak_instant = np.argmax(instant_runoff)
        peak_routed = np.argmax(routed)
        assert peak_routed >= peak_instant

    def test_routing_mass_conservation(self):
        """Test that routing conserves mass."""
        n_timesteps = 200
        instant_runoff = np.random.rand(n_timesteps).astype(np.float32) * 10

        routed = cfuse_core.route_runoff(instant_runoff, 3.0, 2.0, 1.0)

        # Total mass should be approximately conserved
        # (small difference due to truncated UH)
        total_instant = np.sum(instant_runoff)
        total_routed = np.sum(routed)

        relative_diff = abs(total_instant - total_routed) / total_instant
        assert relative_diff < 0.05, f"Mass not conserved: {relative_diff:.2%}"

    def test_routing_different_shapes(self):
        """Test routing with different shape parameters."""
        n_timesteps = 100
        instant_runoff = np.zeros(n_timesteps, dtype=np.float32)
        instant_runoff[20] = 50.0

        # Small shape = more spread
        routed_small = cfuse_core.route_runoff(instant_runoff, 1.5, 2.0, 1.0)
        # Large shape = more peaked
        routed_large = cfuse_core.route_runoff(instant_runoff, 5.0, 2.0, 1.0)

        # Large shape should have higher peak
        assert np.max(routed_large) > np.max(routed_small)


# =============================================================================
# PARAMETER CONSISTENCY TESTS
# =============================================================================

class TestParameterConsistency:
    """Test parameter array consistency between Python and C++."""

    def test_param_count_matches(self):
        """Test that Python and C++ agree on parameter count."""
        assert len(PARAM_NAMES) == cfuse_core.NUM_PARAMETERS
        assert len(PARAM_BOUNDS) == cfuse_core.NUM_PARAMETERS
        assert len(DEFAULT_PARAMS) == cfuse_core.NUM_PARAMETERS

    def test_default_params_array_order(self):
        """Test that get_default_params_array matches PARAM_NAMES order."""
        params = get_default_params_array()

        for i, name in enumerate(PARAM_NAMES):
            expected = DEFAULT_PARAMS[name]
            actual = params[i]
            assert np.isclose(expected, actual), f"Mismatch at {name}: {expected} vs {actual}"

    def test_param_bounds_valid(self):
        """Test that all parameter bounds are valid."""
        for name, (lower, upper) in PARAM_BOUNDS.items():
            assert lower < upper, f"Invalid bounds for {name}: [{lower}, {upper}]"
            default = DEFAULT_PARAMS[name]
            assert lower <= default <= upper, \
                f"Default {default} outside bounds [{lower}, {upper}] for {name}"


# =============================================================================
# WATER BALANCE TESTS
# =============================================================================

class TestWaterBalance:
    """Test water balance properties."""

    def test_runoff_bounded_by_input(self, default_params):
        """Test that cumulative runoff doesn't exceed cumulative input."""
        config = VIC_CONFIG.to_dict()
        config['enable_snow'] = False  # Simpler without snow
        n_states = cfuse_core.get_num_active_states(config)
        n_timesteps = 100

        forcing = np.zeros((n_timesteps, 1, 3), dtype=np.float32)
        forcing[:, :, 0] = 10.0  # Precip
        forcing[:, :, 1] = 2.0   # PET
        forcing[:, :, 2] = 20.0  # Warm

        initial_states = np.zeros((1, n_states), dtype=np.float32)

        _, runoff = cfuse_core.run_fuse_batch(
            initial_states, forcing, default_params, config, 1.0
        )

        total_input = np.sum(forcing[:, :, 0])  # Total precip
        total_runoff = np.sum(runoff)

        # Runoff should not exceed input (water can be stored or evaporated)
        assert total_runoff <= total_input * 1.01  # 1% tolerance


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
