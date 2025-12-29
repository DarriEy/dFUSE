#!/usr/bin/env python3
import numpy as np

import cfuse_core


def main() -> None:
    config = {
        "upper_arch": 0,
        "lower_arch": 1,
        "baseflow": 2,
        "percolation": 1,
        "surface_runoff": 1,
        "evaporation": 1,
        "interflow": 1,
        "enable_snow": True,
    }

    n_timesteps = 10
    n_hrus = 1
    n_params = int(cfuse_core.NUM_PARAMETERS)

    initial_states = np.zeros((n_hrus, 2), dtype=np.float32)
    forcing = np.ones((n_timesteps, n_hrus, 3), dtype=np.float32)
    params = np.full((n_params,), 0.5, dtype=np.float32)

    _, runoff = cfuse_core.run_fuse_batch(
        initial_states, forcing, params, config, 1.0
    )
    assert runoff.shape == (n_timesteps, n_hrus)

    grad_runoff = np.ones((n_timesteps, n_hrus), dtype=np.float32)
    if hasattr(cfuse_core, "run_fuse_batch_gradient"):
        grad_params = cfuse_core.run_fuse_batch_gradient(
            initial_states, forcing, params, grad_runoff, config, 1.0
        )
    else:
        grad_params = cfuse_core.run_fuse_batch_gradient_numerical(
            initial_states, forcing, params, grad_runoff, config, 1.0
        )

    assert grad_params.shape == (n_params,)
    assert np.all(np.isfinite(grad_params))

    routed = cfuse_core.route_runoff(runoff[:, 0], 2.5, 1.0, 1.0)
    assert routed.shape == (n_timesteps,)


if __name__ == "__main__":
    main()
