# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2024-12-28
- Rename project/package to cFUSE (cfuse/cfuse_core).
- Switch to MIT license and update metadata.
- Move example data to GitHub release assets; README documents the download path.
- Update the Python optimization workflow to use batch APIs and routed gradients.
- Add a Python batch smoke test to the CMake test suite.
- Add `route_runoff` binding in `cfuse_core`.
- Normalize Python package layout (`cfuse.netcdf`, `cfuse.torch`, legacy model module).
- Add CMake-based Python build (`setup.py`), `MANIFEST.in`, and a build-dist workflow.
