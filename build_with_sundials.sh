#!/bin/bash
# Build dFUSE with SUNDIALS support
#
# Usage:
#   ./build_with_sundials.sh /path/to/sundials/install
#
# Example for SYMFLUENCE sundials:
#   ./build_with_sundials.sh /Users/darrieythorsson/compHydro/data/SYMFLUENCE_data/installs/sundials/install

set -e

SUNDIALS_ROOT="${1:-}"

if [ -z "$SUNDIALS_ROOT" ]; then
    echo "Usage: $0 /path/to/sundials/install"
    echo ""
    echo "If SUNDIALS is not installed, build it first:"
    echo ""
    echo "  cd /path/to/sundials-7.4.0"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install \\"
    echo "           -DENABLE_MPI=OFF \\"
    echo "           -DENABLE_OPENMP=ON \\"
    echo "           -DBUILD_SHARED_LIBS=ON"
    echo "  make -j && make install"
    exit 1
fi

# Check for sundials headers - may be in $ROOT/include or $ROOT/sundials/include
SUNDIALS_INCLUDE=""
if [ -d "$SUNDIALS_ROOT/include/sundials" ]; then
    SUNDIALS_INCLUDE="$SUNDIALS_ROOT"
elif [ -d "$SUNDIALS_ROOT/sundials/include/sundials" ]; then
    SUNDIALS_INCLUDE="$SUNDIALS_ROOT/sundials"
fi

if [ -z "$SUNDIALS_INCLUDE" ]; then
    echo "Error: SUNDIALS not found at $SUNDIALS_ROOT"
    echo ""
    echo "Checked: $SUNDIALS_ROOT/include/sundials/"
    echo "Checked: $SUNDIALS_ROOT/sundials/include/sundials/"
    echo ""
    echo "If you have the source, build SUNDIALS first:"
    echo "  cd /path/to/sundials-7.4.0"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DCMAKE_INSTALL_PREFIX=$SUNDIALS_ROOT \\"
    echo "           -DENABLE_MPI=OFF \\"
    echo "           -DENABLE_OPENMP=ON \\"
    echo "           -DBUILD_SHARED_LIBS=ON"
    echo "  make -j && make install"
    exit 1
fi

echo "Found SUNDIALS at: $SUNDIALS_INCLUDE"
echo ""

# Create build directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring dFUSE with SUNDIALS..."
cmake .. \
    -DDFUSE_BUILD_PYTHON=ON \
    -DDFUSE_USE_NETCDF=ON \
    -DDFUSE_USE_SUNDIALS=ON \
    -DSUNDIALS_ROOT="$SUNDIALS_INCLUDE"

echo ""
echo "Building dFUSE..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

echo ""
echo "Copying Python module..."
cp dfuse_core*.so ../python/

echo ""
echo "Running tests..."
ctest --output-on-failure

echo ""
echo "Build complete!"
echo ""
echo "Python module location: $SCRIPT_DIR/python/dfuse_core*.so"
echo ""
echo "To use:"
echo "  cd $SCRIPT_DIR/python"
echo "  python -c \"import dfuse_core; print('dFUSE loaded successfully')\""
