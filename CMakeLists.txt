cmake_minimum_required(VERSION 3.18)
project(dfuse LANGUAGES CXX)

# Enable CUDA only if requested and available
option(DFUSE_USE_CUDA "Enable CUDA support" OFF)
if(DFUSE_USE_CUDA)
    enable_language(CUDA)
endif()

# ============================================================================
# OPTIONS
# ============================================================================

option(DFUSE_USE_DOUBLE "Use double precision" OFF)
option(DFUSE_USE_ENZYME "Enable Enzyme AD" OFF)
option(DFUSE_USE_SUNDIALS "Enable SUNDIALS ODE solvers" OFF)
option(DFUSE_USE_NETCDF "Enable NetCDF I/O" OFF)
option(DFUSE_BUILD_PYTHON "Build Python bindings" OFF)
option(DFUSE_BUILD_TESTS "Build tests" ON)
option(DFUSE_BUILD_EXAMPLES "Build examples" ON)

# ============================================================================
# COMPILER SETTINGS
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DFUSE_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# ============================================================================
# DEPENDENCIES
# ============================================================================

# Find CUDA if enabled
if(DFUSE_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Find pybind11 for Python bindings
if(DFUSE_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Try to find pybind11 via cmake config
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # Try to find via Python module
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE pybind11_result
        )
        if(pybind11_result EQUAL 0 AND EXISTS "${pybind11_DIR}")
            find_package(pybind11 CONFIG QUIET)
        endif()
    endif()
    
    if(NOT pybind11_FOUND)
        message(WARNING "")
        message(WARNING "==========================================================")
        message(WARNING "pybind11 not found! Python bindings will be DISABLED.")
        message(WARNING "")
        message(WARNING "To enable Python bindings, run:")
        message(WARNING "  pip install pybind11")
        message(WARNING "")
        message(WARNING "Then delete build/ and re-run cmake.")
        message(WARNING "==========================================================")
        message(WARNING "")
        set(DFUSE_BUILD_PYTHON OFF)
    else()
        message(STATUS "Found pybind11: ${pybind11_DIR}")
    endif()
endif()

# Enzyme for automatic differentiation
if(DFUSE_USE_ENZYME)
    # Enzyme is a compiler PLUGIN, not a linked library
    # It transforms LLVM IR at compile time to generate derivatives
    
    # Try to find Enzyme plugin
    # Check common locations
    set(ENZYME_SEARCH_PATHS
        "/opt/homebrew/lib"
        "/opt/homebrew/Cellar/enzyme"
        "/usr/local/lib"
        "/usr/lib"
        "${CMAKE_PREFIX_PATH}/lib"
    )
    
    # Find the ClangEnzyme plugin
    find_file(ENZYME_PLUGIN
        NAMES 
            ClangEnzyme-21.dylib
            ClangEnzyme-20.dylib
            ClangEnzyme-19.dylib
            ClangEnzyme-18.dylib
            ClangEnzyme-17.dylib
            ClangEnzyme.dylib
            LLDEnzyme-21.dylib
            LLVMEnzyme-21.dylib
        PATHS ${ENZYME_SEARCH_PATHS}
        PATH_SUFFIXES enzyme
        NO_DEFAULT_PATH
    )
    
    # Also try globbing for any version
    if(NOT ENZYME_PLUGIN)
        file(GLOB ENZYME_PLUGIN_GLOB 
            "/opt/homebrew/lib/ClangEnzyme*.dylib"
            "/opt/homebrew/Cellar/enzyme/*/lib/ClangEnzyme*.dylib"
        )
        if(ENZYME_PLUGIN_GLOB)
            list(GET ENZYME_PLUGIN_GLOB 0 ENZYME_PLUGIN)
        endif()
    endif()
    
    if(ENZYME_PLUGIN)
        message(STATUS "Found Enzyme plugin: ${ENZYME_PLUGIN}")
        
        # Enzyme is used as a compiler plugin via -fplugin flag
        # It's NOT a linked library - don't add to target_link_libraries!
        set(ENZYME_COMPILE_FLAGS "-fplugin=${ENZYME_PLUGIN}")
        set(DFUSE_ENZYME_AVAILABLE TRUE)
        
        # Add to global CXX flags for Enzyme-enabled builds
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ENZYME_COMPILE_FLAGS}")
        
    else()
        message(WARNING "")
        message(WARNING "==========================================================")
        message(WARNING "Enzyme plugin not found!")
        message(WARNING "")
        message(WARNING "Enzyme is installed but plugin not found in search paths.")
        message(WARNING "")
        message(WARNING "Try specifying the plugin path manually:")
        message(WARNING "  cmake .. -DENZYME_PLUGIN=/path/to/ClangEnzyme-XX.dylib")
        message(WARNING "")
        message(WARNING "Or check your Enzyme installation:")
        message(WARNING "  ls /opt/homebrew/Cellar/enzyme/*/lib/")
        message(WARNING "==========================================================")
        message(WARNING "")
        set(DFUSE_USE_ENZYME OFF)
        set(DFUSE_ENZYME_AVAILABLE FALSE)
    endif()
    
    # CUDA + Enzyme requires Clang as CUDA compiler
    if(DFUSE_USE_CUDA AND DFUSE_ENZYME_AVAILABLE)
        message(STATUS "")
        message(STATUS "==========================================================")
        message(STATUS "CUDA + Enzyme Configuration")
        message(STATUS "")
        message(STATUS "Enzyme AD on CUDA requires Clang as the CUDA compiler.")
        message(STATUS "To enable this, configure with:")
        message(STATUS "  cmake .. -DCMAKE_CUDA_COMPILER=clang++ \\")
        message(STATUS "           -DDFUSE_USE_CUDA=ON \\")
        message(STATUS "           -DDFUSE_USE_ENZYME=ON")
        message(STATUS "==========================================================")
        message(STATUS "")
        
        if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            message(STATUS "Using Clang for CUDA - Enzyme AD on GPU enabled")
            
            if(NOT DEFINED CMAKE_CUDA_TOOLKIT_ROOT)
                set(CMAKE_CUDA_TOOLKIT_ROOT "/usr/local/cuda")
            endif()
            
            set(CLANG_CUDA_FLAGS 
                "--cuda-path=${CMAKE_CUDA_TOOLKIT_ROOT}"
                "-x" "cuda"
            )
            
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
                set(CMAKE_CUDA_ARCHITECTURES "70;80")
            endif()
            
            foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
                list(APPEND CLANG_CUDA_FLAGS "--cuda-gpu-arch=sm_${arch}")
            endforeach()
            
            list(APPEND CLANG_CUDA_FLAGS "${ENZYME_COMPILE_FLAGS}")
            
            string(REPLACE ";" " " CLANG_CUDA_FLAGS_STR "${CLANG_CUDA_FLAGS}")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CLANG_CUDA_FLAGS_STR}")
        else()
            message(WARNING "Using NVCC for CUDA - Enzyme AD only on CPU code")
        endif()
    endif()
endif()

# SUNDIALS for robust ODE solving
if(DFUSE_USE_SUNDIALS)
    # Allow specifying custom SUNDIALS installation path
    # Use: cmake .. -DSUNDIALS_ROOT=/path/to/sundials/install
    if(NOT DEFINED SUNDIALS_ROOT AND DEFINED ENV{SUNDIALS_ROOT})
        set(SUNDIALS_ROOT $ENV{SUNDIALS_ROOT})
    endif()
    
    if(SUNDIALS_ROOT)
        message(STATUS "Looking for SUNDIALS in: ${SUNDIALS_ROOT}")
        # Force use of specified path, skip system paths
        set(SUNDIALS_DIR ${SUNDIALS_ROOT}/lib/cmake/sundials CACHE PATH "SUNDIALS CMake config" FORCE)
        # Clear system paths to avoid finding Homebrew version
        set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)
        find_package(SUNDIALS QUIET COMPONENTS cvode cvodes PATHS ${SUNDIALS_ROOT} NO_DEFAULT_PATH)
        set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH ON)
    else()
        # No custom path - try system but warn about MPI
        find_package(MPI COMPONENTS C CXX QUIET)
        if(MPI_FOUND)
            message(STATUS "Found MPI C: ${MPI_C_INCLUDE_DIRS}")
            if(NOT TARGET MPI::MPI_C AND MPI_C_FOUND)
                add_library(MPI::MPI_C IMPORTED INTERFACE)
                set_target_properties(MPI::MPI_C PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${MPI_C_INCLUDE_DIRS}"
                    INTERFACE_LINK_LIBRARIES "${MPI_C_LIBRARIES}"
                )
            endif()
        endif()
        find_package(SUNDIALS QUIET COMPONENTS cvode cvodes)
    endif()
    
    if(SUNDIALS_FOUND)
        message(STATUS "Found SUNDIALS: ${SUNDIALS_DIR}")
        message(STATUS "SUNDIALS version: ${SUNDIALS_VERSION}")
        
        # Debug: List available SUNDIALS targets
        get_property(_sundials_targets DIRECTORY PROPERTY IMPORTED_TARGETS)
        message(STATUS "Available SUNDIALS targets: ${_sundials_targets}")
        
        # Check for CVODES - try multiple possible target names
        set(SUNDIALS_CVODES_FOUND FALSE)
        if(TARGET SUNDIALS::cvodes)
            message(STATUS "Found CVODES target: SUNDIALS::cvodes")
            set(SUNDIALS_CVODES_FOUND TRUE)
            set(SUNDIALS_CVODES_TARGET SUNDIALS::cvodes)
        elseif(TARGET SUNDIALS::sundials_cvodes)
            message(STATUS "Found CVODES target: SUNDIALS::sundials_cvodes")
            set(SUNDIALS_CVODES_FOUND TRUE)
            set(SUNDIALS_CVODES_TARGET SUNDIALS::sundials_cvodes)
        else()
            # Manual check - look for the library file
            find_library(SUNDIALS_CVODES_LIB 
                NAMES sundials_cvodes
                HINTS ${SUNDIALS_ROOT}/lib
                NO_DEFAULT_PATH
            )
            if(SUNDIALS_CVODES_LIB)
                message(STATUS "Found CVODES library manually: ${SUNDIALS_CVODES_LIB}")
                # Create an imported target for cvodes
                add_library(SUNDIALS::cvodes_manual UNKNOWN IMPORTED)
                set_target_properties(SUNDIALS::cvodes_manual PROPERTIES
                    IMPORTED_LOCATION "${SUNDIALS_CVODES_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "${SUNDIALS_ROOT}/include"
                )
                set(SUNDIALS_CVODES_FOUND TRUE)
                set(SUNDIALS_CVODES_TARGET SUNDIALS::cvodes_manual)
            else()
                message(STATUS "CVODES not found - adjoint sensitivity will use Enzyme only")
            endif()
        endif()
    else()
        # Manual fallback: look for headers and libraries directly
        message(STATUS "SUNDIALS CMake config not found, trying manual detection...")
        
        find_path(SUNDIALS_INCLUDE_DIR sundials/sundials_types.h
            HINTS
                ${SUNDIALS_ROOT}/include
                /opt/homebrew/include
                /usr/local/include
                $ENV{HOME}/sundials/include
        )
        
        find_library(SUNDIALS_CVODE_LIB 
            NAMES sundials_cvode
            HINTS
                ${SUNDIALS_ROOT}/lib
                /opt/homebrew/lib
                /usr/local/lib
                $ENV{HOME}/sundials/lib
        )
        
        # CVODES - for adjoint sensitivity analysis
        find_library(SUNDIALS_CVODES_LIB 
            NAMES sundials_cvodes
            HINTS
                ${SUNDIALS_ROOT}/lib
                /opt/homebrew/lib
                /usr/local/lib
                $ENV{HOME}/sundials/lib
        )
        
        find_library(SUNDIALS_NVECSERIAL_LIB 
            NAMES sundials_nvecserial
            HINTS
                ${SUNDIALS_ROOT}/lib
                /opt/homebrew/lib
                /usr/local/lib
        )
        
        # SUNLinsol and SUNMatrix for linear solvers
        find_library(SUNDIALS_SUNLINSOL_DENSE_LIB 
            NAMES sundials_sunlinsoldense
            HINTS
                ${SUNDIALS_ROOT}/lib
                /opt/homebrew/lib
                /usr/local/lib
        )
        
        find_library(SUNDIALS_SUNMATRIX_DENSE_LIB 
            NAMES sundials_sunmatrixdense
            HINTS
                ${SUNDIALS_ROOT}/lib
                /opt/homebrew/lib
                /usr/local/lib
        )
        
        find_library(SUNDIALS_CORE_LIB 
            NAMES sundials_core sundials_generic
            HINTS
                ${SUNDIALS_ROOT}/lib
                /opt/homebrew/lib
                /usr/local/lib
        )
        
        if(SUNDIALS_INCLUDE_DIR AND SUNDIALS_CVODE_LIB AND SUNDIALS_NVECSERIAL_LIB)
            message(STATUS "Found SUNDIALS (manual): ${SUNDIALS_INCLUDE_DIR}")
            set(SUNDIALS_FOUND TRUE)
            
            # Create imported targets manually
            if(NOT TARGET SUNDIALS::cvode)
                add_library(SUNDIALS::cvode UNKNOWN IMPORTED)
                set_target_properties(SUNDIALS::cvode PROPERTIES
                    IMPORTED_LOCATION "${SUNDIALS_CVODE_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "${SUNDIALS_INCLUDE_DIR}"
                )
                if(SUNDIALS_NVECSERIAL_LIB)
                    target_link_libraries(SUNDIALS::cvode INTERFACE ${SUNDIALS_NVECSERIAL_LIB})
                endif()
                if(SUNDIALS_SUNLINSOL_DENSE_LIB)
                    target_link_libraries(SUNDIALS::cvode INTERFACE ${SUNDIALS_SUNLINSOL_DENSE_LIB})
                endif()
                if(SUNDIALS_SUNMATRIX_DENSE_LIB)
                    target_link_libraries(SUNDIALS::cvode INTERFACE ${SUNDIALS_SUNMATRIX_DENSE_LIB})
                endif()
                if(SUNDIALS_CORE_LIB)
                    target_link_libraries(SUNDIALS::cvode INTERFACE ${SUNDIALS_CORE_LIB})
                endif()
            endif()
            
            # Create CVODES target for adjoint sensitivity
            if(SUNDIALS_CVODES_LIB AND NOT TARGET SUNDIALS::cvodes)
                add_library(SUNDIALS::cvodes UNKNOWN IMPORTED)
                set_target_properties(SUNDIALS::cvodes PROPERTIES
                    IMPORTED_LOCATION "${SUNDIALS_CVODES_LIB}"
                    INTERFACE_INCLUDE_DIRECTORIES "${SUNDIALS_INCLUDE_DIR}"
                )
                if(SUNDIALS_NVECSERIAL_LIB)
                    target_link_libraries(SUNDIALS::cvodes INTERFACE ${SUNDIALS_NVECSERIAL_LIB})
                endif()
                if(SUNDIALS_SUNLINSOL_DENSE_LIB)
                    target_link_libraries(SUNDIALS::cvodes INTERFACE ${SUNDIALS_SUNLINSOL_DENSE_LIB})
                endif()
                if(SUNDIALS_SUNMATRIX_DENSE_LIB)
                    target_link_libraries(SUNDIALS::cvodes INTERFACE ${SUNDIALS_SUNMATRIX_DENSE_LIB})
                endif()
                if(SUNDIALS_CORE_LIB)
                    target_link_libraries(SUNDIALS::cvodes INTERFACE ${SUNDIALS_CORE_LIB})
                endif()
                message(STATUS "Found CVODES for adjoint sensitivity: ${SUNDIALS_CVODES_LIB}")
                set(SUNDIALS_CVODES_FOUND TRUE)
            else()
                message(STATUS "CVODES not found - adjoint sensitivity will be disabled")
                set(SUNDIALS_CVODES_FOUND FALSE)
            endif()
        endif()
    endif()
    
    if(NOT SUNDIALS_FOUND)
        message(WARNING "")
        message(WARNING "==========================================================")
        message(WARNING "SUNDIALS not found. SUNDIALS support will be DISABLED.")
        message(WARNING "")
        message(WARNING "To use a custom SUNDIALS build, specify:")
        message(WARNING "  cmake .. -DSUNDIALS_ROOT=/path/to/sundials/install")
        message(WARNING "")
        message(WARNING "The built-in Euler solvers will be used instead.")
        message(WARNING "==========================================================")
        message(WARNING "")
        set(DFUSE_USE_SUNDIALS OFF)
    endif()
endif()

# NetCDF for standard I/O
if(DFUSE_USE_NETCDF)
    # First try CMake config
    find_package(netCDFCxx QUIET CONFIG)
    
    if(netCDFCxx_FOUND)
        message(STATUS "Found netCDF-C++ via CMake config: ${netCDFCxx_DIR}")
        set(NETCDF_CXX_FOUND TRUE)
        set(NETCDF_CXX_TARGET netCDF::netcdf-cxx4)
    else()
        # Try to find via pkg-config or manual search (Homebrew)
        find_path(NETCDF_CXX_INCLUDE_DIR ncFile.h
            HINTS
                /opt/homebrew/include
                /usr/local/include
                /usr/include
            PATH_SUFFIXES netcdf-cxx4
        )
        
        find_library(NETCDF_CXX_LIBRARY
            NAMES netcdf-cxx4 netcdf_c++4
            HINTS
                /opt/homebrew/lib
                /usr/local/lib
                /usr/lib
        )
        
        if(NETCDF_CXX_INCLUDE_DIR AND NETCDF_CXX_LIBRARY)
            message(STATUS "Found netCDF-C++ manually:")
            message(STATUS "  Include: ${NETCDF_CXX_INCLUDE_DIR}")
            message(STATUS "  Library: ${NETCDF_CXX_LIBRARY}")
            set(NETCDF_CXX_FOUND TRUE)
            
            # Create imported target
            add_library(netcdf_cxx4 UNKNOWN IMPORTED)
            set_target_properties(netcdf_cxx4 PROPERTIES
                IMPORTED_LOCATION "${NETCDF_CXX_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_CXX_INCLUDE_DIR}"
            )
            set(NETCDF_CXX_TARGET netcdf_cxx4)
            
            # Also need NetCDF-C
            find_package(netCDF QUIET)
            if(netCDF_FOUND)
                set_target_properties(netcdf_cxx4 PROPERTIES
                    INTERFACE_LINK_LIBRARIES netCDF::netcdf
                )
            endif()
        else()
            message(STATUS "netCDF-C++ not found")
            set(NETCDF_CXX_FOUND FALSE)
        endif()
    endif()
    
    # Fall back to C library for Python bindings
    if(NOT NETCDF_CXX_FOUND)
        find_package(netCDF QUIET)
        if(netCDF_FOUND)
            message(STATUS "Found netCDF-C: ${netCDF_DIR}")
            message(STATUS "C++ CLI will not be built (requires netcdf-cxx4)")
        else()
            message(WARNING "NetCDF not found. Install with: brew install netcdf netcdf-cxx")
            set(DFUSE_USE_NETCDF OFF)
        endif()
    endif()
endif()

# ============================================================================
# HEADER-ONLY LIBRARY
# ============================================================================

add_library(dfuse INTERFACE)
target_include_directories(dfuse INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compile definitions
if(DFUSE_USE_DOUBLE)
    target_compile_definitions(dfuse INTERFACE DFUSE_USE_DOUBLE)
endif()

if(DFUSE_USE_CUDA)
    target_compile_definitions(dfuse INTERFACE DFUSE_USE_CUDA)
    target_link_libraries(dfuse INTERFACE CUDA::cudart)
endif()

if(DFUSE_USE_ENZYME)
    target_compile_definitions(dfuse INTERFACE DFUSE_USE_ENZYME)
    # Enzyme is a compiler plugin, flags already added to CMAKE_CXX_FLAGS
    # Do NOT link - it's not a library!
endif()

if(DFUSE_USE_SUNDIALS)
    target_compile_definitions(dfuse INTERFACE DFUSE_USE_SUNDIALS)
    target_link_libraries(dfuse INTERFACE SUNDIALS::cvode)
    
    # Link CVODES for adjoint sensitivity if available
    if(SUNDIALS_CVODES_FOUND AND SUNDIALS_CVODES_TARGET)
        message(STATUS "Linking CVODES: ${SUNDIALS_CVODES_TARGET}")
        target_compile_definitions(dfuse INTERFACE DFUSE_USE_CVODES)
        target_link_libraries(dfuse INTERFACE ${SUNDIALS_CVODES_TARGET})
    endif()
    
    # SUNDIALS on some systems (Homebrew) requires MPI headers
    if(MPI_FOUND)
        target_include_directories(dfuse INTERFACE ${MPI_CXX_INCLUDE_DIRS})
        if(TARGET MPI::MPI_CXX)
            target_link_libraries(dfuse INTERFACE MPI::MPI_CXX)
        endif()
        if(TARGET MPI::MPI_C)
            target_link_libraries(dfuse INTERFACE MPI::MPI_C)
        endif()
    endif()
endif()

if(DFUSE_USE_NETCDF)
    target_compile_definitions(dfuse INTERFACE DFUSE_USE_NETCDF)
    if(NETCDF_CXX_FOUND AND NETCDF_CXX_TARGET)
        target_link_libraries(dfuse INTERFACE ${NETCDF_CXX_TARGET})
    elseif(netCDF_FOUND)
        target_link_libraries(dfuse INTERFACE netCDF::netcdf)
    endif()
endif()

# ============================================================================
# PYTHON BINDINGS
# ============================================================================

if(DFUSE_BUILD_PYTHON)
    pybind11_add_module(dfuse_core python/bindings.cpp)
    target_link_libraries(dfuse_core PRIVATE dfuse)
    
    if(DFUSE_USE_CUDA)
        target_compile_definitions(dfuse_core PRIVATE DFUSE_USE_CUDA)
    endif()
    
    # OpenMP for parallel batch execution
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(dfuse_core PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Install Python module
    install(TARGETS dfuse_core
        LIBRARY DESTINATION ${Python3_SITELIB}
    )
    
    # Install Python source
    install(FILES python/dfuse.py
        DESTINATION ${Python3_SITELIB}
    )
endif()

# ============================================================================
# TESTS
# ============================================================================

if(DFUSE_BUILD_TESTS)
    enable_testing()
    
    # Comprehensive test (includes all validation)
    add_executable(test_comprehensive tests/test_comprehensive.cpp)
    target_link_libraries(test_comprehensive PRIVATE dfuse)
    add_test(NAME test_comprehensive COMMAND test_comprehensive)
endif()

# ============================================================================
# EXAMPLES
# ============================================================================

if(DFUSE_BUILD_EXAMPLES)
    # Single basin example
    add_executable(example_single examples/single_basin.cpp)
    target_link_libraries(example_single PRIVATE dfuse)
endif()

# ============================================================================
# CLI EXECUTABLE
# ============================================================================

# Build CLI if NetCDF C++ is available
if(DFUSE_USE_NETCDF AND NETCDF_CXX_FOUND)
    add_executable(dfuse_exe src/dfuse_cli.cpp)
    target_link_libraries(dfuse_exe PRIVATE dfuse)
    set_target_properties(dfuse_exe PROPERTIES OUTPUT_NAME "dfuse")
    
    # OpenMP for parallel HRU execution in distributed mode
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(dfuse_exe PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "CLI OpenMP support: enabled")
    else()
        message(STATUS "CLI OpenMP support: disabled (OpenMP not found)")
    endif()
    
    # Install CLI
    install(TARGETS dfuse_exe
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    message(STATUS "Building CLI: dfuse")
elseif(DFUSE_USE_NETCDF)
    message(STATUS "CLI not built: requires netcdf-cxx4 library")
endif()

# ============================================================================
# INSTALLATION
# ============================================================================

include(GNUInstallDirs)

install(TARGETS dfuse
    EXPORT dfuse-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/dfuse
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT dfuse-targets
    FILE dfuse-config.cmake
    NAMESPACE dfuse::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dfuse
)

# ============================================================================
# PACKAGE CONFIG
# ============================================================================

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dfuse-config-version.cmake"
    VERSION 0.3.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dfuse-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dfuse
)

# ============================================================================
# SUMMARY
# ============================================================================

message(STATUS "")
message(STATUS "dFUSE Configuration:")
message(STATUS "  Version:        0.3.0")
message(STATUS "  Precision:      ${DFUSE_USE_DOUBLE}")
message(STATUS "  CUDA:           ${DFUSE_USE_CUDA}")
message(STATUS "  Enzyme AD:      ${DFUSE_USE_ENZYME}")
message(STATUS "  SUNDIALS:       ${DFUSE_USE_SUNDIALS}")
message(STATUS "  NetCDF:         ${DFUSE_USE_NETCDF}")
message(STATUS "  Python:         ${DFUSE_BUILD_PYTHON}")
message(STATUS "  Tests:          ${DFUSE_BUILD_TESTS}")
message(STATUS "  Examples:       ${DFUSE_BUILD_EXAMPLES}")
message(STATUS "")