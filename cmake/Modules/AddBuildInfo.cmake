# Detect OpenMP
if (USE_OPENMP)
    if (OpenMP_CXX_FOUND)
        set(OPENMP_STATUS "ON")
    else ()
        set(OPENMP_STATUS "OFF (requested but not found)")
    endif ()
else ()
    set(OPENMP_STATUS "OFF (disabled by user)")
endif ()

# Detect BLAS/LAPACK acceleration based on compile definitions
get_target_property(_numerical_compile_defs numerical_ COMPILE_DEFINITIONS)

if (_numerical_compile_defs)
    list(FIND _numerical_compile_defs "EIGEN_USE_BLAS" _found_blas_macro)
    if (NOT _found_blas_macro EQUAL -1)
        set(BLAS_LAPACK_STATUS "ON")
    else()
        set(BLAS_LAPACK_STATUS "OFF")
    endif()
else()
    set(BLAS_LAPACK_STATUS "OFF")
endif()

# Version
set(EGTTOOLS_VERSION "${PROJECT_VERSION}")

# Build date
string(TIMESTAMP CMAKE_BUILD_DATE "%Y-%m-%d")

# macOS deployment target info
if(APPLE)
    if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
        set(MACOS_DEPLOYMENT_TARGET_LINE "macOS Deployment Target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    else()
        set(MACOS_DEPLOYMENT_TARGET_LINE "macOS Deployment Target: (not set)")
    endif()
else()
    set(MACOS_DEPLOYMENT_TARGET_LINE "")
endif()


# Build info generation
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/egttools_build_info.in
        ${CMAKE_CURRENT_BINARY_DIR}/egttools_build_info.txt
        @ONLY
)

install(
        FILES
        ${CMAKE_CURRENT_BINARY_DIR}/egttools_build_info.txt
        DESTINATION ${CMAKE_INSTALL_PREFIX}
)
