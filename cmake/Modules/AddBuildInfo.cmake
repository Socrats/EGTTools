# Define variables for configure_file
if (USE_OPENMP)
    if (OpenMP_CXX_FOUND)
        set(OPENMP_STATUS "ON")
    else ()
        set(OPENMP_STATUS "OFF (requested but not found)")
    endif ()
else ()
    set(OPENMP_STATUS "OFF (disabled by user)")
endif ()

# Version
set(EGTTOOLS_VERSION "${PROJECT_VERSION}")

# Build date
string(TIMESTAMP CMAKE_BUILD_DATE "%Y-%m-%d")

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
