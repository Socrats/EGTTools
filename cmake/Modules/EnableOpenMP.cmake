# cmake/EnableOpenMP.cmake
# ------------------------------------------------------------
# Enables OpenMP support across platforms (Linux, macOS, HPC)
# ------------------------------------------------------------

include_guard(GLOBAL)

option(USE_OPENMP "Enable OpenMP multithreading" ON)

if (USE_OPENMP)

    # Only relevant on macOS
    if (APPLE)
        # Define a CMake cache variable so users can override it with -DLIBOMP_DIR=/path
        set(LIBOMP_DIR "/opt/homebrew/opt/libomp" CACHE PATH "Path to libomp installation on macOS")

        if (NOT EXISTS "${LIBOMP_DIR}")
            message(WARNING "[OpenMP] LIBOMP_DIR does not exist: ${LIBOMP_DIR}")
        else ()
            message(STATUS "[OpenMP] Using LIBOMP_DIR=${LIBOMP_DIR}")
        endif ()

        # Set OpenMP variables manually
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_DIR}/include")
        set(OpenMP_CXX_INCLUDE_DIRS "${LIBOMP_DIR}/include")
        set(OpenMP_omp_LIBRARY "${LIBOMP_DIR}/lib/libomp.dylib")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_C_LIB_NAMES "omp")

        set(CMAKE_INSTALL_RPATH "${LIBOMP_DIR}/lib")
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    endif ()


    find_package(OpenMP)

    if (OpenMP_CXX_FOUND)
        message(STATUS "[OpenMP] Found and enabled")
        message(STATUS "[OpenMP] Libraries: ${OpenMP_CXX_LIBRARIES}")
    else ()
        message(WARNING "[OpenMP] Not found — OpenMP support disabled")
    endif ()

    # Windows system runtime installation
    if (WIN32)
        if (OpenMP_CXX_FOUND)
            set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)
        endif ()

        include(InstallRequiredSystemLibraries)
    endif ()

endif ()

# Summary message for OpenMP status
if (USE_OPENMP)
    if (OpenMP_CXX_FOUND)
        message(STATUS "[EGTtools] OpenMP support: ON")
    else ()
        message(STATUS "[EGTtools] OpenMP support: OFF (requested but not found)")
    endif ()
else ()
    message(STATUS "[EGTtools] OpenMP support: OFF (disabled by user)")
endif ()

