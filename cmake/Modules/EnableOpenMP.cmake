# cmake/EnableOpenMP.cmake
# ------------------------------------------------------------
# Enables OpenMP support across platforms (Linux, macOS, HPC)
# ------------------------------------------------------------

include_guard(GLOBAL)

option(USE_OPENMP "Enable OpenMP multithreading" ON)

if (USE_OPENMP)

    if (APPLE)
        # macOS needs manual override with Homebrew/Conda libomp
        if (DEFINED ENV{LIBOMP_DIR})
            set(LIBOMP_DIR $ENV{LIBOMP_DIR})
        elseif (DEFINED LIBOMP_DIR)
            set(LIBOMP_DIR ${LIBOMP_DIR})
        else ()
            message(WARNING "[OpenMP] LIBOMP_DIR is not set. Defaulting to /opt/homebrew/opt/libomp")
            set(LIBOMP_DIR "/opt/homebrew/opt/libomp")
        endif ()

        # Set up OpenMP variables used by find_package(OpenMP)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_DIR}/include")
        set(OpenMP_CXX_INCLUDE_DIRS "${LIBOMP_DIR}/include")
        set(OpenMP_omp_LIBRARY "${LIBOMP_DIR}/lib/libomp.dylib")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_C_LIB_NAMES "omp")

        # Let CMake handle RPATH and linking cleanly
        set(CMAKE_INSTALL_RPATH "${LIBOMP_DIR}/lib")
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    endif ()


    find_package(OpenMP REQUIRED)

    if (OpenMP_CXX_FOUND)
        message(STATUS "[OpenMP] Found and enabled")
        message(STATUS "[OpenMP] Libraries: ${OpenMP_CXX_LIBRARIES}")
    else ()
        message(WARNING "[OpenMP] Not found — OpenMP support disabled")
    endif ()

endif ()
