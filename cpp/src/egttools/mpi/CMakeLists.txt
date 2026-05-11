# Static library with the PairwisePetscOperator implementation.
# Linked into numerical_mpi_ by the parent CMakeLists.

# Utils.cpp (fermi, sample_simplex, calculate_state) is compiled as an OBJECT
# library so that PETSc's compile flags do not contaminate it.  PETSc headers
# define macros that conflict with LDBL_MAX / assert when applied to vanilla
# C++ translation units.
add_library(egttools_finite_pop_utils_obj OBJECT
    ../finite_populations/Utils.cpp
)
target_include_directories(egttools_finite_pop_utils_obj PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp/include"
)
# Distributions.h conditionally includes Boost which brings in <cfloat>.
# Without HAS_BOOST, LDBL_MAX is undefined.  Mirror what numerical_ does.
target_link_libraries(egttools_finite_pop_utils_obj PRIVATE Boost::multiprecision)
target_compile_definitions(egttools_finite_pop_utils_obj PRIVATE HAS_BOOST)
set_target_properties(egttools_finite_pop_utils_obj PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    POSITION_INDEPENDENT_CODE TRUE
)

add_library(egttools_mpi_core STATIC
    PairwisePetscOperator.cpp
    $<TARGET_OBJECTS:egttools_finite_pop_utils_obj>
)

target_include_directories(egttools_mpi_core PUBLIC
    "${PROJECT_SOURCE_DIR}/cpp/include"
)

target_link_libraries(egttools_mpi_core PUBLIC
    ${EGTTOOLS_PETSC_TARGET}
    ${EGTTOOLS_SLEPC_TARGET}
    MPI::MPI_CXX
    Eigen3::Eigen
)

set_target_properties(egttools_mpi_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    POSITION_INDEPENDENT_CODE TRUE
    CXX_VISIBILITY_PRESET hidden
)
