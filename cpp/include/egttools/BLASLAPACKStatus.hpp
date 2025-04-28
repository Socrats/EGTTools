//
// Created by Elias Fernandez on 28/04/2025.
//
#pragma once
#ifndef EGTTOOLS_BLASLAPACKSTATUS_HPP
#define EGTTOOLS_BLASLAPACKSTATUS_HPP

namespace egttools {
    /**
     * @brief Check if the library was compiled with BLAS/LAPACK acceleration.
     *
     * @return true if BLAS/LAPACK was enabled during compilation, false otherwise.
     */
    inline bool is_blas_lapack_enabled() {
#if defined(EIGEN_USE_BLAS)
        return true;
#else
        return false;
#endif
    }
}

#endif //EGTTOOLS_BLASLAPACKSTATUS_HPP
