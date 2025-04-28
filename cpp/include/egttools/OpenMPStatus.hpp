//
// Created by Elias Fernandez on 28/04/2025.
//
#pragma once
#ifndef EGTTOOLS_OPENMPSTATUS_HPP
#define EGTTOOLS_OPENMPSTATUS_HPP

namespace egttools {
    /**
     * @brief Check if the library was compiled with OpenMP support.
     *
     * @return true if OpenMP was enabled during compilation, false otherwise.
     */
    inline bool is_openmp_enabled() {
#if defined(_OPENMP)
        return true;
#else
        return false;
#endif
    }
}

#endif //EGTTOOLS_OPENMPSTATUS_HPP
