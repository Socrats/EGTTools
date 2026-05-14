//
// Created by Elias Fernandez on 14/05/2026.
//
#pragma once
#ifndef EGTTOOLS_PETSCSTATUS_HPP
#define EGTTOOLS_PETSCSTATUS_HPP

namespace egttools {
    inline bool is_petsc_enabled() {
#if defined(HAS_PETSC_MODULE)
        return true;
#else
        return false;
#endif
    }
}

#endif //EGTTOOLS_PETSCSTATUS_HPP
