//
// Created by Elias Fernandez on 14/05/2026.
//
#pragma once
#ifndef EGTTOOLS_ARPACKSTATUS_HPP
#define EGTTOOLS_ARPACKSTATUS_HPP

namespace egttools {
    inline bool is_arpack_enabled() {
#if defined(HAS_ARPACK)
        return true;
#else
        return false;
#endif
    }
}

#endif //EGTTOOLS_ARPACKSTATUS_HPP
