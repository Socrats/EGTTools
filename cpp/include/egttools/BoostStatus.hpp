//
// Created by Elias Fernandez on 14/05/2026.
//
#pragma once
#ifndef EGTTOOLS_BOOSTSTATUS_HPP
#define EGTTOOLS_BOOSTSTATUS_HPP

namespace egttools {
    // Boost.Multiprecision is a required dependency, so this is always true. Kept for the
    // Python API (egttools.numerical.numerical_.is_boost_enabled) and for compatibility.
    inline bool is_boost_enabled() {
        return true;
    }
}

#endif //EGTTOOLS_BOOSTSTATUS_HPP
