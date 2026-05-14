//
// Created by Elias Fernandez on 14/05/2026.
//
#pragma once
#ifndef EGTTOOLS_BOOSTSTATUS_HPP
#define EGTTOOLS_BOOSTSTATUS_HPP

namespace egttools {
    inline bool is_boost_enabled() {
#if defined(HAS_BOOST)
        return true;
#else
        return false;
#endif
    }
}

#endif //EGTTOOLS_BOOSTSTATUS_HPP
