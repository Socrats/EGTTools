//
// Created by Elias Fernandez on 21/11/2022.
//
#pragma once
#ifndef EGTTOOLS_MATH_HPP
#define EGTTOOLS_MATH_HPP

#if (HAS_BOOST)
#include <boost/multiprecision/cpp_int.hpp>
#endif

namespace egttools::math {

#if (HAS_BOOST)
    using cpp_int = boost::multiprecision::cpp_int;
#endif

#if (HAS_BOOST)

    template<typename IntType>
    cpp_int factorial(IntType number) {
        cpp_int num = 1;
        for (IntType i = 1; i <= number; ++i) {
            num = num * i;
        }
        return num;
    }
#else
    constexpr size_t MAX_FACTORIAL = 170;

    template<typename IntType>
    IntType factorial(IntType number) {
        IntType num = 1;
        for (IntType i = 1; i <= number; ++i) {
            num = num * i;
        }
        return num;
    }
#endif
}// namespace egttools::math

#endif//EGTTOOLS_MATH_HPP
