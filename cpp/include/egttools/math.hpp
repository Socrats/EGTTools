//
// Created by Elias Fernandez on 21/11/2022.
//
#pragma once
#ifndef EGTTOOLS_MATH_HPP
#define EGTTOOLS_MATH_HPP

#include <boost/multiprecision/cpp_int.hpp>

namespace egttools::math {

    using cpp_int = boost::multiprecision::cpp_int;


    template<typename IntType>
    cpp_int factorial(IntType number) {
        cpp_int num = 1;
        for (IntType i = 1; i <= number; ++i) {
            num = num * i;
        }
        return num;
    }
}// namespace egttools::math

#endif//EGTTOOLS_MATH_HPP
