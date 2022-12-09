//
// Copied from https://stackoverflow.com/questions/54738011/pybind11-boostmultiprecisioncpp-int-to-python
//
#pragma once
#include <pybind11/pybind11.h>

#include <boost/multiprecision/cpp_int.hpp>
#include <sstream>

namespace pybind11::detail {
    using cpp_int = boost::multiprecision::cpp_int;
    using uint128_t = boost::multiprecision::uint128_t;
    namespace py = pybind11;

    template<>
    struct type_caster<cpp_int> {
        /**
         * This macro establishes the name 'cpp_int' in
         * function signatures and declares a local variable
         * 'value' of type cpp_int
         */
        PYBIND11_TYPE_CASTER(cpp_int, _("cpp_int"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a cpp_int
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            // Convert into base 16 string (PyNumber_ToBase prepend '0x')
            PyObject* tmp = PyNumber_ToBase(src.ptr(), 16);
            if (!tmp) return false;

            auto s = py::cast<std::string>(tmp);
            value = cpp_int{s};// explicit cast from string to cpp_int,
                               // don't need a base here because
                               // `PyNumber_ToBase` already prepended '0x'
            Py_DECREF(tmp);

            /* Ensure return code was OK (to avoid out-of-range errors etc) */
            return !PyErr_Occurred();
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an cpp_int instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(const cpp_int& src, return_value_policy, handle) {
            // Convert cpp_int to base 16 string
            std::ostringstream oss;
            oss << std::hex << src;
            return PyLong_FromString(oss.str().c_str(), nullptr, 16);
        }
    };

    template<>
    struct type_caster<uint128_t> {
        /**
         * This macro establishes the name 'cpp_int' in
         * function signatures and declares a local variable
         * 'value' of type cpp_int
         */
        PYBIND11_TYPE_CASTER(uint128_t, _("uint128_t"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a cpp_int
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            // Convert into base 16 string (PyNumber_ToBase prepend '0x')
            PyObject* tmp = PyNumber_ToBase(src.ptr(), 16);
            if (!tmp) return false;

            auto s = py::cast<std::string>(tmp);
            value = uint128_t{s};// explicit cast from string to cpp_int,
                                 // don't need a base here because
                                 // `PyNumber_ToBase` already prepended '0x'
            Py_DECREF(tmp);

            /* Ensure return code was OK (to avoid out-of-range errors etc) */
            return !PyErr_Occurred();
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an cpp_int instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(const uint128_t& src, return_value_policy, handle) {
            // Convert cpp_int to base 16 string
            std::ostringstream oss;
            oss << std::hex << src;
            return PyLong_FromString(oss.str().c_str(), nullptr, 16);
        }
    };
}// namespace pybind11::detail