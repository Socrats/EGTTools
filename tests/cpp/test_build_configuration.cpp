// Checks on how the test program itself is built.
//
// tests/CMakeLists.txt undefines NDEBUG for the test program, so the assert()s in the library
// headers it includes (Group, LruCache, finite_populations/Utils.hpp) are checked in every build
// type. This test fails if that stops working, for example after a change to the build flags.
#include <cassert>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("assert() is enabled in the test program", "[build]") {
#ifdef NDEBUG
    FAIL("NDEBUG is defined: assert() is compiled out in the library headers the tests include.");
#else
    bool evaluated = false;
    assert((evaluated = true));
    CHECK(evaluated);
#endif
}
