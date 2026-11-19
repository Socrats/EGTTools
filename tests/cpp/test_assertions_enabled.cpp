// Guard for the C++ test suite: the tests check results with assert(), which is
// compiled out when NDEBUG is defined. tests/CMakeLists.txt undefines NDEBUG for
// every test program; this test fails if that stops working, for example after a
// change to the build flags.
#include <cassert>
#include <iostream>

int main() {
#ifdef NDEBUG
    std::cerr << "NDEBUG is defined: assert() is compiled out and the C++ tests check nothing.\n";
    return 1;
#else
    bool evaluated = false;
    assert((evaluated = true));
    if (!evaluated) {
        std::cerr << "assert() did not evaluate its argument.\n";
        return 1;
    }
    std::cout << "Assertions are enabled.\n";
    return 0;
#endif
}
