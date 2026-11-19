// Canary for sanitizer builds (EGTTOOLS_SANITIZE): each mode commits one deliberate
// error that the corresponding sanitizer must report. CTest runs one test per enabled
// sanitizer and passes only if the sanitizer's report appears in the output, which
// shows that the build is instrumented. Registered only in sanitizer builds.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

namespace {
    // volatile keeps the compiler from proving the error away at compile time.
    volatile int sink = 0;

    int heap_overflow() {
        int *values = new int[4];
        volatile int index = 4;
        sink = values[index];// one past the end
        delete[] values;
        return 0;
    }

    int signed_overflow() {
        volatile int value = std::numeric_limits<int>::max();
        sink = value + 1;
        return 0;
    }

    int data_race() {
        int shared = 0;
        auto increment = [&shared] {
            for (int i = 0; i < 100000; ++i) ++shared;
        };
        std::thread first(increment), second(increment);
        first.join();
        second.join();
        sink = shared;
        return 0;
    }
}// namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: testSanitizerCanary address|undefined|thread\n";
        return 2;
    }
    if (std::strcmp(argv[1], "address") == 0) return heap_overflow();
    if (std::strcmp(argv[1], "undefined") == 0) return signed_overflow();
    if (std::strcmp(argv[1], "thread") == 0) return data_race();
    std::cerr << "unknown mode " << argv[1] << '\n';
    return 2;
}
