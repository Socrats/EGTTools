# cmake/Modules/EnableSanitizers.cmake
# ------------------------------------------------------------
# Builds every target defined after this point with the sanitizers listed in
# EGTTOOLS_SANITIZE, to detect memory errors, undefined behaviour and data races.
# For development and CI only: sanitized binaries are slower and must not be shipped.
#
#   -DEGTTOOLS_SANITIZE="address;undefined"   AddressSanitizer + UndefinedBehaviorSanitizer
#   -DEGTTOOLS_SANITIZE=thread                ThreadSanitizer (not combinable with address/leak)
#   -DEGTTOOLS_SANITIZE=leak                  LeakSanitizer alone (Linux)
#
# GCC and Clang support all of them; MSVC supports only "address". Undefined behaviour
# aborts the program (-fno-sanitize-recover) so that a test that triggers it fails.
# Link-time optimisation is turned off: it interferes with the instrumentation and makes
# reports harder to read. Use CMAKE_BUILD_TYPE=RelWithDebInfo or Debug for readable stacks.
#
# The tests in tests/CMakeLists.txt add testSanitizerCanary, which triggers each enabled
# sanitizer on purpose and passes only if the sanitizer reports it.
# ------------------------------------------------------------

include_guard(GLOBAL)

set(EGTTOOLS_SANITIZE "" CACHE STRING
        "Sanitizers to build with, separated by ';': address, undefined, thread, leak")

if (NOT EGTTOOLS_SANITIZE)
    return()
endif ()

set(_egttools_known_sanitizers address undefined thread leak)
foreach (_sanitizer IN LISTS EGTTOOLS_SANITIZE)
    if (NOT _sanitizer IN_LIST _egttools_known_sanitizers)
        message(FATAL_ERROR "[EGTtools] Unknown sanitizer '${_sanitizer}' in EGTTOOLS_SANITIZE. "
                "Valid values: ${_egttools_known_sanitizers}")
    endif ()
endforeach ()

if ("thread" IN_LIST EGTTOOLS_SANITIZE AND ("address" IN_LIST EGTTOOLS_SANITIZE OR "leak" IN_LIST EGTTOOLS_SANITIZE))
    message(FATAL_ERROR "[EGTtools] The thread sanitizer cannot be combined with address or leak.")
endif ()

if (MSVC)
    if (NOT EGTTOOLS_SANITIZE STREQUAL "address")
        message(FATAL_ERROR "[EGTtools] MSVC supports only EGTTOOLS_SANITIZE=address "
                "(got '${EGTTOOLS_SANITIZE}').")
    endif ()
    add_compile_options(/fsanitize=address /Zi)
    add_link_options(/INCREMENTAL:NO /DEBUG)
else ()
    list(JOIN EGTTOOLS_SANITIZE "," _egttools_sanitize_flags)
    add_compile_options(-fsanitize=${_egttools_sanitize_flags} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${_egttools_sanitize_flags})
    if ("undefined" IN_LIST EGTTOOLS_SANITIZE)
        add_compile_options(-fno-sanitize-recover=undefined)
    endif ()
endif ()

# Read by cpp/src/CMakeLists.txt to set INTERPROCEDURAL_OPTIMIZATION.
set(LTO_SUPPORTED OFF)

message(STATUS "[EGTtools] Sanitizers enabled: ${EGTTOOLS_SANITIZE} (IPO / LTO disabled)")
