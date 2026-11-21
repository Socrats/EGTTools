// The two LRU caches used for fitness values: LRUCache (LruCache.hpp) and ThreadSafeLRUCache
// (utils/ThreadSafeLRUCache.hpp). Each test checks the behaviour the header documents.
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <egttools/LruCache.hpp>
#include <egttools/utils/ThreadSafeLRUCache.hpp>

using egttools::Utils::LRUCache;
using egttools::Utils::ThreadSafeLRUCache;

TEST_CASE("LRUCache evicts the least recently used item", "[cache]") {
    LRUCache<std::string, double> cache(2);
    cache.insert("a", 1.0);
    cache.insert("b", 2.0);
    double value = 0.0;
    REQUIRE(cache.get("a", value));// "a" becomes the most recently used
    CHECK(value == 1.0);
    cache.insert("c", 3.0);// evicts "b"
    CHECK(cache.current_size() == 2);
    CHECK(cache.exists("a"));
    CHECK_FALSE(cache.exists("b"));
    CHECK(cache.exists("c"));
    CHECK_FALSE(cache.get("b", value));
}

TEST_CASE("LRUCache insert of an existing key keeps its value and refreshes it", "[cache]") {
    // Documented in LruCache.hpp: an existing key is only moved to the front.
    LRUCache<std::string, double> cache(2);
    cache.insert("a", 1.0);
    cache.insert("b", 2.0);
    cache.insert("a", 9.0);
    CHECK(cache.get("a") == 1.0);
    cache.insert("c", 3.0);// "b" is now the least recently used
    CHECK(cache.exists("a"));
    CHECK_FALSE(cache.exists("b"));
}

TEST_CASE("LRUCache erase, clear and a smaller maximum size", "[cache]") {
    LRUCache<int, int> cache(3);
    for (int i = 0; i < 3; ++i) cache.insert(i, 10 * i);
    cache.erase(1);
    CHECK_FALSE(cache.exists(1));
    CHECK(cache.current_size() == 2);
    // A smaller maximum takes effect at the next insertion, which keeps the most recent items.
    cache.set_max_size(1);
    CHECK(cache.max_size() == 1);
    cache.insert(5, 50);
    CHECK(cache.current_size() == 1);
    CHECK(cache.exists(5));
    cache.clear();
    CHECK(cache.current_size() == 0);
}

TEST_CASE("ThreadSafeLRUCache evicts in insertion order because get does not refresh", "[cache]") {
    // Documented in ThreadSafeLRUCache.hpp: get() takes a shared lock and skips the LRU update,
    // so eviction follows the order of put() calls. The cached fitness values never change, so
    // this only affects which entry is recomputed.
    ThreadSafeLRUCache<std::string, double> cache(2);
    cache.put("a", 1.0);
    cache.put("b", 2.0);
    REQUIRE(cache.get("a").has_value());
    CHECK(*cache.get("a") == 1.0);
    cache.put("c", 3.0);// evicts "a", the oldest put
    CHECK_FALSE(cache.get("a").has_value());
    CHECK(*cache.get("b") == 2.0);
    CHECK(*cache.get("c") == 3.0);
}

TEST_CASE("ThreadSafeLRUCache put updates and refreshes an existing key", "[cache]") {
    ThreadSafeLRUCache<std::string, double> cache(2);
    cache.put("a", 1.0);
    cache.put("b", 2.0);
    cache.put("a", 5.0);// updated and moved to the front
    CHECK(*cache.get("a") == 5.0);
    cache.put("c", 3.0);// evicts "b"
    CHECK_FALSE(cache.get("b").has_value());
    CHECK(cache.get("a").has_value());
}

TEST_CASE("ThreadSafeLRUCache remove, clear and size 0", "[cache]") {
    ThreadSafeLRUCache<int, int> cache(2);
    cache.put(1, 10);
    cache.remove(1);
    CHECK_FALSE(cache.get(1).has_value());
    cache.put(2, 20);
    cache.clear();
    CHECK_FALSE(cache.get(2).has_value());
    // Size 0 disables the cache.
    ThreadSafeLRUCache<int, int> disabled(0);
    disabled.put(1, 10);
    CHECK_FALSE(disabled.get(1).has_value());
}

TEST_CASE("ThreadSafeLRUCache keeps the values put by concurrent threads", "[cache][threads]") {
    // Each thread writes its own keys; with room for all of them nothing is evicted, so every
    // key must be readable afterwards with its value. (Data races are checked by TSan, PERF-06.)
    constexpr int nb_threads = 4, keys_per_thread = 250;
    ThreadSafeLRUCache<int, int> cache(nb_threads * keys_per_thread);
    std::vector<std::thread> threads;
    for (int t = 0; t < nb_threads; ++t)
        threads.emplace_back([&cache, t] {
            for (int i = 0; i < keys_per_thread; ++i) {
                const int key = t * keys_per_thread + i;
                cache.put(key, 2 * key);
                static_cast<void>(cache.get(key));
            }
        });
    for (auto &thread: threads) thread.join();
    int missing_or_wrong = 0;
    for (int key = 0; key < nb_threads * keys_per_thread; ++key) {
        const auto value = cache.get(key);
        missing_or_wrong += !value.has_value() || *value != 2 * key;
    }
    CHECK(missing_or_wrong == 0);
}
