//
// Created by Elias Fernandez on 06/11/2024.
//
#include <egttools/utils/ThreadSafeLRUCache.hpp>
#include <iostream>
#include <string>

int main() {
    egttools::Utils::ThreadSafeLRUCache<std::string, double> cache(2);

    cache.put("[0,0,0,0]", 0.1);
    cache.put("[2,3,4,5]", 2.1);

    // Accessing key 1 should make it most recently used
    std::cout << "Key 1: " << *cache.get("[0,0,0,0]") << std::endl;

    // Adding a third item should evict the least recently used (key 2)
    cache.put("[1,2,1,2]", 3.5);

    if (auto value = cache.get("[2,3,4,5]"); value) {
        std::cout << "Key 2: " << *value << std::endl;
    } else {
        std::cout << "Key 2 was evicted." << std::endl;
    }

    return 0;
}