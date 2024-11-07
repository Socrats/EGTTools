//
// Created by Elias Fernandez on 06/11/2024.
//
#pragma once
#ifndef EGTTOOLS_UTILS_THREADSAFELRUCACHE_HPP
#define EGTTOOLS_UTILS_THREADSAFELRUCACHE_HPP


#include <list>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

namespace egttools::Utils {

    template<typename Key, typename Value>
    class ThreadSafeLRUCache {
    public:
        explicit ThreadSafeLRUCache(size_t max_size) : max_size_(max_size) {}

        // Retrieves a value from the cache. Returns std::nullopt if the key does not exist.
        std::optional<Value> get(const Key& key) {
            std::unique_lock lock(mutex_);// Exclusive lock to modify usage order
            auto it = cache_items_map_.find(key);
            if (it == cache_items_map_.end()) {
                return std::nullopt;// Key not found
            }
            // Move accessed item to the front of the list (most recently used)
            cache_items_list_.splice(cache_items_list_.begin(), cache_items_list_, it->second.first);
            return it->second.second;// Return the found value
        }

        // Inserts or updates a value in the cache.
        void put(const Key& key, const Value& value) {
            std::unique_lock lock(mutex_);// Exclusive lock for writing
            auto it = cache_items_map_.find(key);

            // If the item exists, update the value and move it to the front
            if (it != cache_items_map_.end()) {
                it->second.second = value;
                cache_items_list_.splice(cache_items_list_.begin(), cache_items_list_, it->second.first);
                return;
            }

            // If the cache is full, remove the least recently used item
            if (cache_items_map_.size() == max_size_) {
                auto lru = cache_items_list_.back();// LRU item is at the back
                cache_items_map_.erase(lru);        // Remove LRU item from map
                cache_items_list_.pop_back();       // Remove from list
            }

            // Insert the new item at the front of the list
            cache_items_list_.emplace_front(key);
            cache_items_map_[key] = {cache_items_list_.begin(), value};
        }

        // Removes a key from the cache.
        void remove(const Key& key) {
            std::unique_lock lock(mutex_);
            auto it = cache_items_map_.find(key);
            if (it != cache_items_map_.end()) {
                cache_items_list_.erase(it->second.first);// Remove from list
                cache_items_map_.erase(it);               // Remove from map
            }
        }

        // Clears the entire cache.
        void clear() {
            std::unique_lock lock(mutex_);
            cache_items_list_.clear();
            cache_items_map_.clear();
        }

    private:
        size_t max_size_;
        std::list<Key> cache_items_list_;// Stores keys in usage order (front = most recent)
        std::unordered_map<Key, std::pair<typename std::list<Key>::iterator, Value>> cache_items_map_;
        mutable std::shared_mutex mutex_;// shared_mutex allows multiple readers, single writer
    };

}// namespace egttools::Utils

#endif//EGTTOOLS_UTILS_THREADSAFELRUCACHE_HPP
