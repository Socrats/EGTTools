/** Copyright (c) 2019-2021  Elias Fernandez
  *
  * This file is part of EGTtools.
  *
  * EGTtools is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * EGTtools is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/

#pragma once
#ifndef EGTTOOLS_LRUCACHE_HPP
#define EGTTOOLS_LRUCACHE_HPP


#include <list>
#include <unordered_map>
#include <cassert>

namespace egttools::Utils {
    template<class TKey, class TValue>
    class LRUCache {
    public:
        using KVPair = std::pair<TKey, TValue>;
        using DictPair = std::list<KVPair>;
        using CacheMap = std::unordered_map<TKey, typename DictPair::iterator>;

        /**
         * @brief Constructs an LRU Cache of maximum size @param size.
         *
         * @param size maximum size of the cache
         */
        explicit LRUCache(size_t size);

        ~LRUCache() { clear(); }

        /**
         * @brief inserts an item in the cache.
         *
         * First checks if the item already exists. If it does, it will put the
         * item at the front of the list.
         * Otherwise, it will insert the element at the front of the list and add an entry
         * in the HashMap.
         *
         * If the cache exceeds its maximum size, it will evict the last elements of the list.
         *
         * @param key Key of the element to be inserted in cache
         * @param value Value/Data/Item to be inserted in cache
         */
        void insert(const TKey &key, const TValue &value);

        /**
         * @brief Retrieves an element from cache with @param key.
         *
         * If the @param key is not in cache, it will produce an assertion error.
         *
         * @param key of the element
         * @return a reference to the element.
         */
        TValue& get(const TKey &key);

        /**
         * @brief Retrieves an element from cache.
         *
         * If the @param key is not in cache, it will return false, otherwise it will return true.
         * The element will be returned in value
         *
         * @param key Key of the element/item searched
         * @param value container for the value/item searched
         * @return true if element found, otherwise false.
         */
        bool get(const TKey &key, TValue& value);

        /**
         * @brief Erases an item from the cache.
         *
         * If the item with @param key is not in cache, produces an assertion error.
         *
         * @param key Key of the item to be erased.
         */
        void erase(const TKey &key);

        /**
         * @brief Checks if an item is in cache.
         * @param key Key of the item searched.
         * @return true if item in cache, otherwise false.
         */
        bool exists(const TKey &key);

        /**
         * @brief Removes all elements from cache (which are destroyed), and leaves the cache with size of 0.
         */
        void clear();

        // getters
        [[nodiscard]] size_t max_size() const;
        [[nodiscard]] size_t current_size() const;

        // setters
        void set_max_size(size_t max_size);

    private:
        DictPair item_list_; // List with the references to the values and keys
        CacheMap cache_map_; // Hash table to get pointers to the (key, value) pairs
        size_t max_size_; // maximum storing capacity of the cache

        void clean_extra_space_();
    };

    template<class TKey, class TValue>
    LRUCache<TKey, TValue>::LRUCache(size_t size) : max_size_(size) {
        // To increase performance of hash table
        cache_map_.reserve(size / 10);
        cache_map_.max_load_factor(0.25);
    }

    template<class TKey, class TValue>
    void LRUCache<TKey, TValue>::insert(const TKey &key, const TValue &value) {
        auto it = cache_map_.find(key);
        // If key already in cache
        if (it != cache_map_.end()) {
            item_list_.splice(item_list_.begin(), item_list_, it->second);
        } else {
            item_list_.push_front(std::make_pair(key, value));
            cache_map_.insert(std::make_pair(key, item_list_.begin()));
        }

        // After inserting we need to cleanup if current_size_ > max_size_
        clean_extra_space_();
    }

    template<class TKey, class TValue>
    TValue& LRUCache<TKey, TValue>::get(const TKey &key) {
        assert(exists(key));
        auto it = cache_map_.find(key);
        // update position of the item
        item_list_.splice(item_list_.begin(), item_list_, it->second);

        return it->second->second;
    }

    template<class TKey, class TValue>
    bool LRUCache<TKey, TValue>::get(const TKey &key, TValue &value) {
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) return false;

        // update position of the item
        item_list_.splice(item_list_.begin(), item_list_, it->second);
        value = it->second->second;
        return true;
    }

    template<class TKey, class TValue>
    void LRUCache<TKey, TValue>::erase(const TKey &key) {
        assert(exists(key));
        auto it = cache_map_.find(key);
        item_list_.erase(it->second);
        cache_map_.erase(it);
    }

    template<class TKey, class TValue>
    inline bool LRUCache<TKey, TValue>::exists(const TKey &key) {
        return (cache_map_.count(key) > 0);
    }

    template<class TKey, class TValue>
    void LRUCache<TKey, TValue>::clear() {
        item_list_.clear();
        cache_map_.clear();
    }

    template<class TKey, class TValue>
    inline void LRUCache<TKey, TValue>::clean_extra_space_() {
        while(cache_map_.size() > max_size_) {
            auto last_it = item_list_.end();
            last_it--;
            cache_map_.erase(last_it->first);
            item_list_.pop_back();
        }
    }

    template<class TKey, class TValue>
    size_t LRUCache<TKey, TValue>::max_size() const {
        return max_size_;
    }

    template<class TKey, class TValue>
    size_t LRUCache<TKey, TValue>::current_size() const {
        return cache_map_.size();
    }

    template<class TKey, class TValue>
    void LRUCache<TKey, TValue>::set_max_size(size_t max_size) {
        max_size_ = max_size;
    }
}



#endif //EGTTOOLS_LRUCACHE_HPP
