// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file distributed_shared_mutex.h
/// @brief Shared ownership without one counter contended by every reader.

#include <array>
#include <cstddef>
#include <functional>
#include <shared_mutex>
#include <thread>

namespace util {

/// @brief Shared ownership distributed across independent reader counters.
///
/// @details Readers lock one cache-line-separated shard; writers lock every
/// shard in order. This trades a larger mutex and more expensive writes for independent
/// reader counters. Hash collisions share a shard and preserve exclusion.
/// Like std::shared_mutex, ownership must be released by the acquiring thread.
class DistributedSharedMutex {
public:
  void lock_shared() { shards_[reader_shard()].mutex.lock_shared(); }
  bool try_lock_shared() { return shards_[reader_shard()].mutex.try_lock_shared(); }
  void unlock_shared() { shards_[reader_shard()].mutex.unlock_shared(); }

  void lock() {
    size_t acquired = 0;
    try {
      for (; acquired < kShards; ++acquired)
        shards_[acquired].mutex.lock();
    } catch (...) {
      unlock_prefix(acquired);
      throw;
    }
  }

  bool try_lock() {
    size_t acquired = 0;
    try {
      for (; acquired < kShards; ++acquired) {
        if (!shards_[acquired].mutex.try_lock()) {
          unlock_prefix(acquired);
          return false;
        }
      }
    } catch (...) {
      unlock_prefix(acquired);
      throw;
    }
    return true;
  }

  void unlock() { unlock_prefix(kShards); }

private:
  static constexpr size_t kShards = 128;
  struct alignas(64) Shard {
    std::shared_mutex mutex;
  };
  std::array<Shard, kShards> shards_;

  static size_t reader_shard() {
    // Derive the slot from the thread identity so an inline lock and unlock
    // agree even when compiled into different shared libraries.
    static thread_local const size_t shard =
        std::hash<std::thread::id>{}(std::this_thread::get_id()) % kShards;
    return shard;
  }

  void unlock_prefix(size_t count) {
    while (count != 0)
      shards_[--count].mutex.unlock();
  }
};

} // namespace util
