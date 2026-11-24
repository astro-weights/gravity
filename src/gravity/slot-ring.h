#pragma once

// gravity streaming: fixed ring of aligned expert slots, pin refcount, LRU eviction of unpinned slots

#include "read-pool.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace gravity {

struct expert_key {
    uint32_t layer  = 0;
    uint32_t expert = 0;

    bool operator<(const expert_key & o) const { return layer != o.layer ? layer < o.layer : expert < o.expert; }

    bool operator==(const expert_key & o) const { return layer == o.layer && expert == o.expert; }
};

struct acquire_result {
    int         slot       = -1;
    bool        hit        = false;  // key was already bound to the slot
    bool        ready      = false;  // slot holds the key bytes (hit only)
    bool        needs_fill = false;  // miss: caller must fill, then mark_ready() or fail_fill()
    std::string error;

    bool ok() const { return slot >= 0; }
};

// Thread safe. A pinned slot is never evicted. LRU order uses a logical tick, so it depends only on the call sequence.
class slot_ring {
  public:
    slot_ring(size_t n_slots, size_t slot_bytes, size_t alignment);

    slot_ring(const slot_ring &)             = delete;
    slot_ring & operator=(const slot_ring &) = delete;

    // false if any buffer failed to allocate or args are bad; error() says why
    bool ok() const { return err.empty(); }

    const std::string & error() const { return err; }

    // slot bound to key (ready or still filling), or -1
    int lookup(const expert_key & key) const;

    // Pins the returned slot. Miss: lowest free slot, else least recently used unpinned slot. All pinned -> error.
    acquire_result acquire(const expert_key & key);

    bool pin(int slot);
    bool unpin(int slot);  // false if slot bad or not pinned

    bool mark_ready(int slot);
    // unbind the key after a failed fill; slot becomes free once its pins drop to 0
    bool fail_fill(int slot);

    void * data(int slot) const;  // nullptr if slot bad

    size_t slot_bytes() const { return bytes; }

    size_t n_slots() const { return slots.size(); }

    size_t n_free() const;  // no key and no pins
    int    pins(int slot) const;
    bool   is_ready(int slot) const;

  private:
    struct entry {
        aligned_buffer buf;
        expert_key     key;
        bool           bound = false;
        bool           ready = false;
        int            pins  = 0;
        uint64_t       tick  = 0;
    };

    bool valid(int slot) const { return slot >= 0 && (size_t) slot < slots.size(); }

    mutable std::mutex        mtx;
    std::vector<entry>        slots;
    std::map<expert_key, int> index;
    size_t                    bytes = 0;
    uint64_t                  clock = 0;
    std::string               err;
};

}  // namespace gravity
