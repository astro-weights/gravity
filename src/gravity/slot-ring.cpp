#include "slot-ring.h"

namespace gravity {

slot_ring::slot_ring(size_t n_slots, size_t slot_bytes, size_t alignment) : bytes(slot_bytes), align(alignment) {
    if (n_slots == 0 || slot_bytes == 0) {
        err = "slot ring needs n_slots > 0 and slot_bytes > 0";
        return;
    }
    try {
        slots.resize(n_slots);
    } catch (...) {
        slots.clear();
        err = "slot ring: out of memory";
        return;
    }
    for (auto & e : slots) {
        e.buf = aligned_buffer(slot_bytes, alignment);
        if (!e.buf.ok()) {
            slots.clear();
            err = "slot ring: aligned alloc failed (bad alignment or out of memory)";
            return;
        }
    }
}

int slot_ring::lookup(const expert_key & key) const {
    std::lock_guard<std::mutex> lock(mtx);
    auto                        it = index.find(key);
    return it == index.end() ? -1 : it->second;
}

acquire_result slot_ring::acquire(const expert_key & key) {
    acquire_result              res;
    std::lock_guard<std::mutex> lock(mtx);
    if (slots.empty()) {
        res.error = err.empty() ? "slot ring empty" : err;
        return res;
    }
    auto it = index.find(key);
    if (it != index.end()) {
        entry & e = slots[it->second];
        e.pins++;
        e.tick    = ++clock;
        res.slot  = it->second;
        res.hit   = true;
        res.ready = e.ready;
        return res;
    }

    int victim = -1;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (!slots[i].bound && slots[i].pins == 0) {
            victim = (int) i;
            break;
        }
    }
    if (victim < 0) {
        for (size_t i = 0; i < slots.size(); ++i) {
            const entry & e = slots[i];
            if (e.pins == 0 && (victim < 0 || e.tick < slots[victim].tick)) {
                victim = (int) i;
            }
        }
    }
    if (victim < 0) {
        res.error      = "all " + std::to_string(slots.size()) + " slots pinned";
        res.all_pinned = true;
        return res;
    }
    entry & e = slots[victim];
    try {
        index[key] = victim;
    } catch (...) {
        res.error = "slot ring: out of memory";
        return res;
    }
    if (e.bound) {
        index.erase(e.key);
    }
    e.key          = key;
    e.bound        = true;
    e.ready        = false;
    e.pins         = 1;
    e.tick         = ++clock;
    res.slot       = victim;
    res.needs_fill = true;
    return res;
}

bool slot_ring::pin(int slot) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!valid(slot)) {
        return false;
    }
    slots[slot].pins++;
    return true;
}

bool slot_ring::unpin(int slot) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!valid(slot) || slots[slot].pins <= 0) {
        return false;
    }
    slots[slot].pins--;
    return true;
}

bool slot_ring::mark_ready(int slot) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!valid(slot) || !slots[slot].bound) {
        return false;
    }
    slots[slot].ready = true;
    return true;
}

bool slot_ring::fail_fill(int slot) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!valid(slot) || !slots[slot].bound) {
        return false;
    }
    entry & e = slots[slot];
    index.erase(e.key);
    e.bound = false;
    e.ready = false;
    e.tick  = 0;
    return true;
}

void * slot_ring::data(int slot) const {
    std::lock_guard<std::mutex> lock(mtx);
    return valid(slot) ? slots[slot].buf.data() : nullptr;
}

size_t slot_ring::n_free() const {
    std::lock_guard<std::mutex> lock(mtx);
    size_t                      n = 0;
    for (const auto & e : slots) {
        n += (!e.bound && e.pins == 0) ? 1 : 0;
    }
    return n;
}

int slot_ring::pins(int slot) const {
    std::lock_guard<std::mutex> lock(mtx);
    return valid(slot) ? slots[slot].pins : -1;
}

bool slot_ring::is_ready(int slot) const {
    std::lock_guard<std::mutex> lock(mtx);
    return valid(slot) && slots[slot].ready;
}

}  // namespace gravity
