#include "residency.h"

#include <algorithm>
#include <cerrno>

#if defined(__APPLE__) || defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#    define GRAVITY_HAVE_MMAN 1
#endif

namespace gravity {

residency_lru::residency_lru(size_t budget_bytes, size_t page_size) : budget(budget_bytes), page(page_size ? page_size : 4096) {}

residency_status residency_lru::add_tensor(int layer, uintptr_t base, int n_expert, size_t slice_bytes) {
    if (units_ready) {
        return { false, "add_tensor after first use" };
    }
    if (layer < 0 || n_expert <= 0 || slice_bytes == 0) {
        return { false, "bad layer, n_expert or slice size" };
    }
    if ((size_t) layer >= layers.size()) {
        layers.resize(layer + 1);
    }
    layer_info & li = layers[layer];
    if (li.n_expert != 0 && li.n_expert != n_expert) {
        return { false, "n_expert differs inside layer " + std::to_string(layer) };
    }
    li.n_expert = n_expert;
    li.unit_bytes += slice_bytes;
    li.tensors.push_back({ base, slice_bytes });
    total += slice_bytes * (size_t) n_expert;
    resident += slice_bytes * (size_t) n_expert;
    return {};
}

void residency_lru::ensure_units() {
    if (units_ready) {
        return;
    }
    units_ready = true;
    size_t n = 0;
    for (size_t il = 0; il < layers.size(); ++il) {
        layers[il].first_unit = n;
        for (int e = 0; e < layers[il].n_expert; ++e) {
            unit_layer.push_back((int) il);
            unit_expert.push_back(e);
        }
        n += layers[il].n_expert;
    }
    last_use.assign(n, 0);
    is_resident.assign(n, 1);
}

size_t residency_lru::unit_of(int layer, int expert) const {
    return layers[layer].first_unit + (size_t) expert;
}

void residency_lru::mark_used(int layer, int expert) {
    ensure_units();
    if (layer < 0 || (size_t) layer >= layers.size() || expert < 0 || expert >= layers[layer].n_expert) {
        bad_ids++;
        return;
    }
    const size_t u = unit_of(layer, expert);
    if (!is_resident[u]) {
        is_resident[u] = 1;
        resident += layers[layer].unit_bytes;
        st.n_refault_units++;
    }
    last_use[u] = step;
}

std::vector<residency_range> residency_lru::end_step(bool evict_used) {
    ensure_units();
    std::vector<residency_range> out;
    if (resident > budget) {
        std::vector<size_t> cand;
        for (size_t u = 0; u < is_resident.size(); ++u) {
            if (is_resident[u] && (evict_used || last_use[u] != step)) {
                cand.push_back(u);
            }
        }
        // oldest first, ties by unit index so the order is deterministic
        std::sort(cand.begin(), cand.end(), [&](size_t a, size_t b) {
            return last_use[a] != last_use[b] ? last_use[a] < last_use[b] : a < b;
        });
        for (size_t u : cand) {
            if (resident <= budget) {
                break;
            }
            const layer_info & li = layers[unit_layer[u]];
            const size_t       e  = (size_t) unit_expert[u];
            for (const slice_set & t : li.tensors) {
                const residency_range r = residency_inner(t.base + e * t.slice_bytes, t.slice_bytes, page);
                if (r.len > 0) {
                    out.push_back(r);
                    st.bytes_evicted += r.len;
                }
            }
            is_resident[u] = 0;
            resident -= li.unit_bytes;
            st.n_evicted++;
        }
    }
    st.n_steps++;
    step++;
    return out;
}

std::vector<residency_range> residency_lru::all_ranges() const {
    std::vector<residency_range> out;
    for (const layer_info & li : layers) {
        for (const slice_set & t : li.tensors) {
            out.push_back({ t.base, t.slice_bytes * (size_t) li.n_expert });
        }
    }
    return out;
}

residency_range residency_inner(uintptr_t addr, size_t len, size_t page_size) {
    const uintptr_t beg = (addr + page_size - 1) / page_size * page_size;
    const uintptr_t end = (addr + len) / page_size * page_size;
    if (end <= beg) {
        return { beg, 0 };
    }
    return { beg, (size_t) (end - beg) };
}

const char * drop_method_name(drop_method m) {
    switch (m) {
        case drop_method::none:             return "none";
        case drop_method::msync_invalidate: return "msync(MS_INVALIDATE)";
        case drop_method::madv_dontneed:    return "madvise(MADV_DONTNEED)";
        case drop_method::posix_dontneed:   return "posix_madvise(POSIX_MADV_DONTNEED)";
        case drop_method::madv_pageout:     return "madvise(MADV_PAGEOUT)";
    }
    return "?";
}

drop_method default_drop_method() {
#if defined(__APPLE__)
    return drop_method::msync_invalidate;
#elif defined(__linux__)
    return drop_method::madv_dontneed;
#else
    return drop_method::none;
#endif
}

int drop_pages(void * addr, size_t len, drop_method m) {
#ifdef GRAVITY_HAVE_MMAN
    int rc = 0;
    switch (m) {
        case drop_method::none:
            return 0;
        case drop_method::msync_invalidate:
            rc = msync(addr, len, MS_INVALIDATE);
            break;
        case drop_method::madv_dontneed:
            rc = madvise(addr, len, MADV_DONTNEED);
            break;
        case drop_method::posix_dontneed:
            return posix_madvise(addr, len, POSIX_MADV_DONTNEED);
        case drop_method::madv_pageout:
#    ifdef MADV_PAGEOUT
            rc = madvise(addr, len, MADV_PAGEOUT);
            break;
#    else
            return ENOSYS;
#    endif
    }
    return rc == 0 ? 0 : errno;
#else
    (void) addr;
    (void) len;
    return m == drop_method::none ? 0 : ENOSYS;
#endif
}

int64_t resident_pages(const void * addr, size_t len) {
#ifdef GRAVITY_HAVE_MMAN
    const size_t ps = os_page_size();
    const size_t n  = (len + ps - 1) / ps;
#    if defined(__APPLE__)
    std::vector<char> v(n);
#    else
    std::vector<unsigned char> v(n);
#    endif
    if (mincore(const_cast<void *>(addr), len, v.data()) != 0) {
        return -1;
    }
    int64_t r = 0;
    for (auto c : v) {
        r += (c & 1);
    }
    return r;
#else
    (void) addr;
    (void) len;
    return -1;
#endif
}

size_t os_page_size() {
#ifdef GRAVITY_HAVE_MMAN
    const long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? (size_t) ps : 4096;
#else
    return 4096;
#endif
}

} // namespace gravity
