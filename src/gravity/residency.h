#pragma once

// gravity residency: keep the resident bytes of mmap'd expert tensors under a budget (Mac option c).
// The GPU reads the experts from the file mmap; this drops the pages of cold experts so the OS
// can reuse the memory. A dropped page faults back in from the same file, so outputs do not change.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gravity {

struct residency_range {
    uintptr_t addr = 0;
    size_t    len  = 0;
};

struct residency_status {
    bool        ok = true;
    std::string error;
};

struct residency_stats {
    uint64_t n_steps         = 0;
    uint64_t n_evicted       = 0; // expert units dropped
    uint64_t bytes_evicted   = 0; // page-aligned bytes passed to the OS
    uint64_t n_refault_units = 0; // units used again after a drop (GPU faults their pages back in)
};

// Pure LRU accounting. A unit is one expert of one layer: its slice in every expert tensor of that layer.
// All units start resident (load and warmup touch every expert). Not thread safe.
class residency_lru {
  public:
    residency_lru(size_t budget_bytes, size_t page_size);

    // Expert e of this tensor is [base + e*slice_bytes, base + (e+1)*slice_bytes).
    // All tensors of one layer must have the same n_expert.
    residency_status add_tensor(int layer, uintptr_t base, int n_expert, size_t slice_bytes);

    // Out-of-range ids are ignored and counted in n_bad_ids().
    void mark_used(int layer, int expert);

    // Ends the current step. Drops LRU units not used in this step until the estimate is <= budget.
    // evict_used: units used in this step may go too (last), for when the next steps do not read the mmap (MoE cache).
    // Returns the page-aligned inner ranges of the dropped slices.
    std::vector<residency_range> end_step(bool evict_used = false);

    size_t resident_bytes() const { return resident; }
    size_t total_bytes() const { return total; }
    size_t budget_bytes() const { return budget; }
    uint64_t n_bad_ids() const { return bad_ids; }
    const residency_stats & stats() const { return st; }

    // every managed slice, for mincore checks
    std::vector<residency_range> all_ranges() const;

  private:
    struct slice_set {
        uintptr_t base;
        size_t    slice_bytes;
    };
    struct layer_info {
        int                    n_expert = 0;
        size_t                 unit_bytes = 0;
        size_t                 first_unit = 0;
        std::vector<slice_set> tensors;
    };

    void   ensure_units();
    size_t unit_of(int layer, int expert) const;

    size_t budget;
    size_t page;
    size_t resident = 0;
    size_t total    = 0;
    uint64_t step   = 1;
    uint64_t bad_ids = 0;
    bool   units_ready = false;

    std::vector<layer_info> layers;
    std::vector<int>        unit_layer;
    std::vector<int>        unit_expert;
    std::vector<uint64_t>   last_use; // 0 = never seen since start
    std::vector<uint8_t>    is_resident;
    residency_stats         st;
};

// page-aligned inner part of [addr, addr+len): rounds start up, end down; len 0 if no full page inside
residency_range residency_inner(uintptr_t addr, size_t len, size_t page_size);

// ways to ask the OS to drop clean file-backed pages of a mapping
enum class drop_method {
    none,
    msync_invalidate, // macOS: drops the pages (measured), Linux: no effect on clean pages
    madv_dontneed,    // Linux: unmaps (RSS drops, page cache keeps them), macOS: no effect (measured)
    posix_dontneed,
    madv_pageout,     // Linux >= 5.4: reclaims, page cache too
};

const char * drop_method_name(drop_method m);

// method used by the residency manager on this OS
drop_method default_drop_method();

// 0 on success, else errno (ENOSYS when the method does not exist here)
int drop_pages(void * addr, size_t len, drop_method m);

// pages of [addr, addr+len) the OS reports in core (mincore). addr must be page-aligned. -1 on error.
int64_t resident_pages(const void * addr, size_t len);

size_t os_page_size();

} // namespace gravity
