#pragma once

// gravity streaming S1: memory budget detection + stepped fill/shrink policy.
// Parsers and policy are pure (text/numbers in, values out). OS reads are thin.

#include <cstdint>
#include <functional>
#include <string>

namespace gravity {

constexpr uint64_t MEM_GIB = 1024ull * 1024 * 1024;

// no exceptions across the API: error is empty on success
template <typename T>
struct mem_result {
    T           value{};
    std::string error;
    bool ok() const { return error.empty(); }
};

struct mem_info {
    uint64_t    total               = 0;
    uint64_t    available           = 0;
    uint64_t    cgroup_limit        = 0; // 0 = no limit
    uint64_t    cgroup_used         = 0;
    uint64_t    zfs_arc_reclaimable = 0; // ARC is not counted in MemAvailable
    std::string source;                  // "linux" or "macos"
    std::string note;                    // missing inputs, not errors
};

struct meminfo_values {
    uint64_t total     = 0;
    uint64_t available = 0;
};

struct cgroup_values {
    uint64_t limit = 0; // 0 = no limit
    uint64_t used  = 0;
};

// host_statistics64 counts. free_pages must exclude speculative pages (the
// kernel's free_count includes them, so the thin reader subtracts first).
struct vm_counts {
    uint64_t free_pages        = 0;
    uint64_t inactive_pages    = 0;
    uint64_t purgeable_pages   = 0;
    uint64_t speculative_pages = 0;
    uint64_t page_size         = 0;
    uint64_t memsize           = 0; // hw.memsize
};

mem_result<meminfo_values> parse_meminfo(const std::string & text);
// memory.max ("max" = no limit) and memory.current
mem_result<cgroup_values>  parse_cgroup_v2(const std::string & max_text, const std::string & current_text);
// limit_in_bytes values near 2^63 mean no limit
mem_result<cgroup_values>  parse_cgroup_v1(const std::string & limit_text, const std::string & usage_text);
// size - c_min, floor 0
mem_result<uint64_t>       parse_zfs_arcstats(const std::string & text);
mem_info                   mem_info_from_vm_counts(const vm_counts & c);

// returns false when the file does not exist
using file_reader = std::function<bool(const std::string & path, std::string & out)>;

// Linux assembly over injected reads. Missing optional file = field 0 + note.
mem_result<mem_info> read_linux_mem_info(const file_reader & read);

// host: /proc + /sys on Linux, host_statistics64 + sysctl on macOS
mem_result<mem_info> read_mem_info();

struct budget_cfg {
    uint64_t keep_min_bytes = 7 * MEM_GIB;
    double   keep_frac      = 0.15;
    uint64_t step_bytes     = 3 * MEM_GIB / 2;
    uint64_t max_bytes      = 0; // user cap on our fill, 0 = none
    uint32_t cooldown_steps = 2; // next_action calls with no grow after a shrink
};

uint64_t effective_available(const mem_info & m);
uint64_t keep_for_os(uint64_t total, const budget_cfg & cfg);
// bytes we may still add on top of what we already hold
uint64_t compute_budget(const mem_info & m, const budget_cfg & cfg);

struct step_action {
    enum kind_t { hold, grow, shrink, lock_now };
    kind_t   kind  = hold;
    uint64_t bytes = 0;
};

class budget_stepper {
  public:
    explicit budget_stepper(const budget_cfg & cfg) : cfg_(cfg) {}
    // current_bytes = what we hold now; m = fresh reading (our fill is already out of m.available)
    step_action next_action(uint64_t current_bytes, const mem_info & m);

  private:
    budget_cfg cfg_;
    uint32_t   cooldown_left_ = 0;
    bool       locked_        = false;
};

} // namespace gravity
