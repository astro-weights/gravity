#pragma once

// gravity streaming: choose page cache or direct IO for expert reads, plus the startup probe that feeds the rule

#include <cstddef>
#include <cstdint>
#include <string>

namespace gravity {

enum class io_mode {
    page_cache,
    direct, // O_DIRECT on Linux, F_NOCACHE on macOS
};

const char * io_mode_name(io_mode mode);

struct io_probe_result {
    bool        ok          = false; // false -> rule picks page cache
    double      cached_gbps = 0.0;   // buffered pread path
    double      direct_gbps = 0.0;   // open_direct path
    uint64_t    bytes_read  = 0;     // both paths together
    double      elapsed_ms  = 0.0;
    std::string error;
};

struct io_mode_cfg {
    double min_direct_gbps = 4.0;
    double min_speedup     = 1.3;   // direct must beat cached by this factor
    bool   memory_bound    = false; // user capped RAM: page cache fill is the cost, so skip min_speedup
};

struct io_mode_choice {
    io_mode     mode = io_mode::page_cache;
    std::string reason;
};

// Pure rule. Eval mode does not enter here: mode changes speed only, never the bytes read.
io_mode_choice choose_io_mode(uint64_t                model_bytes,
                              uint64_t                budget_bytes,
                              const io_probe_result & probe,
                              const io_mode_cfg &     cfg = {});

struct direct_fd {
    int         fd        = -1;
    size_t      alignment = 0; // offset, length and buffer address of each read must be a multiple of this
    std::string error;

    bool ok() const { return fd >= 0; }
};

// Read-only open that bypasses the page cache. Caller closes fd.
direct_fd open_direct(const std::string & path);

struct io_probe_cfg {
    uint64_t max_bytes = 256ull << 20; // total for both paths
    int64_t  max_ms    = 2000;         // total for both paths
    size_t   block     = 1u << 20;
};

// Timed reads at spread offsets: cached path first, then direct path on other offsets so it does not hit pages the first pass cached.
io_probe_result probe_io(const std::string & path, const io_probe_cfg & cfg = {});

} // namespace gravity
