#include "io-mode.h"

#include "read-pool.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef _WIN32
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace gravity {

const char * io_mode_name(io_mode mode) {
    return mode == io_mode::direct ? "direct" : "page_cache";
}

static std::string fmt_gbps(double v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

io_mode_choice choose_io_mode(uint64_t                model_bytes,
                              uint64_t                budget_bytes,
                              const io_probe_result & probe,
                              const io_mode_cfg &     cfg) {
    io_mode_choice res;
    if (model_bytes <= budget_bytes) {
        res.reason = "model fits budget, page cache keeps hot experts";
        return res;
    }
    if (!probe.ok || !std::isfinite(probe.direct_gbps) || !std::isfinite(probe.cached_gbps) || probe.direct_gbps <= 0.0) {
        res.reason = "probe missing or failed" + (probe.error.empty() ? std::string() : ": " + probe.error);
        return res;
    }
    if (probe.direct_gbps < cfg.min_direct_gbps) {
        res.reason = "direct " + fmt_gbps(probe.direct_gbps) + " GB/s below " + fmt_gbps(cfg.min_direct_gbps) + " GB/s";
        return res;
    }
    if (!cfg.memory_bound && probe.direct_gbps < cfg.min_speedup * probe.cached_gbps) {
        res.reason = "direct " + fmt_gbps(probe.direct_gbps) + " GB/s not " + fmt_gbps(cfg.min_speedup) +
                     "x cached " + fmt_gbps(probe.cached_gbps) + " GB/s";
        return res;
    }
    res.mode = io_mode::direct;
    if (cfg.memory_bound) {
        res.reason = "memory bound: model over budget, direct " + fmt_gbps(probe.direct_gbps) + " GB/s vs cached " +
                     fmt_gbps(probe.cached_gbps) + " GB/s";
        return res;
    }
    res.reason = "model over budget, direct " + fmt_gbps(probe.direct_gbps) + " GB/s vs cached " +
                 fmt_gbps(probe.cached_gbps) + " GB/s";
    return res;
}

#ifdef _WIN32

direct_fd open_direct(const std::string &) {
    direct_fd res;
    res.error = "direct IO not supported on this platform";
    return res;
}

io_probe_result probe_io(const std::string &, const io_probe_cfg &) {
    io_probe_result res;
    res.error = "probe not supported on this platform";
    return res;
}

#else

static constexpr size_t default_align = 4096;

direct_fd open_direct(const std::string & path) {
    direct_fd res;
#    if defined(__linux__)
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        res.error = "open O_DIRECT " + path + ": " + strerror(errno);
        return res;
    }
    res.alignment = default_align;
#        ifdef STATX_DIOALIGN
    // kernel 6.1+ reports the real direct IO alignment; older kernels leave the mask bit clear
    struct statx stx {};
    if (statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) == 0 && (stx.stx_mask & STATX_DIOALIGN) &&
        stx.stx_dio_offset_align != 0) {
        res.alignment = std::max<size_t>(stx.stx_dio_offset_align, stx.stx_dio_mem_align);
    }
#        endif
    res.fd = fd;
#    elif defined(__APPLE__)
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        res.error = "open " + path + ": " + strerror(errno);
        return res;
    }
    if (fcntl(fd, F_NOCACHE, 1) == -1) {
        res.error = "fcntl F_NOCACHE " + path + ": " + strerror(errno);
        close(fd);
        return res;
    }
    res.fd        = fd;
    res.alignment = default_align;
#    else
    res.error = "direct IO not supported on this platform: " + path;
#    endif
    return res;
}

namespace {

using probe_clock = std::chrono::steady_clock;

struct pass_result {
    uint64_t    bytes = 0;
    double      sec   = 0.0;
    std::string error;
};

// phase 0 reads even slots, phase 1 odd slots, so the direct pass does not read pages the cached pass just cached
pass_result timed_pass(int fd, size_t align, uint64_t file_size, size_t block, uint64_t max_bytes, int64_t max_ms, int phase) {
    pass_result res;
    aligned_buffer buf(block, std::max<size_t>(align, default_align));
    if (!buf.ok()) {
        res.error = "probe buffer alloc failed";
        return res;
    }
    const uint64_t n_slots = std::max<uint64_t>(1, file_size / block);
    const uint64_t n_reads = std::max<uint64_t>(1, std::min<uint64_t>(max_bytes / block, n_slots / 2));

    const auto t0       = probe_clock::now();
    const auto deadline = t0 + std::chrono::milliseconds(max_ms);
    for (uint64_t i = 0; i < n_reads; ++i) {
        uint64_t slot_idx = ((2 * i + (uint64_t) phase) * n_slots) / (2 * n_reads);
        slot_idx          = std::min(slot_idx, n_slots - 1);
        const uint64_t off = slot_idx * block;
        if (off >= file_size) {
            break;
        }
        size_t len = (size_t) std::min<uint64_t>(block, file_size - off);
        if (align != 0) {
            // tail of the file: aligned length past EOF, kernel returns the short count
            len = (len + align - 1) & ~(align - 1);
        }
        read_req req;
        req.fd            = fd;
        req.offset        = off;
        req.len           = len;
        req.dst           = buf.data();
        req.alignment     = align;
        req.allow_partial = true;
        read_completion c = read_full(req);
        if (c.status != read_status::ok) {
            res.error = c.error;
            return res;
        }
        res.bytes += c.bytes;
        if (probe_clock::now() >= deadline) {
            break;
        }
    }
    res.sec = std::chrono::duration<double>(probe_clock::now() - t0).count();
    if (res.bytes == 0) {
        res.error = "probe read 0 bytes";
    }
    return res;
}

double to_gbps(const pass_result & p) {
    return (double) p.bytes / std::max(p.sec, 1e-9) / 1e9;
}

} // namespace

io_probe_result probe_io(const std::string & path, const io_probe_cfg & cfg) {
    io_probe_result res;
    const auto      t0 = probe_clock::now();

    if (cfg.block == 0 || cfg.max_bytes == 0 || cfg.max_ms <= 0) {
        res.error = "bad probe cfg";
        return res;
    }

    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        res.error = "open " + path + ": " + strerror(errno);
        return res;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        res.error = std::string("fstat: ") + strerror(errno);
        close(fd);
        return res;
    }
    const uint64_t file_size = (uint64_t) st.st_size;
    if (file_size == 0) {
        res.error = "empty file";
        close(fd);
        return res;
    }

    const uint64_t half_bytes = cfg.max_bytes / 2;
    const int64_t  half_ms    = std::max<int64_t>(1, cfg.max_ms / 2);
    // block must stay a multiple of the direct alignment and within the byte bound
    size_t block = (size_t) std::min<uint64_t>(cfg.block, std::max<uint64_t>(half_bytes, default_align));
    block        = std::max(default_align, block & ~(default_align - 1));

    pass_result cached = timed_pass(fd, 0, file_size, block, half_bytes, half_ms, 0);
    close(fd);
    if (!cached.error.empty()) {
        res.error = "cached pass: " + cached.error;
        return res;
    }
    res.cached_gbps = to_gbps(cached);
    res.bytes_read  = cached.bytes;

    direct_fd dfd = open_direct(path);
    if (!dfd.ok()) {
        res.error = dfd.error;
    } else {
        if (block % dfd.alignment != 0) {
            block = (block + dfd.alignment - 1) & ~(dfd.alignment - 1);
        }
        pass_result direct = timed_pass(dfd.fd, dfd.alignment, file_size, block, half_bytes, half_ms, 1);
        close(dfd.fd);
        if (!direct.error.empty()) {
            res.error = "direct pass: " + direct.error;
        } else {
            res.direct_gbps = to_gbps(direct);
            res.bytes_read += direct.bytes;
            res.ok = true;
        }
    }
    res.elapsed_ms = std::chrono::duration<double, std::milli>(probe_clock::now() - t0).count();
    return res;
}

#endif

} // namespace gravity
