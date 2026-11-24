// gravity streaming test: io mode rule, probe, read pool

#include "io-mode.h"
#include "read-pool.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#    include <fcntl.h>
#    include <unistd.h>
#endif

using namespace gravity;

static int n_fail = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++n_fail;                                                     \
        }                                                                 \
    } while (0)

static void test_rule() {
    const uint64_t gb = 1ull << 30;
    io_probe_result fast;
    fast.ok          = true;
    fast.cached_gbps = 3.0;
    fast.direct_gbps = 6.0;

    // fits budget -> cache, even with a fast drive
    CHECK(choose_io_mode(10 * gb, 20 * gb, fast).mode == io_mode::page_cache);
    CHECK(choose_io_mode(20 * gb, 20 * gb, fast).mode == io_mode::page_cache);

    // fast and >= 1.3x -> direct
    io_mode_choice c = choose_io_mode(30 * gb, 20 * gb, fast);
    CHECK(c.mode == io_mode::direct);
    CHECK(!c.reason.empty());

    // slow drive -> cache
    io_probe_result slow = fast;
    slow.direct_gbps     = 3.9;
    slow.cached_gbps     = 1.0;
    CHECK(choose_io_mode(30 * gb, 20 * gb, slow).mode == io_mode::page_cache);

    // fast but < 1.3x cached -> cache
    io_probe_result close_r = fast;
    close_r.direct_gbps     = 5.0;
    close_r.cached_gbps     = 4.0;
    CHECK(choose_io_mode(30 * gb, 20 * gb, close_r).mode == io_mode::page_cache);

    // exactly at both thresholds -> direct
    io_probe_result edge = fast;
    edge.direct_gbps     = 4.0;
    edge.cached_gbps     = 4.0 / 1.3;
    CHECK(choose_io_mode(30 * gb, 20 * gb, edge).mode == io_mode::direct);

    // probe failed / missing / garbage -> cache
    io_probe_result failed = fast;
    failed.ok              = false;
    failed.error           = "open failed";
    c                      = choose_io_mode(30 * gb, 20 * gb, failed);
    CHECK(c.mode == io_mode::page_cache);
    CHECK(c.reason.find("open failed") != std::string::npos);
    CHECK(choose_io_mode(30 * gb, 20 * gb, io_probe_result{}).mode == io_mode::page_cache);
    io_probe_result nan_r = fast;
    nan_r.direct_gbps     = NAN;
    CHECK(choose_io_mode(30 * gb, 20 * gb, nan_r).mode == io_mode::page_cache);

    // thresholds are flags
    io_mode_cfg cfg;
    cfg.min_direct_gbps = 10.0;
    CHECK(choose_io_mode(30 * gb, 20 * gb, fast, cfg).mode == io_mode::page_cache);
    cfg.min_direct_gbps = 1.0;
    cfg.min_speedup     = 1.0;
    CHECK(choose_io_mode(30 * gb, 20 * gb, slow, cfg).mode == io_mode::direct);
}

static void test_aligned_buffer() {
    aligned_buffer b(5000, 4096);
    CHECK(b.ok());
    CHECK(b.size() == 8192);
    CHECK((uintptr_t) b.data() % 4096 == 0);
    aligned_buffer moved(std::move(b));
    CHECK(moved.ok() && !b.ok());
    aligned_buffer bad(100, 3000);
    CHECK(!bad.ok());
    aligned_buffer zero(0, 4096);
    CHECK(zero.ok() && zero.size() == 4096);
}

#ifndef _WIN32

struct temp_file {
    std::string path;
    std::vector<uint8_t> data;

    temp_file(size_t size, uint32_t seed) {
        char tmpl[] = "./gravity-io-test-XXXXXX";
        int  fd     = mkstemp(tmpl);
        if (fd < 0) {
            fprintf(stderr, "mkstemp failed: %s\n", strerror(errno));
            return;
        }
        path = tmpl;
        data.resize(size);
        std::mt19937 rng(seed);
        for (auto & b : data) {
            b = (uint8_t) rng();
        }
        size_t done = 0;
        while (done < size) {
            ssize_t w = write(fd, data.data() + done, size - done);
            if (w <= 0) {
                fprintf(stderr, "write failed\n");
                path.clear();
                break;
            }
            done += (size_t) w;
        }
        fsync(fd); // dirty pages would make the direct path pay for writeback
        close(fd);
    }

    ~temp_file() {
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }
};

static void test_probe() {
    {
        temp_file f(64u << 20, 1);
        CHECK(!f.path.empty());
        io_probe_cfg cfg;
        cfg.max_ms = 2000;
        const auto      t0 = std::chrono::steady_clock::now();
        io_probe_result r  = probe_io(f.path, cfg);
        const double    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("probe 64 MiB: ok=%d cached=%.2f GB/s direct=%.2f GB/s bytes=%llu %.1f ms %s\n", r.ok, r.cached_gbps,
               r.direct_gbps, (unsigned long long) r.bytes_read, ms, r.error.c_str());
        CHECK(std::isfinite(r.cached_gbps) && r.cached_gbps > 0);
        CHECK(r.bytes_read <= cfg.max_bytes + 8192);
        CHECK(ms < cfg.max_ms + 1000);
        if (r.ok) {
            CHECK(std::isfinite(r.direct_gbps) && r.direct_gbps > 0);
        } else {
            // e.g. tmpfs refuses O_DIRECT: rule must fall back to cache
            CHECK(choose_io_mode(2, 1, r).mode == io_mode::page_cache);
        }
    }
    {
        temp_file       f(1024, 2);
        io_probe_result r = probe_io(f.path);
        printf("probe 1 KiB: ok=%d cached=%.2f direct=%.2f bytes=%llu %s\n", r.ok, r.cached_gbps, r.direct_gbps,
               (unsigned long long) r.bytes_read, r.error.c_str());
        CHECK(std::isfinite(r.cached_gbps) && r.cached_gbps > 0);
        CHECK(r.bytes_read <= 2048); // never past EOF
        if (r.ok) {
            CHECK(std::isfinite(r.direct_gbps) && r.direct_gbps > 0);
        }
    }
    {
        // tight bounds still give a result and respect the byte cap
        temp_file    f(8u << 20, 3);
        io_probe_cfg cfg;
        cfg.max_bytes     = 1u << 20;
        io_probe_result r = probe_io(f.path, cfg);
        CHECK(r.cached_gbps > 0);
        CHECK(r.bytes_read <= cfg.max_bytes + 8192);
    }
    CHECK(!probe_io("./does-not-exist-gravity").ok);
    CHECK(!open_direct("./does-not-exist-gravity").ok());
}

static void test_pool_random_reads() {
    const size_t size = 8u << 20;
    temp_file    f(size, 4);
    const int    fd = open(f.path.c_str(), O_RDONLY);
    CHECK(fd >= 0);

    read_pool pool(4);
    CHECK(pool.n_threads() == 4);

    const int                 n = 1000;
    std::vector<read_req>     reqs(n);
    std::vector<std::vector<uint8_t>> bufs(n);
    std::mt19937_64           rng(5);
    for (int i = 0; i < n; ++i) {
        uint64_t off = rng() % size;
        size_t   len = 1 + rng() % 65536;
        if (off + len > size) {
            len = size - off;
        }
        bufs[i].assign(len, 0);
        reqs[i].fd     = fd;
        reqs[i].offset = off;
        reqs[i].len    = len;
        reqs[i].dst    = bufs[i].data();
        reqs[i].tag    = (uint64_t) i;
    }

    // submit from 4 threads at once, then wait from the main thread
    std::vector<read_id>     ids(n, 0);
    std::vector<std::thread> subs;
    for (int t = 0; t < 4; ++t) {
        subs.emplace_back([&, t] {
            for (int i = t; i < n; i += 4) {
                ids[i] = pool.submit(reqs[i]).id;
            }
        });
    }
    for (auto & t : subs) {
        t.join();
    }
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        if (ids[i] == 0) {
            ++bad;
            continue;
        }
        read_completion c = pool.wait(ids[i]);
        if (c.status != read_status::ok || c.tag != (uint64_t) i || c.bytes != reqs[i].len ||
            memcmp(bufs[i].data(), f.data.data() + reqs[i].offset, reqs[i].len) != 0) {
            if (bad < 5) {
                fprintf(stderr, "read %d mismatch: %s %s\n", i, read_status_name(c.status), c.error.c_str());
            }
            ++bad;
        }
    }
    CHECK(bad == 0);

    // collected id is forgotten
    read_completion again = pool.wait(ids[0]);
    CHECK(again.status == read_status::error);
    close(fd);
}

static void test_pool_errors() {
    const size_t size = 10000;
    temp_file    f(size, 6);
    int          fd = open(f.path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    read_pool pool(2);

    aligned_buffer buf(8192, 4096);

    // misaligned direct request rejected at submit
    read_req r;
    r.fd        = fd;
    r.dst       = buf.data();
    r.alignment = 4096;
    r.offset    = 100;
    r.len       = 4096;
    submit_result s = pool.submit(r);
    CHECK(!s.ok() && !s.error.empty());
    r.offset = 0;
    r.len    = 100;
    CHECK(!pool.submit(r).ok());
    r.len = 4096;
    r.dst = (char *) buf.data() + 8;
    CHECK(!pool.submit(r).ok());
    r.dst = buf.data();
    s = pool.submit(r); // aligned is fine on any fd
    CHECK(s.ok());
    CHECK(pool.wait(s.id).status == read_status::ok);

    // null dst, bad fd, overflow
    read_req z;
    z.fd  = fd;
    z.len = 10;
    CHECK(!pool.submit(z).ok());
    z.dst = buf.data();
    z.fd  = -1;
    CHECK(!pool.submit(z).ok());
    z.fd     = fd;
    z.offset = UINT64_MAX - 2;
    CHECK(!pool.submit(z).ok());

    // EOF before len -> error; allow_partial -> ok with short count
    read_req e;
    e.fd     = fd;
    e.offset = size - 100;
    e.len    = 500;
    e.dst    = buf.data();
    e.tag    = 77;
    read_completion c = pool.wait(pool.submit(e).id);
    CHECK(c.status == read_status::error && c.bytes == 100 && c.tag == 77);
    e.allow_partial = true;
    c               = pool.wait(pool.submit(e).id);
    CHECK(c.status == read_status::ok && c.bytes == 100);
    e.offset = size + 4096; // fully past EOF
    e.allow_partial = false;
    c               = pool.wait(pool.submit(e).id);
    CHECK(c.status == read_status::error && c.bytes == 0);

    // zero-length read is ok
    read_req zl;
    zl.fd = fd;
    c     = pool.wait(pool.submit(zl).id);
    CHECK(c.status == read_status::ok && c.bytes == 0);

    // read on closed fd -> error, no crash
    close(fd);
    read_req cl;
    cl.fd  = fd;
    cl.len = 10;
    cl.dst = buf.data();
    c      = pool.wait(pool.submit(cl).id);
    CHECK(c.status == read_status::error && c.err != 0 && !c.error.empty());

    // unknown id
    read_completion u;
    CHECK(pool.poll(123456789, u) && u.status == read_status::error);
}

// fake read: requests with tag gate_tag block until open(); counts started reads
struct gate {
    std::mutex              m;
    std::condition_variable cv;
    bool                    is_open = false;
    int                     n_held  = 0;

    static constexpr uint64_t gate_tag = 0xfeed;

    read_fn fn() {
        return [this](const read_req & r) {
            if (r.tag == gate_tag) {
                std::unique_lock<std::mutex> lock(m);
                ++n_held;
                cv.notify_all();
                cv.wait(lock, [this] { return is_open; });
            }
            return read_full(r);
        };
    }

    void wait_held(int n) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] { return n_held >= n; });
    }

    void open() {
        std::lock_guard<std::mutex> lock(m);
        is_open = true;
        cv.notify_all();
    }
};

static void test_pool_cancel_shutdown() {
    temp_file f(4096, 7);
    int       fd = open(f.path.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    char data[64];

    read_req hold;
    hold.fd  = fd;
    hold.len = 16;
    hold.dst = data;
    hold.tag = gate::gate_tag;

    read_req q;
    q.fd  = fd;
    q.len = 64;
    q.dst = data;
    q.tag = 9;

    {
        // cancel before start; running request cannot be cancelled
        gate      g;
        read_pool pool(1, g.fn());
        read_id   hold_id = pool.submit(hold).id;
        g.wait_held(1);
        CHECK(!pool.cancel(hold_id));
        read_completion tmp;
        CHECK(!pool.poll(hold_id, tmp));

        read_id q1 = pool.submit(q).id;
        read_id q2 = pool.submit(q).id;
        CHECK(pool.cancel(q1));
        CHECK(!pool.cancel(q1));
        read_completion c = pool.wait(q1);
        CHECK(c.status == read_status::cancelled && c.tag == 9 && c.bytes == 0);

        g.open();
        c = pool.wait(hold_id);
        CHECK(c.status == read_status::ok && c.bytes == 16);
        c = pool.wait(q2);
        CHECK(c.status == read_status::ok && c.bytes == 64);
        CHECK(!pool.cancel(q2));
    }

    {
        // shutdown with pending: queued complete as cancelled, running ones finish
        gate      g;
        read_pool pool(2, g.fn());
        char      d1[16];
        char      d2[16];
        read_req  hold1 = hold;
        read_req  hold2 = hold;
        hold1.dst       = d1;
        hold2.dst       = d2;
        read_id   h1    = pool.submit(hold1).id;
        read_id   h2    = pool.submit(hold2).id;
        g.wait_held(2);
        std::vector<read_id> pend;
        for (int i = 0; i < 20; ++i) {
            pend.push_back(pool.submit(q).id);
        }
        std::thread opener([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            g.open();
        });
        pool.shutdown();
        opener.join();
        for (read_id id : pend) {
            read_completion c = pool.wait(id);
            CHECK(c.status == read_status::cancelled && c.tag == 9);
        }
        CHECK(pool.wait(h1).status == read_status::ok);
        CHECK(pool.wait(h2).status == read_status::ok);
        CHECK(!pool.submit(q).ok());
        CHECK(pool.n_threads() == 0);
        pool.shutdown(); // idempotent
    }

    {
        // destructor with pending work must not hang or crash
        std::vector<std::vector<char>> bufs(100, std::vector<char>(64));
        read_pool                      pool(2);
        for (int i = 0; i < 100; ++i) {
            read_req r = q;
            r.dst      = bufs[i].data();
            pool.submit(r);
        }
    }

    {
        // throwing read fn -> error status, worker survives
        read_pool pool(1, [](const read_req & r) -> read_completion { if (r.tag == 9) { throw 1; } return read_full(r); });
        read_completion c = pool.wait(pool.submit(q).id);
        CHECK(c.status == read_status::error && c.tag == 9);
        c = pool.wait(pool.submit(q).id);
        CHECK(c.status == read_status::error);
    }
    close(fd);
}

static void test_direct_reads() {
    const size_t size = (1u << 20) + 1000; // tail not a multiple of the alignment
    temp_file    f(size, 8);
    direct_fd    d = open_direct(f.path);
    if (!d.ok()) {
        printf("open_direct unsupported here, skip: %s\n", d.error.c_str());
        return;
    }
    CHECK(d.alignment >= 512 && (d.alignment & (d.alignment - 1)) == 0);
    read_pool      pool(2);
    aligned_buffer buf(1u << 20, d.alignment);
    read_req       r;
    r.fd        = d.fd;
    r.offset    = 0;
    r.len       = 1u << 20;
    r.dst       = buf.data();
    r.alignment = d.alignment;
    read_completion c = pool.wait(pool.submit(r).id);
    CHECK(c.status == read_status::ok && c.bytes == r.len);
    CHECK(memcmp(buf.data(), f.data.data(), r.len) == 0);

    // tail: aligned length past EOF; short read = error unless allowed
    r.offset = 1u << 20;
    r.len    = (1000 + d.alignment - 1) & ~(d.alignment - 1);
    c        = pool.wait(pool.submit(r).id);
    CHECK(c.status == read_status::error && c.bytes == 1000);
    r.allow_partial = true;
    c               = pool.wait(pool.submit(r).id);
    CHECK(c.status == read_status::ok && c.bytes == 1000);
    CHECK(memcmp(buf.data(), f.data.data() + (1u << 20), 1000) == 0);
    close(d.fd);
}

#endif

int main() {
    test_rule();
    test_aligned_buffer();
#ifndef _WIN32
    test_probe();
    test_pool_random_reads();
    test_pool_errors();
    test_pool_cancel_shutdown();
    test_direct_reads();
#endif
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all ok\n");
    return 0;
}
