// gravity streaming test: slot ring and demand-over-prefetch scheduler

#include "read-pool.h"
#include "scheduler.h"
#include "slot-ring.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#    include <unistd.h>
#endif

using namespace gravity;

static int n_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++n_fail;                                                       \
        }                                                                   \
    } while (0)

static expert_key K(uint32_t l, uint32_t e) {
    expert_key k;
    k.layer  = l;
    k.expert = e;
    return k;
}

static void test_ring_lru() {
    slot_ring ring(3, 64, 64);
    CHECK(ring.ok());
    CHECK(ring.n_free() == 3);
    for (uint32_t e = 0; e < 3; ++e) {
        acquire_result a = ring.acquire(K(0, e));
        CHECK(a.ok() && !a.hit && a.needs_fill && a.slot == (int) e);
        CHECK(ring.mark_ready(a.slot));
        CHECK(ring.unpin(a.slot));
    }
    CHECK(ring.n_free() == 0);

    // touch 0 -> LRU order 1, 2, 0
    acquire_result h = ring.acquire(K(0, 0));
    CHECK(h.ok() && h.hit && h.ready && !h.needs_fill && h.slot == 0);
    CHECK(ring.unpin(h.slot));

    acquire_result d = ring.acquire(K(0, 3));
    CHECK(d.ok() && d.needs_fill && d.slot == 1);
    CHECK(ring.lookup(K(0, 1)) == -1);
    ring.mark_ready(d.slot);
    ring.unpin(d.slot);

    acquire_result e = ring.acquire(K(0, 4));
    CHECK(e.ok() && e.slot == 2);
    CHECK(ring.lookup(K(0, 2)) == -1);
    ring.unpin(e.slot);

    acquire_result f = ring.acquire(K(0, 5));
    CHECK(f.ok() && f.slot == 0);
    ring.unpin(f.slot);

    CHECK(!ring.unpin(0));  // not pinned
    CHECK(!ring.unpin(7));  // bad slot
    CHECK(!ring.pin(-1));
    CHECK(ring.data(9) == nullptr);
    CHECK(((uintptr_t) ring.data(0) % 64) == 0);
}

static void test_ring_pinned() {
    slot_ring      ring(2, 64, 16);
    acquire_result a = ring.acquire(K(1, 0));  // stays pinned
    acquire_result b = ring.acquire(K(1, 1));
    ring.mark_ready(a.slot);
    ring.mark_ready(b.slot);
    ring.unpin(b.slot);

    acquire_result c = ring.acquire(K(1, 2));
    CHECK(c.ok() && c.slot == b.slot);

    // all pinned -> error, does not block
    acquire_result d = ring.acquire(K(1, 3));
    CHECK(!d.ok() && !d.error.empty());

    ring.unpin(c.slot);
    for (int i = 0; i < 10; ++i) {
        acquire_result x = ring.acquire(K(2, (uint32_t) i));
        CHECK(x.ok() && x.slot != a.slot);
        ring.unpin(x.slot);
    }
    CHECK(ring.lookup(K(1, 0)) == a.slot);
    CHECK(ring.is_ready(a.slot));

    // pin refcount: two pins need two unpins before eviction
    CHECK(ring.pin(a.slot));
    CHECK(ring.pins(a.slot) == 2);
    ring.unpin(a.slot);
    CHECK(ring.pins(a.slot) == 1);
    ring.unpin(a.slot);
    CHECK(ring.pins(a.slot) == 0);

    // failed fill frees the slot once unpinned
    slot_ring      r2(2, 64, 16);
    acquire_result f = r2.acquire(K(3, 0));
    CHECK(r2.n_free() == 1);
    CHECK(r2.fail_fill(f.slot));
    CHECK(r2.lookup(K(3, 0)) == -1);
    CHECK(r2.n_free() == 1);
    r2.unpin(f.slot);
    CHECK(r2.n_free() == 2);

    slot_ring bad(0, 64, 16);
    CHECK(!bad.ok());
    CHECK(!bad.acquire(K(0, 0)).ok());
    slot_ring bad_align(2, 64, 3);
    CHECK(!bad_align.ok());
}

#ifndef _WIN32

static const uint32_t n_layers  = 3;
static const uint32_t n_experts = 64;
static const size_t   max_span  = 40000;

struct synth_file {
    std::string                       path;
    int                               fd = -1;
    std::vector<uint8_t>              data;
    std::map<expert_key, expert_span> spans;

    synth_file() {
        char tmpl[] = "./gravity-sched-test-XXXXXX";
        fd          = mkstemp(tmpl);
        if (fd < 0) {
            fprintf(stderr, "mkstemp failed: %s\n", strerror(errno));
            return;
        }
        path = tmpl;
        std::mt19937 rng(1234);
        uint64_t     off = 0;
        for (uint32_t l = 0; l < n_layers; ++l) {
            for (uint32_t e = 0; e < n_experts; ++e) {
                off += rng() % 5000;  // gap, unaligned starts
                expert_span sp;
                sp.offset = off;
                // mix of tiny, exact chunk multiples and odd sizes
                switch (rng() % 4) {
                    case 0:
                        sp.len = 1 + rng() % 100;
                        break;
                    case 1:
                        sp.len = 4096 * (1 + rng() % 8);
                        break;
                    default:
                        sp.len = 1 + rng() % max_span;
                        break;
                }
                spans[K(l, e)] = sp;
                off += sp.len;
            }
        }
        data.resize(off + 1000);
        for (auto & b : data) {
            b = (uint8_t) rng();
        }
        size_t done = 0;
        while (done < data.size()) {
            ssize_t w = write(fd, data.data() + done, data.size() - done);
            if (w <= 0) {
                fprintf(stderr, "write failed\n");
                close(fd);
                fd = -1;
                return;
            }
            done += (size_t) w;
        }
    }

    ~synth_file() {
        if (fd >= 0) {
            close(fd);
        }
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }

    locate_fn locator() const {
        return [this](const expert_key & k, expert_span & out) {
            auto it = spans.find(k);
            if (it == spans.end()) {
                return false;
            }
            out = it->second;
            return true;
        };
    }

    bool same(const expert_key & k, const demand_result & r) const {
        auto it = spans.find(k);
        return it != spans.end() && r.ok() && r.len == it->second.len &&
               memcmp(r.data, data.data() + it->second.offset, r.len) == 0;
    }
};

// counts reads per file offset
struct counting_read {
    std::mutex              mtx;
    std::map<uint64_t, int> by_off;
    std::atomic<int>        total{ 0 };

    read_fn fn() {
        return [this](const read_req & req) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                by_off[req.offset]++;
            }
            total++;
            return read_full(req);
        };
    }

    int at(uint64_t off) {
        std::lock_guard<std::mutex> lock(mtx);
        return by_off.count(off) ? by_off[off] : 0;
    }
};

// a read at hold_off blocks until open()
struct gate {
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    is_open = false;
    std::atomic<int>        entered{ 0 };

    void wait_open() {
        entered++;
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return is_open; });
    }

    void open() {
        std::lock_guard<std::mutex> lock(mtx);
        is_open = true;
        cv.notify_all();
    }

    bool wait_entered(int n) {
        for (int i = 0; i < 2000 && entered.load() < n; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return entered.load() >= n;
    }
};

static size_t slot_bytes_for(size_t align) {
    return max_span + 2 * (align > 0 ? align : 1);
}

static void test_demand_exact(const synth_file & sf, size_t align) {
    read_pool     pool(2);
    slot_ring     ring(8, slot_bytes_for(align), 4096);
    scheduler_cfg cfg;
    cfg.chunk = 4096;
    scheduler s(pool, ring, sf.fd, align, cfg, sf.locator());
    for (const auto & kv : sf.spans) {
        demand_result r = s.demand(kv.first);
        if (!sf.same(kv.first, r)) {
            fprintf(stderr, "  demand %u/%u align %zu: %s\n", kv.first.layer, kv.first.expert, align, r.error.c_str());
        }
        CHECK(sf.same(kv.first, r));
        CHECK(s.release(r.slot));
    }
    // hit path
    expert_key    k  = sf.spans.rbegin()->first;
    demand_result r1 = s.demand(k);
    CHECK(sf.same(k, r1));
    s.release(r1.slot);
    CHECK(s.stats().demand_hits >= 1);

    // unknown key, span too big for slot
    CHECK(!s.demand(K(99, 0)).ok());
    std::map<expert_key, expert_span> big;
    big[K(0, 0)] = { 0, ring.slot_bytes() + 1 };
    scheduler     s2(pool, ring, sf.fd, align, cfg, [&](const expert_key & key, expert_span & out) {
        auto it = big.find(key);
        if (it == big.end()) {
            return false;
        }
        out = it->second;
        return true;
    });
    demand_result rb = s2.demand(K(0, 0));
    CHECK(!rb.ok() && !rb.error.empty());
    prefetch_result pb = s2.prefetch({ K(0, 0), K(5, 5) });
    CHECK(pb.queued == 0 && !pb.error.empty());
}

static void test_concurrent(const synth_file & sf, size_t align) {
    counting_read cr;
    read_pool     pool(4, cr.fn());
    slot_ring     ring(24, slot_bytes_for(align), 4096);
    scheduler_cfg cfg;
    cfg.chunk                 = 4096;
    cfg.max_prefetch_inflight = 4;
    scheduler s(pool, ring, sf.fd, align, cfg, sf.locator());

    std::atomic<int>         bad{ 0 };
    std::atomic<int>         running{ 4 };
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t) {
        th.emplace_back([&, t] {
            std::mt19937 rng(100 + t);
            for (int i = 0; i < 1000; ++i) {
                expert_key    k = K(rng() % n_layers, rng() % n_experts);
                demand_result r = s.demand(k);
                if (!sf.same(k, r)) {
                    if (bad.load() < 5) {
                        fprintf(stderr, "  thread %d demand %u/%u: %s\n", t, k.layer, k.expert, r.error.c_str());
                    }
                    bad++;
                }
                if (r.ok()) {
                    s.release(r.slot);
                }
            }
            running--;
        });
    }
    std::thread pf([&] {
        std::mt19937 rng(7);
        while (running.load() > 0) {
            std::vector<expert_key> keys;
            for (int i = 0; i < 6; ++i) {
                keys.push_back(K(rng() % n_layers, rng() % n_experts));
            }
            s.prefetch(keys);
            if (rng() % 3 == 0) {
                s.step();
            }
            if (rng() % 5 == 0) {
                s.drop_prefetch();
            }
            std::this_thread::yield();
        }
    });
    for (auto & t : th) {
        t.join();
    }
    pf.join();
    CHECK(bad.load() == 0);
    step_result sr = s.step();
    CHECK(sr.errors.empty());
    scheduler_stats st = s.stats();
    printf("  concurrent align %zu: reads %llu hits %llu joined %llu misses %llu prefetched %llu dropped %llu\n", align,
           (unsigned long long) st.reads, (unsigned long long) st.demand_hits, (unsigned long long) st.demand_joined,
           (unsigned long long) st.demand_misses, (unsigned long long) st.prefetch_filled,
           (unsigned long long) st.prefetch_dropped);
    CHECK(st.demand_hits + st.demand_joined + st.demand_misses == 4000);
    // started prefetch fills still finish in the background; wait for them, then every submitted read ran once
    s.drop_prefetch();
    for (int i = 0; i < 5000 && (uint64_t) cr.total.load() != s.stats().reads; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK((uint64_t) cr.total.load() == s.stats().reads);
}

static void test_join_inflight(const synth_file & sf) {
    expert_key  k  = K(1, 7);
    expert_span sp = sf.spans.at(k);

    gate          g;
    counting_read cr;
    read_fn       inner = cr.fn();
    read_pool     pool(2, [&](const read_req & req) {
        if (req.offset == sp.offset) {
            g.wait_open();
        }
        return inner(req);
    });
    slot_ring     ring(4, slot_bytes_for(0), 4096);
    scheduler_cfg cfg;
    cfg.chunk = 1 << 20;  // one chunk per expert
    scheduler s(pool, ring, sf.fd, 0, cfg, sf.locator());

    CHECK(s.prefetch({ k }).queued == 1);
    CHECK(g.wait_entered(1));

    demand_result r;
    std::thread   d([&] { r = s.demand(k); });
    for (int i = 0; i < 2000 && s.stats().demand_joined == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(s.stats().demand_joined == 1);
    // another key demanded meanwhile is not stuck behind the held read
    demand_result o = s.demand(K(0, 3));
    CHECK(sf.same(K(0, 3), o));
    s.release(o.slot);
    g.open();
    d.join();
    CHECK(sf.same(k, r));
    CHECK(cr.at(sp.offset) == 1);
    s.release(r.slot);
    // prefetch of a ready key is a no-op
    CHECK(s.prefetch({ k }).queued == 0);
}

static void test_stale_drop(const synth_file & sf, bool use_step) {
    expert_key     a = K(0, 1), b = K(0, 2), c = K(0, 3), p = K(2, 9);
    gate           g;
    counting_read  cr;
    read_fn        inner = cr.fn();
    const uint64_t hold  = sf.spans.at(a).offset;
    read_pool      pool(1, [&](const read_req & req) {
        if (req.offset == hold) {
            g.wait_open();
        }
        return inner(req);
    });
    slot_ring      ring(6, slot_bytes_for(0), 4096);
    scheduler_cfg  cfg;
    cfg.chunk                 = 1 << 20;
    cfg.max_prefetch_inflight = 1;
    scheduler s(pool, ring, sf.fd, 0, cfg, sf.locator());

    demand_result pinned = s.demand(p);  // held across the drop
    CHECK(sf.same(p, pinned));
    const size_t free_before = ring.n_free();

    CHECK(s.prefetch({ a, b, c }).queued == 3);
    CHECK(g.wait_entered(1));
    // b and c not started: no slot bound yet
    CHECK(ring.n_free() == free_before - 1);
    if (use_step) {
        step_result sr = s.step();
        CHECK(sr.dropped == 2);
        CHECK(sr.errors.empty());
    } else {
        CHECK(s.drop_prefetch() == 2);
    }
    CHECK(s.drop_prefetch() == 0);
    g.open();

    demand_result ra = s.demand(a);  // started fill runs to the end
    CHECK(sf.same(a, ra));
    s.release(ra.slot);
    CHECK(cr.at(sf.spans.at(a).offset) == 1);
    CHECK(cr.at(sf.spans.at(b).offset) == 0);
    CHECK(cr.at(sf.spans.at(c).offset) == 0);
    CHECK(ring.lookup(b) == -1 && ring.lookup(c) == -1);
    CHECK(ring.pins(pinned.slot) == 1);
    CHECK(sf.same(p, pinned));
    CHECK(s.stats().prefetch_dropped == 2);
    s.release(pinned.slot);

    // dropped key can be demanded later
    demand_result rb = s.demand(b);
    CHECK(sf.same(b, rb));
    s.release(rb.slot);
}

static void test_read_error(const synth_file & sf) {
    expert_key        bad = K(2, 5);
    const uint64_t    off = sf.spans.at(bad).offset;
    std::atomic<bool> broken{ true };
    read_pool         pool(2, [&](const read_req & req) {
        if (broken.load() && req.offset == off) {
            read_completion c;
            c.status = read_status::error;
            c.err    = 5;
            c.error  = "fake EIO";
            return c;
        }
        return read_full(req);
    });
    slot_ring         ring(4, slot_bytes_for(0), 4096);
    scheduler_cfg     cfg;
    cfg.chunk = 4096;
    scheduler s(pool, ring, sf.fd, 0, cfg, sf.locator());

    demand_result r = s.demand(bad);
    CHECK(!r.ok());
    CHECK(r.error.find("fake EIO") != std::string::npos);
    CHECK(ring.lookup(bad) == -1);
    CHECK(ring.n_free() == 4);

    // failed prefetch: error at step, slot free, key not resident
    CHECK(s.prefetch({ bad }).queued == 1);
    step_result sr;
    for (int i = 0; i < 2000; ++i) {
        step_result x = s.step();
        sr.errors.insert(sr.errors.end(), x.errors.begin(), x.errors.end());
        if (!sr.errors.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(sr.errors.size() == 1);
    CHECK(ring.lookup(bad) == -1);
    CHECK(ring.n_free() == 4);

    broken           = false;
    demand_result ok = s.demand(bad);
    CHECK(sf.same(bad, ok));
    s.release(ok.slot);

    // EOF inside the span -> short read error
    std::map<expert_key, expert_span> past;
    past[K(0, 0)] = { sf.data.size() - 10, 100 };
    scheduler     s2(pool, ring, sf.fd, 0, cfg, [&](const expert_key & key, expert_span & out) {
        auto it = past.find(key);
        if (it == past.end()) {
            return false;
        }
        out = it->second;
        return true;
    });
    demand_result re = s2.demand(K(0, 0));
    CHECK(!re.ok());
    CHECK(ring.lookup(K(0, 0)) == -1);
}

static std::vector<std::pair<int, uint64_t>> det_run(const synth_file & sf) {
    std::vector<std::pair<int, uint64_t>> out;
    read_pool                             pool(4);
    slot_ring                             ring(10, slot_bytes_for(4096), 4096);
    scheduler_cfg                         cfg;
    cfg.chunk         = 8192;
    cfg.deterministic = true;
    scheduler        s(pool, ring, sf.fd, 4096, cfg, sf.locator());
    std::mt19937     rng(42);
    std::vector<int> held;
    for (int i = 0; i < 600; ++i) {
        expert_key k = K(rng() % n_layers, rng() % 20);
        CHECK(s.prefetch({ K(rng() % n_layers, rng() % n_experts) }).queued == 0);
        demand_result r = s.demand(k);
        CHECK(sf.same(k, r));
        uint64_t h = 1469598103934665603ull;
        for (size_t j = 0; j < r.len; ++j) {
            h = (h ^ r.data[j]) * 1099511628211ull;
        }
        out.push_back({ r.slot, h });
        held.push_back(r.slot);
        if (held.size() > 3 || rng() % 2) {  // hold a few pins at once
            s.release(held.front());
            held.erase(held.begin());
        }
    }
    for (int sl : held) {
        s.release(sl);
    }
    CHECK(s.stats().reads > 0 && s.stats().prefetch_filled == 0);
    return out;
}

static void test_deterministic(const synth_file & sf) {
    auto a = det_run(sf);
    auto b = det_run(sf);
    CHECK(a.size() == 600);
    CHECK(a == b);
}

#endif

int main() {
    test_ring_lru();
    test_ring_pinned();
#ifndef _WIN32
    synth_file sf;
    if (sf.fd < 0) {
        fprintf(stderr, "could not create test file\n");
        return 1;
    }
    test_demand_exact(sf, 0);
    test_demand_exact(sf, 4096);
    test_concurrent(sf, 0);
    test_concurrent(sf, 4096);
    test_join_inflight(sf);
    test_stale_drop(sf, true);
    test_stale_drop(sf, false);
    test_read_error(sf);
    test_deterministic(sf);
#endif
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all gravity scheduler tests passed\n");
    return 0;
}
