// gravity streaming test: mem-budget (S1)
#include "mem-budget.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

using namespace gravity;

static int g_fail = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++;                                                            \
        }                                                                        \
    } while (0)

static const uint64_t GiB = MEM_GIB;

static mem_info make_info(uint64_t total, uint64_t avail) {
    mem_info m;
    m.total     = total;
    m.available = avail;
    m.source    = "test";
    return m;
}

static void test_meminfo() {
    auto r = parse_meminfo("MemTotal:       16384 kB\nMemFree:  1 kB\nMemAvailable:    8192 kB\n");
    CHECK(r.ok());
    CHECK(r.value.total == 16384ull * 1024);
    CHECK(r.value.available == 8192ull * 1024);
    CHECK(!parse_meminfo("").ok());
    CHECK(!parse_meminfo("MemTotal: 100 kB\n").ok());                          // no MemAvailable
    CHECK(!parse_meminfo("MemTotal: abc kB\nMemAvailable: 1 kB\n").ok());      // not a number
    CHECK(!parse_meminfo("MemTotal: 100 MB\nMemAvailable: 1 kB\n").ok());      // unit
    CHECK(!parse_meminfo("MemTotal: 99999999999999999999 kB\nMemAvailable: 1 kB\n").ok()); // overflow
    CHECK(!parse_meminfo("MemTotal:\nMemAvailable: 1 kB\n").ok());
    CHECK(!parse_meminfo(std::string("\0\xff garbage\n", 12)).ok());
}

static void test_cgroup() {
    auto v2 = parse_cgroup_v2("max\n", "12345\n");
    CHECK(v2.ok() && v2.value.limit == 0 && v2.value.used == 12345);
    v2 = parse_cgroup_v2("4294967296\n", "1000\n");
    CHECK(v2.ok() && v2.value.limit == 4294967296ull && v2.value.used == 1000);
    CHECK(!parse_cgroup_v2("lots", "1").ok());
    CHECK(!parse_cgroup_v2("max", "").ok());
    CHECK(!parse_cgroup_v2("-5", "1").ok());

    auto v1 = parse_cgroup_v1("9223372036854771712\n", "500\n"); // typical unlimited sentinel
    CHECK(v1.ok() && v1.value.limit == 0 && v1.value.used == 500);
    v1 = parse_cgroup_v1("2147483648", "100");
    CHECK(v1.ok() && v1.value.limit == 2147483648ull);
    CHECK(!parse_cgroup_v1("", "1").ok());
    CHECK(!parse_cgroup_v1("1", "x").ok());
}

static void test_zfs() {
    const std::string hdr = "13 1 0x01 123 33456 1000 2000\nname type data\n";
    auto r = parse_zfs_arcstats(hdr + "hits 4 77\nsize 4 8000\nc_min 4 3000\nc_max 4 9000\n");
    CHECK(r.ok() && r.value == 5000);
    r = parse_zfs_arcstats(hdr + "size 4 1000\nc_min 4 3000\n"); // c_min > size
    CHECK(r.ok() && r.value == 0);
    CHECK(!parse_zfs_arcstats("").ok());
    CHECK(!parse_zfs_arcstats(hdr + "size 4 1000\n").ok());
    CHECK(!parse_zfs_arcstats(hdr + "size 4 xyz\nc_min 4 1\n").ok());
}

static void test_macos_counts() {
    vm_counts c;
    c.page_size         = 16384;
    c.free_pages        = 100;
    c.inactive_pages    = 200;
    c.purgeable_pages   = 30;
    c.speculative_pages = 20;
    c.memsize           = 18 * GiB;
    auto m              = mem_info_from_vm_counts(c);
    CHECK(m.total == 18 * GiB);
    CHECK(m.available == 350ull * 16384);
    CHECK(m.source == "macos");
    c.free_pages = 1ull << 40; // overlapping classes must not exceed the machine
    m = mem_info_from_vm_counts(c);
    CHECK(m.available == m.total);
}

static file_reader fake_fs(const std::map<std::string, std::string> & files) {
    return [files](const std::string & p, std::string & out) {
        auto it = files.find(p);
        if (it == files.end()) {
            return false;
        }
        out = it->second;
        return true;
    };
}

static void test_linux_assembly() {
    const std::string mi = "MemTotal: 33554432 kB\nMemAvailable: 16777216 kB\n";
    {
        // v2, nested: parent is tighter than own cgroup, no zfs
        auto r = read_linux_mem_info(fake_fs({
            { "/proc/meminfo", mi },
            { "/proc/self/cgroup", "0::/a/b\n" },
            { "/sys/fs/cgroup/a/b/memory.max", "max\n" },
            { "/sys/fs/cgroup/a/b/memory.current", "100\n" },
            { "/sys/fs/cgroup/a/memory.max", "8589934592\n" },
            { "/sys/fs/cgroup/a/memory.current", "1000\n" },
        }));
        CHECK(r.ok());
        CHECK(r.value.total == 32 * GiB);
        CHECK(r.value.available == 16 * GiB);
        CHECK(r.value.cgroup_limit == 8 * GiB && r.value.cgroup_used == 1000);
        CHECK(r.value.zfs_arc_reclaimable == 0);
        CHECK(r.value.note.find("zfs") != std::string::npos);
        CHECK(effective_available(r.value) == 8 * GiB - 1000); // cgroup tighter
    }
    {
        // v1 + zfs
        auto r = read_linux_mem_info(fake_fs({
            { "/proc/meminfo", mi },
            { "/proc/self/cgroup", "12:cpu,cpuacct:/x\n4:memory:/x\n" },
            { "/sys/fs/cgroup/memory/x/memory.limit_in_bytes", "2147483648\n" },
            { "/sys/fs/cgroup/memory/x/memory.usage_in_bytes", "1073741824\n" },
            { "/proc/spl/kstat/zfs/arcstats", "1 1 0x01 1 1 1 1\nname type data\nsize 4 5000\nc_min 4 1000\n" },
        }));
        CHECK(r.ok());
        CHECK(r.value.cgroup_limit == 2 * GiB && r.value.cgroup_used == 1 * GiB);
        CHECK(r.value.zfs_arc_reclaimable == 4000);
    }
    {
        // no cgroup files at all, root path
        auto r = read_linux_mem_info(fake_fs({ { "/proc/meminfo", mi }, { "/proc/self/cgroup", "0::/\n" } }));
        CHECK(r.ok() && r.value.cgroup_limit == 0);
        CHECK(!r.value.note.empty());
    }
    {
        // container root: limit sits directly in /sys/fs/cgroup
        auto r = read_linux_mem_info(fake_fs({ { "/proc/meminfo", mi },
                                               { "/proc/self/cgroup", "0::/\n" },
                                               { "/sys/fs/cgroup/memory.max", "1073741824" },
                                               { "/sys/fs/cgroup/memory.current", "5" } }));
        CHECK(r.ok() && r.value.cgroup_limit == 1 * GiB);
    }
    CHECK(!read_linux_mem_info(fake_fs({})).ok());                          // no meminfo
    CHECK(!read_linux_mem_info(fake_fs({ { "/proc/meminfo", "junk" } })).ok());
    {
        // bad zfs text is a note, not an error
        auto r = read_linux_mem_info(fake_fs({ { "/proc/meminfo", mi }, { "/proc/spl/kstat/zfs/arcstats", "junk" } }));
        CHECK(r.ok() && r.value.zfs_arc_reclaimable == 0);
    }
}

static void test_effective_and_keep() {
    mem_info m = make_info(32 * GiB, 10 * GiB);
    CHECK(effective_available(m) == 10 * GiB);
    m.zfs_arc_reclaimable = 2 * GiB;
    CHECK(effective_available(m) == 12 * GiB);
    m.cgroup_limit = 8 * GiB;
    m.cgroup_used  = 3 * GiB;
    CHECK(effective_available(m) == 5 * GiB); // cgroup tighter
    m.cgroup_limit = 100 * GiB;
    m.cgroup_used  = 1 * GiB;
    CHECK(effective_available(m) == 12 * GiB); // host tighter
    m.cgroup_limit = 1 * GiB;
    m.cgroup_used  = 2 * GiB; // over limit
    CHECK(effective_available(m) == 0);

    budget_cfg cfg;
    CHECK(keep_for_os(18 * GiB, cfg) == 7 * GiB); // 15% = 2.7 GiB < 7
    const uint64_t k128 = keep_for_os(128 * GiB, cfg);
    const uint64_t want = (uint64_t) (19.2 * (double) GiB);
    CHECK(k128 + 2 >= want && k128 <= want + 2);

    mem_info a = make_info(18 * GiB, 12 * GiB);
    CHECK(compute_budget(a, cfg) == 5 * GiB);
    a.available = 6 * GiB;
    CHECK(compute_budget(a, cfg) == 0); // below keep, no underflow
    mem_info c = make_info(64 * GiB, 60 * GiB);
    c.cgroup_limit = 16 * GiB; // keep uses cgroup size, not the host
    CHECK(compute_budget(c, cfg) == 16 * GiB - keep_for_os(16 * GiB, cfg));
}

static void test_stepper() {
    budget_cfg cfg;
    cfg.max_bytes = 0;
    const uint64_t step = cfg.step_bytes;
    CHECK(step == 3 * GiB / 2);

    // 18 GiB box, 15 GiB free, keep 7 GiB: room 8 GiB on top of what we hold
    {
        budget_stepper s(cfg);
        uint64_t       held = 0;
        const uint64_t free0 = 15 * GiB;
        int grows = 0;
        step_action a;
        for (int i = 0; i < 20; i++) {
            mem_info m = make_info(18 * GiB, free0 - held);
            a = s.next_action(held, m);
            if (a.kind != step_action::grow) {
                break;
            }
            CHECK(a.bytes == step);
            held += a.bytes;
            grows++;
        }
        CHECK(grows == 5); // 5 * 1.5 = 7.5 GiB; a 6th step would leave < keep
        CHECK(a.kind == step_action::lock_now);
        mem_info m = make_info(18 * GiB, free0 - held);
        CHECK(s.next_action(held, m).kind == step_action::hold); // lock only once
        CHECK(s.next_action(held, m).kind == step_action::hold);
    }
    // user cap: last step is partial, then lock
    {
        cfg.max_bytes = 4 * GiB;
        budget_stepper s(cfg);
        uint64_t held = 0;
        std::vector<uint64_t> got;
        for (int i = 0; i < 10; i++) {
            auto a = s.next_action(held, make_info(128 * GiB, 100 * GiB - held));
            if (a.kind != step_action::grow) {
                CHECK(a.kind == step_action::lock_now);
                break;
            }
            got.push_back(a.bytes);
            held += a.bytes;
        }
        CHECK(held == 4 * GiB);
        CHECK(got.size() == 3 && got[2] == 1 * GiB);
        cfg.max_bytes = 0;
    }
    // pressure: shrink one step, then cooldown holds, then grow again
    {
        budget_stepper s(cfg);
        uint64_t held = 6 * GiB;
        auto a = s.next_action(held, make_info(18 * GiB, 6 * GiB)); // below keep 7
        CHECK(a.kind == step_action::shrink && a.bytes == step);
        held -= a.bytes;
        for (uint32_t i = 0; i < cfg.cooldown_steps; i++) {
            a = s.next_action(held, make_info(18 * GiB, 20 * GiB));
            CHECK(a.kind == step_action::hold);
        }
        a = s.next_action(held, make_info(18 * GiB, 20 * GiB));
        CHECK(a.kind == step_action::grow);
        // pressure again during cooldown still shrinks, and shrink never exceeds what we hold
        budget_stepper s2(cfg);
        a = s2.next_action(1 * GiB, make_info(18 * GiB, 1 * GiB));
        CHECK(a.kind == step_action::shrink && a.bytes == 1 * GiB);
        a = s2.next_action(0, make_info(18 * GiB, 1 * GiB));
        CHECK(a.kind == step_action::hold); // nothing to give back
        a = s2.next_action(0, make_info(18 * GiB, 20 * GiB));
        CHECK(a.kind == step_action::hold); // cooldown
    }
    // lock again after pressure released and refilled
    {
        budget_stepper s(cfg);
        cfg.max_bytes = 0;
        CHECK(s.next_action(3 * GiB, make_info(18 * GiB, 8 * GiB)).kind == step_action::lock_now); // room 1 GiB < step
        CHECK(s.next_action(3 * GiB, make_info(18 * GiB, 8 * GiB)).kind == step_action::hold);
        CHECK(s.next_action(3 * GiB, make_info(18 * GiB, 5 * GiB)).kind == step_action::shrink);
    }
}

static void test_host_smoke() {
    auto r = read_mem_info();
    if (!r.ok()) {
        std::fprintf(stderr, "read_mem_info: %s\n", r.error.c_str());
        g_fail++;
        return;
    }
    CHECK(r.value.total > 0);
    CHECK(r.value.available <= r.value.total);
    CHECK(!r.value.source.empty());
#if defined(__APPLE__)
    uint64_t memsize = 0;
    size_t   len     = sizeof(memsize);
    CHECK(sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0);
    CHECK(r.value.total == memsize);
#endif
}

int main() {
    test_meminfo();
    test_cgroup();
    test_zfs();
    test_macos_counts();
    test_linux_assembly();
    test_effective_and_keep();
    test_stepper();
    test_host_smoke();
    if (g_fail) {
        std::fprintf(stderr, "%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
