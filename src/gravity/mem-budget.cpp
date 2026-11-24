#include "mem-budget.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace gravity {

namespace {

constexpr uint64_t U64_MAX = ~0ull;
// cgroup v1 "no limit" is ~2^63 rounded to a page; anything this big is unlimited
constexpr uint64_t CGROUP_V1_UNLIMITED = 1ull << 62;

std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) {
        b++;
    }
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        e--;
    }
    return s.substr(b, e - b);
}

// strict decimal: digits only, no sign, no overflow
bool parse_u64(const std::string & text, uint64_t & out) {
    const std::string s = trim(text);
    if (s.empty()) {
        return false;
    }
    uint64_t v = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        const uint64_t d = (uint64_t) (ch - '0');
        if (v > (U64_MAX - d) / 10) {
            return false;
        }
        v = v * 10 + d;
    }
    out = v;
    return true;
}

std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::istringstream       in(text);
    std::string              line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

std::vector<std::string> split_ws(const std::string & line) {
    std::vector<std::string> out;
    std::istringstream       in(line);
    std::string              tok;
    while (in >> tok) {
        out.push_back(tok);
    }
    return out;
}

template <typename T>
mem_result<T> fail(const std::string & msg) {
    mem_result<T> r;
    r.error = msg;
    return r;
}

// "MemTotal:  16384 kB" -> bytes
bool parse_meminfo_line(const std::string & line, const char * key, uint64_t & out, std::string & err) {
    const std::string prefix = std::string(key) + ":";
    if (line.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    const auto tok = split_ws(line.substr(prefix.size()));
    if (tok.empty() || tok.size() > 2 || (tok.size() == 2 && tok[1] != "kB")) {
        err = std::string("meminfo: bad ") + key + " line";
        return true;
    }
    uint64_t v = 0;
    if (!parse_u64(tok[0], v)) {
        err = std::string("meminfo: bad ") + key + " number";
        return true;
    }
    if (tok.size() == 2) {
        if (v > U64_MAX / 1024) {
            err = std::string("meminfo: ") + key + " overflow";
            return true;
        }
        v *= 1024;
    }
    out = v;
    return true;
}

uint64_t headroom(const cgroup_values & c) {
    return c.used >= c.limit ? 0 : c.limit - c.used;
}

std::string dir_parent(const std::string & dir) {
    const size_t pos = dir.find_last_of('/');
    return pos == std::string::npos ? std::string() : dir.substr(0, pos);
}

// cgroup path may be "/", "/a/b" or odd forms; normalise to "" or "/a/b"
std::string norm_cgroup_dir(std::string p) {
    while (!p.empty() && p.back() == '/') {
        p.pop_back();
    }
    if (!p.empty() && p[0] != '/') {
        p = "/" + p;
    }
    return p;
}

// walk from the process cgroup to the root and keep the tightest limit:
// a parent can cap us below our own memory.max
bool tightest_cgroup(const file_reader & read, const std::string & base, std::string dir, const char * limit_file,
                     const char * used_file, bool v2, cgroup_values & best) {
    bool found = false;
    for (;;) {
        std::string lim;
        std::string use;
        if (read(base + dir + "/" + limit_file, lim) && read(base + dir + "/" + used_file, use)) {
            const auto r = v2 ? parse_cgroup_v2(lim, use) : parse_cgroup_v1(lim, use);
            if (r.ok() && r.value.limit != 0 && (!found || headroom(r.value) < headroom(best))) {
                best  = r.value;
                found = true;
            }
        }
        if (dir.empty()) {
            break;
        }
        dir = dir_parent(dir);
    }
    return found;
}

} // namespace

mem_result<meminfo_values> parse_meminfo(const std::string & text) {
    meminfo_values v;
    bool           have_total = false;
    bool           have_avail = false;
    std::string    err;
    for (const auto & line : split_lines(text)) {
        if (parse_meminfo_line(line, "MemTotal", v.total, err)) {
            have_total = err.empty();
        } else if (parse_meminfo_line(line, "MemAvailable", v.available, err)) {
            have_avail = err.empty();
        }
        if (!err.empty()) {
            return fail<meminfo_values>(err);
        }
    }
    if (!have_total) {
        return fail<meminfo_values>("meminfo: MemTotal missing");
    }
    if (!have_avail) {
        return fail<meminfo_values>("meminfo: MemAvailable missing");
    }
    mem_result<meminfo_values> r;
    r.value = v;
    return r;
}

mem_result<cgroup_values> parse_cgroup_v2(const std::string & max_text, const std::string & current_text) {
    cgroup_values v;
    const std::string m = trim(max_text);
    if (m != "max" && !parse_u64(m, v.limit)) {
        return fail<cgroup_values>("cgroup v2: bad memory.max");
    }
    if (!parse_u64(current_text, v.used)) {
        return fail<cgroup_values>("cgroup v2: bad memory.current");
    }
    mem_result<cgroup_values> r;
    r.value = v;
    return r;
}

mem_result<cgroup_values> parse_cgroup_v1(const std::string & limit_text, const std::string & usage_text) {
    cgroup_values v;
    if (!parse_u64(limit_text, v.limit)) {
        return fail<cgroup_values>("cgroup v1: bad limit_in_bytes");
    }
    if (!parse_u64(usage_text, v.used)) {
        return fail<cgroup_values>("cgroup v1: bad usage_in_bytes");
    }
    if (v.limit >= CGROUP_V1_UNLIMITED) {
        v.limit = 0;
    }
    mem_result<cgroup_values> r;
    r.value = v;
    return r;
}

mem_result<uint64_t> parse_zfs_arcstats(const std::string & text) {
    uint64_t size  = 0;
    uint64_t c_min = 0;
    bool     have_size  = false;
    bool     have_c_min = false;
    for (const auto & line : split_lines(text)) {
        // rows are "name type data"; the two header lines never match these names
        const auto tok = split_ws(line);
        if (tok.size() != 3) {
            continue;
        }
        if (tok[0] == "size" || tok[0] == "c_min") {
            uint64_t v = 0;
            if (!parse_u64(tok[2], v)) {
                return fail<uint64_t>("arcstats: bad " + tok[0] + " number");
            }
            (tok[0] == "size" ? size : c_min) = v;
            (tok[0] == "size" ? have_size : have_c_min) = true;
        }
    }
    if (!have_size || !have_c_min) {
        return fail<uint64_t>("arcstats: size or c_min missing");
    }
    mem_result<uint64_t> r;
    r.value = size > c_min ? size - c_min : 0;
    return r;
}

mem_info mem_info_from_vm_counts(const vm_counts & c) {
    mem_info m;
    m.source = "macos";
    m.total  = c.memsize;
    const uint64_t pages = c.free_pages + c.inactive_pages + c.purgeable_pages + c.speculative_pages;
    m.available          = c.page_size != 0 && pages > U64_MAX / c.page_size ? c.memsize : pages * c.page_size;
    // purgeable can overlap other classes; never report more than the machine has
    m.available = std::min(m.available, m.total);
    return m;
}

mem_result<mem_info> read_linux_mem_info(const file_reader & read) {
    std::string text;
    if (!read("/proc/meminfo", text)) {
        return fail<mem_info>("cannot read /proc/meminfo");
    }
    const auto mi = parse_meminfo(text);
    if (!mi.ok()) {
        return fail<mem_info>(mi.error);
    }
    mem_result<mem_info> r;
    mem_info &           m = r.value;
    m.source               = "linux";
    m.total                = mi.value.total;
    m.available            = std::min(mi.value.available, mi.value.total);

    // cgroup: pick v2 path first, else the v1 memory controller
    std::string proc_cg;
    bool        have_cg = false;
    if (read("/proc/self/cgroup", proc_cg)) {
        std::string v2_dir;
        std::string v1_dir;
        bool        has_v2 = false;
        bool        has_v1 = false;
        for (const auto & line : split_lines(proc_cg)) {
            const size_t a = line.find(':');
            const size_t b = a == std::string::npos ? a : line.find(':', a + 1);
            if (b == std::string::npos) {
                continue;
            }
            const std::string ctrl = line.substr(a + 1, b - a - 1);
            const std::string path = norm_cgroup_dir(trim(line.substr(b + 1)));
            if (a == 1 && line[0] == '0' && ctrl.empty()) {
                v2_dir = path;
                has_v2 = true;
            } else if (("," + ctrl + ",").find(",memory,") != std::string::npos) {
                v1_dir = path;
                has_v1 = true;
            }
        }
        cgroup_values cg;
        if (has_v2 && tightest_cgroup(read, "/sys/fs/cgroup", v2_dir, "memory.max", "memory.current", true, cg)) {
            have_cg = true;
        } else if (has_v1 && tightest_cgroup(read, "/sys/fs/cgroup/memory", v1_dir, "memory.limit_in_bytes",
                                             "memory.usage_in_bytes", false, cg)) {
            have_cg = true;
        }
        if (have_cg) {
            m.cgroup_limit = cg.limit;
            m.cgroup_used  = cg.used;
        }
    }
    if (!have_cg) {
        m.note += "no cgroup memory limit found; ";
    }

    std::string arc;
    if (!read("/proc/spl/kstat/zfs/arcstats", arc)) {
        m.note += "no zfs arcstats; ";
    } else {
        const auto z = parse_zfs_arcstats(arc);
        if (z.ok()) {
            m.zfs_arc_reclaimable = z.value;
        } else {
            m.note += z.error + "; ";
        }
    }
    return r;
}

#if defined(__APPLE__)
static mem_result<mem_info> read_macos_mem_info() {
    vm_counts c;
    uint64_t  memsize = 0;
    size_t    len     = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) != 0 || memsize == 0) {
        return fail<mem_info>("sysctl hw.memsize failed");
    }
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS || page == 0) {
        return fail<mem_info>("host_page_size failed");
    }
    vm_statistics64_data_t st;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &st, &cnt) != KERN_SUCCESS) {
        return fail<mem_info>("host_statistics64 failed");
    }
    c.memsize = memsize;
    c.page_size = page;
    // free_count already contains speculative pages
    c.free_pages        = st.free_count >= st.speculative_count ? st.free_count - st.speculative_count : 0;
    c.speculative_pages = st.speculative_count;
    c.inactive_pages    = st.inactive_count;
    c.purgeable_pages   = st.purgeable_count;
    mem_result<mem_info> r;
    r.value = mem_info_from_vm_counts(c);
    return r;
}
#endif

mem_result<mem_info> read_mem_info() {
#if defined(__linux__)
    const file_reader rd = [](const std::string & path, std::string & out) {
        FILE * f = std::fopen(path.c_str(), "r");
        if (!f) {
            return false;
        }
        char   buf[4096];
        size_t n;
        out.clear();
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
            out.append(buf, n);
        }
        std::fclose(f);
        return true;
    };
    return read_linux_mem_info(rd);
#elif defined(__APPLE__)
    return read_macos_mem_info();
#else
    return fail<mem_info>("mem-budget: unsupported OS");
#endif
}

uint64_t effective_available(const mem_info & m) {
    const uint64_t host = m.zfs_arc_reclaimable > U64_MAX - m.available ? U64_MAX : m.available + m.zfs_arc_reclaimable;
    if (m.cgroup_limit == 0) {
        return host;
    }
    return std::min(host, m.cgroup_limit > m.cgroup_used ? m.cgroup_limit - m.cgroup_used : 0);
}

uint64_t keep_for_os(uint64_t total, const budget_cfg & cfg) {
    const uint64_t frac = (uint64_t) ((double) total * cfg.keep_frac);
    return std::max(cfg.keep_min_bytes, frac);
}

// inside a cgroup the machine we live in is the limit, not the host
static uint64_t effective_total(const mem_info & m) {
    return m.cgroup_limit != 0 && m.cgroup_limit < m.total ? m.cgroup_limit : m.total;
}

uint64_t compute_budget(const mem_info & m, const budget_cfg & cfg) {
    const uint64_t keep  = keep_for_os(effective_total(m), cfg);
    const uint64_t avail = effective_available(m);
    return avail > keep ? avail - keep : 0;
}

step_action budget_stepper::next_action(uint64_t current_bytes, const mem_info & m) {
    step_action a;
    const uint64_t keep = keep_for_os(effective_total(m), cfg_);

    if (effective_available(m) < keep) {
        // pressure: give back one step; unlock so a later full state can lock again
        const uint64_t n = std::min(cfg_.step_bytes, current_bytes);
        cooldown_left_   = cfg_.cooldown_steps;
        locked_          = false;
        if (n > 0) {
            a.kind  = step_action::shrink;
            a.bytes = n;
        }
        return a;
    }
    if (cooldown_left_ > 0) {
        cooldown_left_--;
        return a;
    }

    const uint64_t room = compute_budget(m, cfg_);
    const uint64_t cap_left = cfg_.max_bytes == 0 ? U64_MAX : (cfg_.max_bytes > current_bytes ? cfg_.max_bytes - current_bytes : 0);
    const uint64_t want = std::min(cfg_.step_bytes, cap_left);
    // full = user cap reached, or not enough headroom for the next step
    if (want == 0 || room < want) {
        if (!locked_) {
            locked_ = true;
            a.kind  = step_action::lock_now;
        }
        return a;
    }
    locked_ = false;
    a.kind  = step_action::grow;
    a.bytes = want;
    return a;
}

} // namespace gravity
