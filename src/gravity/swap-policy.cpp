#include "swap-policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace gravity {

usage_tracker::usage_tracker(int n_layers, int n_experts, usage_config cfg)
    : n_layers_(std::max(n_layers, 0)), n_experts_(std::max(n_experts, 0)), cfg_(cfg) {
    cells_.resize((size_t) n_layers_ * (size_t) n_experts_);
}

// Decay is applied lazily: raw * decay^(epochs since write). One record then N steps
// equals w * pow(decay, N), and step() stays O(1).
static double decayed(double raw, uint64_t age, double decay) {
    return age == 0 ? raw : raw * std::pow(decay, (double) age);
}

bool usage_tracker::record(int layer, int expert, double weight) {
    if (!valid(layer, expert) || !std::isfinite(weight)) {
        return false;
    }
    cell & c = cells_[idx(layer, expert)];
    c.raw    = decayed(c.raw, epoch_ - c.epoch, cfg_.decay) + weight;
    c.epoch  = epoch_;
    return true;
}

void usage_tracker::step() {
    epoch_++;
}

double usage_tracker::use(int layer, int expert) const {
    if (!valid(layer, expert)) {
        return 0.0;
    }
    const cell & c = cells_[idx(layer, expert)];
    return decayed(c.raw, epoch_ - c.epoch, cfg_.decay);
}

swap_decision usage_tracker::should_swap(int layer, int candidate, const std::vector<int> & resident) const {
    swap_decision d;
    if (!valid(layer, candidate)) {
        d.error = "layer or candidate out of range";
        return d;
    }
    for (int e : resident) {
        if (!valid(layer, e)) {
            d.error = "resident expert out of range";
            return d;
        }
    }
    if (resident.empty() || std::find(resident.begin(), resident.end(), candidate) != resident.end()) {
        return d;
    }

    int    victim     = -1;
    double victim_use = 0.0;
    for (int e : resident) {
        const double u = use(layer, e);
        if (victim < 0 || u < victim_use || (u == victim_use && e < victim)) {
            victim     = e;
            victim_use = u;
        }
    }

    const double cu = use(layer, candidate);
    if (cu >= cfg_.swap_min_use && cu >= cfg_.swap_ratio * victim_use) {
        d.swap   = true;
        d.victim = victim;
    }
    return d;
}

ranking usage_tracker::top_k(int k) const {
    ranking r;
    r.n_layers  = n_layers_;
    r.n_experts = n_experts_;
    r.k         = std::clamp(k, 0, n_experts_);
    r.experts.resize((size_t) n_layers_);

    std::vector<std::pair<double, int>> row((size_t) n_experts_);
    for (int l = 0; l < n_layers_; l++) {
        for (int e = 0; e < n_experts_; e++) {
            row[(size_t) e] = { use(l, e), e };
        }
        std::sort(row.begin(), row.end(), [](const auto & a, const auto & b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        for (int i = 0; i < r.k; i++) {
            r.experts[(size_t) l].push_back(row[(size_t) i].second);
        }
    }
    return r;
}

// ---- ranking file ----
// Layout, all little endian: magic[8] u32 version u32 sha_len sha[sha_len]
// u32 n_layers u32 n_experts u32 k u32 ids[n_layers*k] u64 fnv1a64(all bytes before it)

static const char     RANK_MAGIC[8] = { 'G', 'R', 'V', 'R', 'A', 'N', 'K', '\0' };
static const uint32_t RANK_VERSION  = 1;
static const size_t   RANK_MAX_SHA  = 256;
static const size_t   RANK_MAX_FILE = (size_t) 256 << 20;
static const uint32_t RANK_MAX_LAYERS = 1u << 16;

static uint64_t fnv1a64(const char * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char) p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void put_u32(std::string & s, uint32_t v) {
    for (int i = 0; i < 4; i++) {
        s.push_back((char) ((v >> (8 * i)) & 0xff));
    }
}

static void put_u64(std::string & s, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        s.push_back((char) ((v >> (8 * i)) & 0xff));
    }
}

static uint32_t get_u32(const std::string & s, size_t off) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        v |= (uint32_t) (unsigned char) s[off + (size_t) i] << (8 * i);
    }
    return v;
}

static uint64_t get_u64(const std::string & s, size_t off) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= (uint64_t) (unsigned char) s[off + (size_t) i] << (8 * i);
    }
    return v;
}

static ranking_result fail(std::string msg) {
    ranking_result r;
    r.error = std::move(msg);
    return r;
}

ranking_result save_ranking(const std::string & path, const std::string & model_sha256, const ranking & r) {
    if (model_sha256.empty() || model_sha256.size() > RANK_MAX_SHA) {
        return fail("bad model sha256 length");
    }
    if (r.n_layers < 0 || r.n_experts < 0 || r.k < 0 || r.k > r.n_experts || r.experts.size() != (size_t) r.n_layers) {
        return fail("inconsistent ranking shape");
    }
    for (const auto & row : r.experts) {
        if (row.size() != (size_t) r.k) {
            return fail("ranking row size differs from k");
        }
        for (int e : row) {
            if (e < 0 || e >= r.n_experts) {
                return fail("ranking expert id out of range");
            }
        }
    }

    std::string buf(RANK_MAGIC, sizeof(RANK_MAGIC));
    put_u32(buf, RANK_VERSION);
    put_u32(buf, (uint32_t) model_sha256.size());
    buf += model_sha256;
    put_u32(buf, (uint32_t) r.n_layers);
    put_u32(buf, (uint32_t) r.n_experts);
    put_u32(buf, (uint32_t) r.k);
    for (const auto & row : r.experts) {
        for (int e : row) {
            put_u32(buf, (uint32_t) e);
        }
    }
    put_u64(buf, fnv1a64(buf.data(), buf.size()));

    // Write next to the target so rename stays on one filesystem and readers never see a partial file.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return fail("cannot open " + tmp);
        }
        f.write(buf.data(), (std::streamsize) buf.size());
        f.flush();
        if (!f) {
            f.close();
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return fail("write failed for " + tmp);
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return fail("rename failed for " + path);
    }

    ranking_result ok;
    ok.ok = true;
    return ok;
}

ranking_result load_ranking(const std::string & path, const std::string & model_sha256,
                            int expected_layers, int expected_experts) {
    std::error_code ec;
    const auto      fsize = std::filesystem::file_size(path, ec);
    if (ec) {
        return fail("cannot stat " + path);
    }
    if (fsize > RANK_MAX_FILE) {
        return fail("ranking file too large");
    }
    std::string buf;
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return fail("cannot open " + path);
        }
        buf.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // fixed part: magic + version + sha_len, then sha, then 3 counts; 8 byte checksum last
    size_t off = 0;
    if (buf.size() < sizeof(RANK_MAGIC) + 8 || std::string(buf, 0, sizeof(RANK_MAGIC)) != std::string(RANK_MAGIC, sizeof(RANK_MAGIC))) {
        return fail("bad magic or truncated file");
    }
    off += sizeof(RANK_MAGIC);
    if (get_u32(buf, off) != RANK_VERSION) {
        return fail("unsupported ranking version");
    }
    off += 4;
    const uint32_t sha_len = get_u32(buf, off);
    off += 4;
    if (sha_len == 0 || sha_len > RANK_MAX_SHA || (uint64_t) off + sha_len + 12 + 8 > buf.size()) {
        return fail("bad sha length or truncated file");
    }
    if (buf.compare(off, sha_len, model_sha256) != 0 || sha_len != model_sha256.size()) {
        return fail("model sha256 mismatch");
    }
    off += sha_len;
    const uint32_t nl = get_u32(buf, off);
    const uint32_t ne = get_u32(buf, off + 4);
    const uint32_t k  = get_u32(buf, off + 8);
    off += 12;
    // nl cap: with k == 0 the file stays tiny but experts.assign(nl) would not
    if (nl > RANK_MAX_LAYERS || ne > (uint32_t) INT32_MAX || k > ne) {
        return fail("bad ranking shape");
    }
    if ((expected_layers >= 0 && (uint32_t) expected_layers != nl) ||
        (expected_experts >= 0 && (uint32_t) expected_experts != ne)) {
        return fail("ranking shape mismatch");
    }

    // 64 bit math: nl and k come from the file and must not overflow the size check
    const uint64_t n_ids = (uint64_t) nl * k;
    if (n_ids > RANK_MAX_FILE / 4) {
        return fail("ranking too large");
    }
    const uint64_t want = (uint64_t) off + n_ids * 4 + 8;
    if (buf.size() < want) {
        return fail("truncated file");
    }
    if (buf.size() > want) {
        return fail("trailing bytes");
    }
    if (get_u64(buf, buf.size() - 8) != fnv1a64(buf.data(), buf.size() - 8)) {
        return fail("bad checksum");
    }

    ranking_result res;
    res.value.n_layers  = (int) nl;
    res.value.n_experts = (int) ne;
    res.value.k         = (int) k;
    res.value.experts.assign(nl, {});
    for (uint32_t l = 0; l < nl; l++) {
        for (uint32_t i = 0; i < k; i++) {
            const uint32_t e = get_u32(buf, off);
            off += 4;
            if (e >= ne) {
                return fail("expert id out of range");
            }
            res.value.experts[l].push_back((int) e);
        }
    }
    res.ok = true;
    return res;
}

} // namespace gravity
