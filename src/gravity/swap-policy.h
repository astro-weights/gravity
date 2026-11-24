#pragma once

// gravity streaming: expert usage tracking, swap rule, saved ranking

#include <cstdint>
#include <string>
#include <vector>

namespace gravity {

struct usage_config {
    double decay        = 0.7; // factor applied per step()
    double swap_min_use = 2.0; // candidate must reach this decayed use
    double swap_ratio   = 1.5; // candidate must beat victim by this factor
};

struct swap_decision {
    bool        swap   = false;
    int         victim = -1; // lowest-use resident, set only when swap is true
    std::string error;       // set when layer/candidate/resident ids are out of range
};

// top-k expert ids per layer, best first
struct ranking {
    int                           n_layers  = 0;
    int                           n_experts = 0;
    int                           k         = 0;
    std::vector<std::vector<int>> experts; // [n_layers][k]

    bool operator==(const ranking & o) const {
        return n_layers == o.n_layers && n_experts == o.n_experts && k == o.k && experts == o.experts;
    }
};

struct ranking_result {
    bool        ok = false;
    std::string error;
    gravity::ranking value;
};

// Decayed use counts per (layer, expert). Not thread safe.
class usage_tracker {
  public:
    usage_tracker(int n_layers, int n_experts, usage_config cfg = {});

    // Add weight to one expert. Returns false (no change) when ids are out of range.
    bool record(int layer, int expert, double weight = 1.0);

    // Decay all counts by cfg.decay. Caller picks the unit: once per token or once per batch.
    void step();

    // Decayed use now. 0 for out-of-range ids.
    double use(int layer, int expert) const;

    // Swap only if use(candidate) >= swap_min_use and use(candidate) >= swap_ratio * use(victim).
    // A candidate already resident, or an empty resident set, gives swap=false without error.
    swap_decision should_swap(int layer, int candidate, const std::vector<int> & resident) const;

    // Top k experts of every layer by use, ties by lower id. k is clamped to [0, n_experts].
    ranking top_k(int k) const;

    int n_layers() const { return n_layers_; }
    int n_experts() const { return n_experts_; }

  private:
    struct cell {
        double   raw   = 0.0;
        uint64_t epoch = 0; // step count when raw was last written
    };

    bool valid(int layer, int expert) const {
        return layer >= 0 && layer < n_layers_ && expert >= 0 && expert < n_experts_;
    }
    size_t idx(int layer, int expert) const { return (size_t) layer * (size_t) n_experts_ + (size_t) expert; }

    int                 n_layers_;
    int                 n_experts_;
    usage_config        cfg_;
    uint64_t            epoch_ = 0;
    std::vector<cell>   cells_;
};

// Atomic write (tmp file + rename). Same ranking gives the same bytes.
ranking_result save_ranking(const std::string & path, const std::string & model_sha256, const ranking & r);

// Refuses wrong model sha, wrong shape, bad checksum, truncated or malformed file.
// expected_layers/expected_experts < 0 means do not check that dimension.
ranking_result load_ranking(const std::string & path, const std::string & model_sha256,
                            int expected_layers = -1, int expected_experts = -1);

} // namespace gravity
