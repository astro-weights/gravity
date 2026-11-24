#pragma once

// gravity streaming: expert read scheduler. Demand reads go before queued prefetch; prefetch reads in chunks into late bound slots.

#include "read-pool.h"
#include "slot-ring.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gravity {

struct expert_span {
    uint64_t offset = 0;  // in the file
    uint64_t len    = 0;
};

// false if key unknown. Called with the scheduler lock held: must not call back into the scheduler.
using locate_fn = std::function<bool(const expert_key & key, expert_span & out)>;

struct scheduler_cfg {
    size_t chunk                 = 512 * 1024;  // rounded up to the alignment
    size_t max_prefetch_inflight = 4;           // prefetch chunks in the read pool at once
    bool   deterministic         = false;       // eval mode: prefetch off, slot choice depends only on demand order
};

struct demand_result {
    int             slot = -1;       // pinned; give back with scheduler::release()
    const uint8_t * data = nullptr;  // file bytes [offset, offset + len)
    size_t          len  = 0;
    std::string     error;

    bool ok() const { return slot >= 0; }
};

struct prefetch_result {
    size_t      queued = 0;
    std::string error;  // first key that could not be queued (unknown key or too big)
};

struct step_result {
    size_t                   dropped = 0;  // prefetch fills never started
    std::vector<std::string> errors;       // failed prefetch fills since the last step
};

struct scheduler_stats {
    uint64_t reads            = 0;  // chunk reads submitted to the pool
    uint64_t demand_hits      = 0;  // key already ready
    uint64_t demand_joined    = 0;  // waited on a fill already in flight
    uint64_t demand_misses    = 0;  // started a new fill
    uint64_t prefetch_filled  = 0;
    uint64_t prefetch_dropped = 0;
};

// Thread safe. The ring and pool must outlive the scheduler. Every read id it submits is collected.
class scheduler {
  public:
    scheduler(read_pool & pool, slot_ring & ring, int fd, size_t alignment, scheduler_cfg cfg, locate_fn locate);
    ~scheduler();  // drops queued prefetch, waits for reads in flight

    scheduler(const scheduler &)             = delete;
    scheduler & operator=(const scheduler &) = delete;

    // Blocks until the key bytes sit in a slot pinned for the caller, or an error.
    demand_result demand(const expert_key & key);
    bool          release(int slot);  // unpin a slot from demand()

    // Queue keys for background fill. Keys already ready, in flight or queued are skipped. No-op in deterministic mode.
    prefetch_result prefetch(const std::vector<expert_key> & keys);

    // Drop queued prefetch that has not started. Started fills run to the end.
    size_t drop_prefetch();

    // End of a token or layer step: drop_prefetch() plus prefetch errors since the last step.
    step_result step();

    scheduler_stats stats() const;

  private:
    struct fill {
        expert_key  key;
        expert_span span;
        uint64_t    rd_off     = 0;  // span rounded out to the alignment
        uint64_t    rd_len     = 0;
        size_t      n_chunks   = 0;
        size_t      next_chunk = 0;
        size_t      pending    = 0;   // chunks in the pool
        int         slot       = -1;  // bound when the first chunk is issued
        bool        prefetch   = true;
        bool        done       = false;
        bool        failed     = false;
        std::string error;
    };

    using fill_ptr = std::shared_ptr<fill>;

    // all below: lock held
    std::string   make_fill(const expert_key & key, fill_ptr & out);
    std::string   bind_slot(const fill_ptr & f);
    // submit chunks of f; ids go to out. max = 0 means all remaining.
    void          issue(const fill_ptr & f, size_t max, std::vector<std::pair<read_id, size_t>> & out);
    void          apply(const fill_ptr & f, size_t chunk, const read_completion & c);
    void          maybe_finish(const fill_ptr & f);
    void          fail(const fill_ptr & f, const std::string & error);
    void          pump_prefetch();
    size_t        drop_queued();
    // wait for (id, chunk) pairs of f without the lock
    void          collect(std::unique_lock<std::mutex> &                  lock,
                          const fill_ptr &                                f,
                          const std::vector<std::pair<read_id, size_t>> & ids);
    demand_result finish_demand(const fill_ptr & f);

    void collector();

    read_pool &   pool;
    slot_ring &   ring;
    int           fd;
    size_t        align;
    scheduler_cfg cfg;
    locate_fn     locate;

    mutable std::mutex      mtx;
    std::condition_variable done_cv;       // a fill finished
    std::condition_variable issue_cv;      // prefetch chunk issued or stopping

    std::map<expert_key, fill_ptr> fills;  // queued or in flight
    std::deque<fill_ptr>           queue;  // prefetch not fully issued, FIFO

    // prefetch chunk reads in the pool, collected by the collector thread in order
    struct chunk_read {
        read_id  id;
        fill_ptr f;
        size_t   chunk;
    };

    std::deque<chunk_read>   inflight;
    size_t                   n_prefetch_pending = 0;  // prefetch chunks not yet applied
    std::vector<std::string> prefetch_errors;
    scheduler_stats          st;
    bool                     stopping = false;
    std::thread              worker;
};

}  // namespace gravity
