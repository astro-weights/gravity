#include "scheduler.h"

#include <algorithm>

namespace gravity {

static const size_t max_kept_errors = 256;

scheduler::scheduler(read_pool &   pool_in,
                     slot_ring &   ring_in,
                     int           fd_in,
                     size_t        alignment,
                     scheduler_cfg cfg_in,
                     locate_fn     locate_in) :
    pool(pool_in),
    ring(ring_in),
    fd(fd_in),
    align(alignment),
    cfg(cfg_in),
    locate(std::move(locate_in)) {
    const size_t a            = align > 0 ? align : 1;
    cfg.chunk                 = std::max(cfg.chunk, a);
    cfg.chunk                 = (cfg.chunk + a - 1) / a * a;  // direct reads need chunk offsets on the alignment
    cfg.max_prefetch_inflight = std::max<size_t>(cfg.max_prefetch_inflight, 1);
    if (!cfg.deterministic) {
        try {
            worker = std::thread([this] { collector(); });
        } catch (...) {
            cfg.deterministic = true;  // no thread -> no prefetch; demand still works
        }
    }
}

scheduler::~scheduler() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        stopping               = true;
        std::deque<fill_ptr> q = queue;
        for (const fill_ptr & f : q) {
            if (f->slot < 0) {
                fills.erase(f->key);
            } else {
                fail(f, "scheduler shut down");
            }
        }
        queue.clear();
        issue_cv.notify_all();
    }
    if (worker.joinable()) {
        worker.join();  // drains inflight, so every prefetch id is collected
    }
}

std::string scheduler::make_fill(const expert_key & key, fill_ptr & out) {
    expert_span sp;
    if (!locate || !locate(key, sp)) {
        return "unknown expert " + std::to_string(key.layer) + "/" + std::to_string(key.expert);
    }
    if (sp.len == 0) {
        return "empty span for expert " + std::to_string(key.layer) + "/" + std::to_string(key.expert);
    }
    const uint64_t a      = align > 0 ? align : 1;
    const uint64_t end    = sp.offset + sp.len;
    const uint64_t rd_off = sp.offset / a * a;
    const uint64_t rd_end = (end + a - 1) / a * a;
    if (end < sp.offset || rd_end < end) {
        return "span overflows";
    }
    if (rd_end - rd_off > ring.slot_bytes()) {
        return "expert " + std::to_string(key.layer) + "/" + std::to_string(key.expert) + " needs " +
               std::to_string(rd_end - rd_off) + " bytes, slot has " + std::to_string(ring.slot_bytes());
    }
    try {
        out = std::make_shared<fill>();
    } catch (...) {
        return "out of memory";
    }
    out->key      = key;
    out->span     = sp;
    out->rd_off   = rd_off;
    out->rd_len   = rd_end - rd_off;
    out->n_chunks = (size_t) ((out->rd_len + cfg.chunk - 1) / cfg.chunk);
    return "";
}

std::string scheduler::bind_slot(const fill_ptr & f) {
    acquire_result a = ring.acquire(f->key);
    if (!a.ok()) {
        return a.error;
    }
    if (a.hit) {
        // only the scheduler binds keys, so this means outside use of the ring
        ring.unpin(a.slot);
        return "key already bound in slot ring";
    }
    f->slot = a.slot;
    return "";
}

void scheduler::issue(const fill_ptr & f, size_t max, std::vector<std::pair<read_id, size_t>> & out) {
    if (f->done || f->next_chunk >= f->n_chunks) {
        return;
    }
    if (f->slot < 0) {
        std::string err = bind_slot(f);
        if (!err.empty()) {
            fail(f, err);
            return;
        }
    }
    uint8_t * base = (uint8_t *) ring.data(f->slot);
    size_t    n    = 0;
    while (f->next_chunk < f->n_chunks && (max == 0 || n < max)) {
        const size_t   c   = f->next_chunk;
        const uint64_t rel = (uint64_t) c * cfg.chunk;
        read_req       req;
        req.fd            = fd;
        req.offset        = f->rd_off + rel;
        req.len           = (size_t) std::min<uint64_t>(cfg.chunk, f->rd_len - rel);
        req.dst           = base + rel;
        req.alignment     = align;
        req.allow_partial = align > 0;  // rounded up end may pass EOF; apply() checks the span bytes
        submit_result s   = pool.submit(req);
        if (!s.ok()) {
            fail(f, "submit: " + s.error);
            return;
        }
        try {
            out.emplace_back(s.id, c);
        } catch (...) {
            // keep the id owned: wait here so it is collected
            pool.cancel(s.id);
            read_completion rc = pool.wait(s.id);
            (void) rc;
            fail(f, "out of memory");
            return;
        }
        f->next_chunk++;
        f->pending++;
        st.reads++;
        n++;
    }
}

void scheduler::apply(const fill_ptr & f, size_t chunk, const read_completion & c) {
    if (f->pending > 0) {
        f->pending--;
    }
    if (!f->failed) {
        if (c.status != read_status::ok) {
            fail(f, std::string("read ") + read_status_name(c.status) + ": " + c.error);
            return;
        }
        const uint64_t chunk_off = f->rd_off + (uint64_t) chunk * cfg.chunk;
        const uint64_t span_end  = f->span.offset + f->span.len;
        const uint64_t chunk_len = std::min<uint64_t>(cfg.chunk, f->rd_len - (uint64_t) chunk * cfg.chunk);
        const uint64_t need      = span_end > chunk_off ? std::min<uint64_t>(chunk_len, span_end - chunk_off) : 0;
        if (c.bytes < need) {
            fail(f, "short read: " + std::to_string(c.bytes) + " of " + std::to_string(need) + " bytes");
            return;
        }
    }
    maybe_finish(f);
}

void scheduler::fail(const fill_ptr & f, const std::string & error) {
    if (!f->failed) {
        f->failed = true;
        f->error  = error;
    }
    f->next_chunk = f->n_chunks;  // issue no more chunks
    auto it       = std::find(queue.begin(), queue.end(), f);
    if (it != queue.end()) {
        queue.erase(it);
    }
    maybe_finish(f);
}

void scheduler::maybe_finish(const fill_ptr & f) {
    if (f->done || f->next_chunk < f->n_chunks || f->pending > 0) {
        return;
    }
    f->done = true;
    if (f->slot >= 0) {
        if (f->failed) {
            ring.fail_fill(f->slot);
        } else {
            ring.mark_ready(f->slot);
        }
        ring.unpin(f->slot);  // the fill's own pin
    }
    auto it = fills.find(f->key);
    if (it != fills.end() && it->second == f) {
        fills.erase(it);
    }
    if (f->prefetch) {
        if (f->failed) {
            if (prefetch_errors.size() < max_kept_errors) {
                try {
                    prefetch_errors.push_back(std::to_string(f->key.layer) + "/" + std::to_string(f->key.expert) +
                                              ": " + f->error);
                } catch (...) {
                }
            }
        } else {
            st.prefetch_filled++;
        }
    }
    done_cv.notify_all();
}

void scheduler::pump_prefetch() {
    while (!stopping && n_prefetch_pending < cfg.max_prefetch_inflight && !queue.empty()) {
        fill_ptr                                f = queue.front();
        std::vector<std::pair<read_id, size_t>> ids;
        issue(f, cfg.max_prefetch_inflight - n_prefetch_pending, ids);
        for (const auto & p : ids) {
            // issue() reserved nothing here, so a push failure must still collect the id
            try {
                inflight.push_back({ p.first, f, p.second });
                n_prefetch_pending++;
            } catch (...) {
                read_completion rc = pool.wait(p.first);
                apply(f, p.second, rc);
            }
        }
        if (!queue.empty() && queue.front() == f && f->next_chunk >= f->n_chunks) {
            queue.pop_front();
        }
        if (!ids.empty()) {
            issue_cv.notify_one();
        }
    }
}

void scheduler::collect(std::unique_lock<std::mutex> &                  lock,
                        const fill_ptr &                                f,
                        const std::vector<std::pair<read_id, size_t>> & ids) {
    for (const auto & p : ids) {
        lock.unlock();
        read_completion c = pool.wait(p.first);
        lock.lock();
        apply(f, p.second, c);
    }
}

demand_result scheduler::finish_demand(const fill_ptr & f) {
    demand_result r;
    if (f->failed || f->slot < 0) {
        if (f->slot >= 0) {
            ring.unpin(f->slot);
        }
        r.error = f->error.empty() ? "fill failed" : f->error;
        return r;
    }
    r.slot = f->slot;
    r.data = (const uint8_t *) ring.data(f->slot) + (f->span.offset - f->rd_off);
    r.len  = (size_t) f->span.len;
    return r;
}

demand_result scheduler::demand(const expert_key & key) {
    std::unique_lock<std::mutex> lock(mtx);
    demand_result                r;
    if (stopping) {
        r.error = "scheduler shut down";
        return r;
    }

    fill_ptr f;
    auto     it = fills.find(key);
    if (it != fills.end()) {
        f           = it->second;
        f->prefetch = false;  // errors now go to this caller
        auto qt     = std::find(queue.begin(), queue.end(), f);
        if (qt != queue.end()) {
            queue.erase(qt);  // jump ahead: this thread issues the rest
        }
        if (f->slot >= 0) {
            st.demand_joined++;
        } else {
            st.demand_misses++;
        }
    } else {
        if (ring.lookup(key) >= 0) {
            // bound with no fill in flight -> ready
            acquire_result a = ring.acquire(key);
            if (a.ok() && a.hit && a.ready) {
                st.demand_hits++;
                expert_span sp;
                if (locate && locate(key, sp)) {
                    const uint64_t a_ = align > 0 ? align : 1;
                    r.slot            = a.slot;
                    r.data            = (const uint8_t *) ring.data(a.slot) + (sp.offset - sp.offset / a_ * a_);
                    r.len             = (size_t) sp.len;
                    return r;
                }
                ring.unpin(a.slot);
                r.error = "unknown expert";
                return r;
            }
            if (a.ok()) {
                ring.unpin(a.slot);
            }
            r.error = a.ok() ? "slot bound but not ready" : a.error;
            return r;
        }
        r.error = make_fill(key, f);
        if (!r.error.empty()) {
            return r;
        }
        f->prefetch = false;
        try {
            fills[key] = f;
        } catch (...) {
            r.error = "out of memory";
            return r;
        }
        st.demand_misses++;
    }

    std::vector<std::pair<read_id, size_t>> ids;
    issue(f, 0, ids);
    if (f->slot >= 0) {
        ring.pin(f->slot);  // this caller's pin, kept on success
    }
    collect(lock, f, ids);
    done_cv.wait(lock, [&] { return f->done; });
    return finish_demand(f);
}

bool scheduler::release(int slot) {
    return ring.unpin(slot);
}

prefetch_result scheduler::prefetch(const std::vector<expert_key> & keys) {
    prefetch_result r;
    if (cfg.deterministic) {
        return r;
    }
    std::lock_guard<std::mutex> lock(mtx);
    if (stopping) {
        r.error = "scheduler shut down";
        return r;
    }
    for (const expert_key & key : keys) {
        if (fills.count(key) || ring.lookup(key) >= 0) {
            continue;
        }
        fill_ptr    f;
        std::string err = make_fill(key, f);
        if (err.empty()) {
            try {
                fills[key] = f;
                try {
                    queue.push_back(f);
                } catch (...) {
                    fills.erase(key);
                    throw;
                }
                r.queued++;
            } catch (...) {
                err = "out of memory";
            }
        }
        if (!err.empty() && r.error.empty()) {
            r.error = err;
        }
    }
    pump_prefetch();
    return r;
}

size_t scheduler::drop_queued() {
    size_t n = 0;
    for (auto it = queue.begin(); it != queue.end();) {
        const fill_ptr & f = *it;
        if (f->slot < 0) {
            fills.erase(f->key);
            it = queue.erase(it);
            n++;
        } else {
            ++it;
        }
    }
    st.prefetch_dropped += n;
    return n;
}

size_t scheduler::drop_prefetch() {
    std::lock_guard<std::mutex> lock(mtx);
    return drop_queued();
}

step_result scheduler::step() {
    step_result                 r;
    std::lock_guard<std::mutex> lock(mtx);
    r.dropped = drop_queued();
    r.errors.swap(prefetch_errors);
    return r;
}

scheduler_stats scheduler::stats() const {
    std::lock_guard<std::mutex> lock(mtx);
    return st;
}

void scheduler::collector() {
    std::unique_lock<std::mutex> lock(mtx);
    for (;;) {
        issue_cv.wait(lock, [&] { return stopping || !inflight.empty(); });
        if (inflight.empty()) {
            return;  // stopping, all prefetch ids collected
        }
        chunk_read cr = inflight.front();
        inflight.pop_front();
        lock.unlock();
        read_completion c = pool.wait(cr.id);
        lock.lock();
        n_prefetch_pending--;
        apply(cr.f, cr.chunk, c);
        pump_prefetch();
    }
}

}  // namespace gravity
