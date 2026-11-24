#include "read-pool.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef _WIN32
#    include <malloc.h>
#else
#    include <unistd.h>
#endif

namespace gravity {

static bool is_pow2(size_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

aligned_buffer::aligned_buffer(size_t size, size_t alignment) {
    if (!is_pow2(alignment)) {
        return;
    }
    // posix_memalign needs a multiple of sizeof(void *)
    const size_t a = std::max(alignment, sizeof(void *));
    if (size == 0) {
        size = a;
    }
    if (size > std::numeric_limits<size_t>::max() - (a - 1)) {
        return;
    }
    const size_t rounded = (size + a - 1) & ~(a - 1);
    void *       p       = nullptr;
#ifdef _WIN32
    p = _aligned_malloc(rounded, a);
#else
    if (posix_memalign(&p, a, rounded) != 0) {
        p = nullptr;
    }
#endif
    if (p) {
        ptr   = p;
        n     = rounded;
        align = a;
    }
}

aligned_buffer::~aligned_buffer() {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

aligned_buffer::aligned_buffer(aligned_buffer && other) noexcept : ptr(other.ptr), n(other.n), align(other.align) {
    other.ptr   = nullptr;
    other.n     = 0;
    other.align = 0;
}

aligned_buffer & aligned_buffer::operator=(aligned_buffer && other) noexcept {
    if (this != &other) {
        aligned_buffer tmp(std::move(*this));
        std::swap(ptr, other.ptr);
        std::swap(n, other.n);
        std::swap(align, other.align);
    }
    return *this;
}

const char * read_status_name(read_status st) {
    switch (st) {
        case read_status::ok:
            return "ok";
        case read_status::error:
            return "error";
        case read_status::cancelled:
            return "cancelled";
    }
    return "?";
}

std::string validate_read_req(const read_req & req) {
    if (req.fd < 0) {
        return "bad fd";
    }
    if (req.len > 0 && req.dst == nullptr) {
        return "null dst";
    }
    const uint64_t off_max = (uint64_t) std::numeric_limits<int64_t>::max();
    if (req.offset > off_max || (uint64_t) req.len > off_max - req.offset) {
        return "offset + len overflows off_t";
    }
    if (req.alignment != 0) {
        if (!is_pow2(req.alignment)) {
            return "alignment not a power of two";
        }
        const uint64_t a = req.alignment;
        if (req.offset % a != 0 || (uint64_t) req.len % a != 0 || (uint64_t) (uintptr_t) req.dst % a != 0) {
            return "direct read not aligned to " + std::to_string(req.alignment) + " (offset " +
                   std::to_string(req.offset) + ", len " + std::to_string(req.len) + ")";
        }
    }
    return "";
}

#ifndef _WIN32
static int64_t os_pread(int fd, void * dst, size_t len, uint64_t offset) {
    return (int64_t) pread(fd, dst, len, (off_t) offset);
}
#endif

read_completion read_full(const read_req & req) {
#ifdef _WIN32
    return read_full_with(req, nullptr);
#else
    return read_full_with(req, os_pread);
#endif
}

read_completion read_full_with(const read_req & req, pread_fn pr) {
    read_completion c;
    c.tag = req.tag;

    std::string bad = validate_read_req(req);
    if (!bad.empty()) {
        c.error = bad;
        return c;
    }

    if (pr == nullptr) {
        c.error = "pread not supported on this platform";
        return c;
    }
    char * dst = (char *) req.dst;
    size_t got = 0;
    while (got < req.len) {
        // the kernel only returns an unaligned count at EOF; the next direct pread would be unaligned
        if (req.alignment != 0 && got % req.alignment != 0) {
            break;
        }
        const size_t  want = std::min(req.len - got, (size_t) std::numeric_limits<int64_t>::max());
        const int64_t r    = pr(req.fd, dst + got, want, req.offset + got);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            c.err   = errno;
            c.bytes = got;
            c.error = std::string("pread: ") + strerror(c.err);
            return c;
        }
        if (r == 0) {
            break;
        }
        got += (size_t) r;
    }
    c.bytes = got;
    if (got < req.len && !req.allow_partial) {
        c.error = "EOF after " + std::to_string(got) + " of " + std::to_string(req.len) + " bytes";
        return c;
    }
    c.status = read_status::ok;
    return c;
}

read_pool::read_pool(int n_threads, read_fn read) {
    try {
        st     = std::make_shared<state>();
        st->fn = read ? std::move(read) : read_fn(read_full);
    } catch (...) {
        st.reset();
        return;
    }
    n_threads = std::max(1, n_threads);
    std::lock_guard<std::mutex> lock(st->mtx);
    for (int i = 0; i < n_threads; ++i) {
        try {
            std::shared_ptr<state> s = st;
            st->workers.emplace_back([s] { worker(s); });
        } catch (...) {
            break; // keep the threads we got; submit fails if none
        }
    }
}

read_pool::~read_pool() {
    shutdown();
}

size_t read_pool::n_threads() const {
    if (!st) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(st->mtx);
    return st->workers.size();
}

submit_result read_pool::submit(const read_req & req) {
    submit_result res;
    try {
        res.error = validate_read_req(req);
        if (!res.error.empty()) {
            return res;
        }
        if (!st) {
            res.error = "read pool not initialized";
            return res;
        }
        std::lock_guard<std::mutex> lock(st->mtx);
        if (st->stopping || st->workers.empty()) {
            res.error = "read pool shut down";
            return res;
        }
        const read_id id = st->next_id;
        slot          sl;
        sl.req = req;
        st->slots.emplace(id, std::move(sl));
        try {
            (req.priority == read_priority::prefetch ? st->prefetch_queue : st->queue).push_back(id);
        } catch (...) {
            st->slots.erase(id);
            throw;
        }
        st->next_id++;
        res.id = id;
        st->work_cv.notify_one();
    } catch (...) {
        res.id    = 0;
        res.error = "submit failed: out of memory";
    }
    return res;
}

bool read_pool::state::take_done(read_id id, read_completion & out) {
    auto it = slots.find(id);
    if (it == slots.end()) {
        out       = read_completion();
        out.error = "unknown read id " + std::to_string(id);
        return true;
    }
    if (!it->second.done) {
        return false;
    }
    out = std::move(it->second.comp);
    slots.erase(it);
    return true;
}

read_completion read_pool::wait(read_id id) {
    read_completion out;
    if (!st) {
        out.error = "read pool not initialized";
        return out;
    }
    std::unique_lock<std::mutex> lock(st->mtx);
    st->done_cv.wait(lock, [&] { return st->take_done(id, out); });
    return out;
}

bool read_pool::poll(read_id id, read_completion & out) {
    if (!st) {
        out       = read_completion();
        out.error = "read pool not initialized";
        return true;
    }
    std::lock_guard<std::mutex> lock(st->mtx);
    return st->take_done(id, out);
}

bool read_pool::cancel(read_id id) {
    if (!st) {
        return false;
    }
    std::lock_guard<std::mutex> lock(st->mtx);
    auto                        it = st->slots.find(id);
    if (it == st->slots.end() || it->second.started || it->second.done) {
        return false;
    }
    auto & q = it->second.req.priority == read_priority::prefetch ? st->prefetch_queue : st->queue;
    q.erase(std::remove(q.begin(), q.end(), id), q.end());
    slot & sl       = it->second;
    sl.done         = true;
    sl.comp         = read_completion();
    sl.comp.tag     = sl.req.tag;
    sl.comp.status  = read_status::cancelled;
    sl.comp.error   = "cancelled";
    st->done_cv.notify_all();
    return true;
}

void read_pool::shutdown() {
    if (!st) {
        return;
    }
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lock(st->mtx);
        st->stopping = true;
        for (auto * q : { &st->queue, &st->prefetch_queue }) {
            for (read_id id : *q) {
                slot & sl      = st->slots[id];
                sl.done        = true;
                sl.comp        = read_completion();
                sl.comp.tag    = sl.req.tag;
                sl.comp.status = read_status::cancelled;
                sl.comp.error  = "cancelled: pool shut down";
            }
            q->clear();
        }
        to_join.swap(st->workers);
        st->work_cv.notify_all();
        st->done_cv.notify_all();
    }
    const std::thread::id self = std::this_thread::get_id();
    for (auto & t : to_join) {
        if (t.get_id() == self) {
            t.detach(); // called from inside read_fn: joining self would throw; state stays alive via shared_ptr
        } else if (t.joinable()) {
            t.join();
        }
    }
}

void read_pool::worker(const std::shared_ptr<state> & st) {
    std::unique_lock<std::mutex> lock(st->mtx);
    for (;;) {
        st->work_cv.wait(lock, [&] { return st->stopping || st->has_queued(); });
        if (!st->has_queued()) {
            return; // stopping
        }
        auto &        q  = st->queue.empty() ? st->prefetch_queue : st->queue;
        const read_id id = q.front();
        q.pop_front();
        auto it = st->slots.find(id);
        if (it == st->slots.end()) {
            continue;
        }
        it->second.started = true;
        const read_req req = it->second.req;
        lock.unlock();

        read_completion c;
        try {
            c = st->fn(req);
        } catch (...) {
            c       = read_completion();
            c.error = "read function threw";
        }
        c.tag = req.tag;

        lock.lock();
        // slot may not be erased while started and not done: wait/poll only erase done slots
        auto jt = st->slots.find(id);
        if (jt != st->slots.end()) {
            jt->second.comp = std::move(c);
            jt->second.done = true;
        }
        st->done_cv.notify_all();
    }
}

} // namespace gravity
