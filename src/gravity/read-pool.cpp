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

read_completion read_full(const read_req & req) {
    read_completion c;
    c.tag = req.tag;

    std::string bad = validate_read_req(req);
    if (!bad.empty()) {
        c.error = bad;
        return c;
    }

#ifdef _WIN32
    c.error = "pread not supported on this platform";
    return c;
#else
    char * dst = (char *) req.dst;
    size_t got = 0;
    while (got < req.len) {
        // the kernel only returns an unaligned count at EOF; the next direct pread would be unaligned
        if (req.alignment != 0 && got % req.alignment != 0) {
            break;
        }
        const size_t  want = std::min(req.len - got, (size_t) std::numeric_limits<ssize_t>::max());
        const ssize_t r    = pread(req.fd, dst + got, want, (off_t) (req.offset + got));
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
#endif
}

read_pool::read_pool(int n_threads, read_fn read) : fn(read ? std::move(read) : read_fn(read_full)) {
    n_threads = std::max(1, n_threads);
    for (int i = 0; i < n_threads; ++i) {
        try {
            workers.emplace_back([this] { worker(); });
        } catch (...) {
            break; // keep the threads we got; submit fails if none
        }
    }
}

read_pool::~read_pool() {
    shutdown();
}

submit_result read_pool::submit(const read_req & req) {
    submit_result res;
    res.error = validate_read_req(req);
    if (!res.error.empty()) {
        return res;
    }
    std::lock_guard<std::mutex> lock(mtx);
    if (stopping || workers.empty()) {
        res.error = "read pool shut down";
        return res;
    }
    const read_id id = next_id++;
    slot          s;
    s.req = req;
    slots.emplace(id, std::move(s));
    queue.push_back(id);
    res.id = id;
    work_cv.notify_one();
    return res;
}

bool read_pool::take_done(read_id id, read_completion & out) {
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
    std::unique_lock<std::mutex> lock(mtx);
    read_completion              out;
    done_cv.wait(lock, [&] { return take_done(id, out); });
    return out;
}

bool read_pool::poll(read_id id, read_completion & out) {
    std::lock_guard<std::mutex> lock(mtx);
    return take_done(id, out);
}

bool read_pool::cancel(read_id id) {
    std::lock_guard<std::mutex> lock(mtx);
    auto                        it = slots.find(id);
    if (it == slots.end() || it->second.started || it->second.done) {
        return false;
    }
    queue.erase(std::remove(queue.begin(), queue.end(), id), queue.end());
    it->second.done        = true;
    it->second.comp        = read_completion();
    it->second.comp.tag    = it->second.req.tag;
    it->second.comp.status = read_status::cancelled;
    it->second.comp.error  = "cancelled";
    done_cv.notify_all();
    return true;
}

void read_pool::shutdown() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lock(mtx);
        stopping = true;
        for (read_id id : queue) {
            slot & s       = slots[id];
            s.done         = true;
            s.comp         = read_completion();
            s.comp.tag     = s.req.tag;
            s.comp.status  = read_status::cancelled;
            s.comp.error   = "cancelled: pool shut down";
        }
        queue.clear();
        to_join.swap(workers);
        work_cv.notify_all();
        done_cv.notify_all();
    }
    for (auto & t : to_join) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void read_pool::worker() {
    std::unique_lock<std::mutex> lock(mtx);
    for (;;) {
        work_cv.wait(lock, [this] { return stopping || !queue.empty(); });
        if (queue.empty()) {
            return; // stopping
        }
        const read_id id = queue.front();
        queue.pop_front();
        auto it = slots.find(id);
        if (it == slots.end()) {
            continue;
        }
        it->second.started  = true;
        const read_req req  = it->second.req;
        lock.unlock();

        read_completion c;
        try {
            c = fn(req);
        } catch (...) {
            c       = read_completion();
            c.error = "read function threw";
        }
        c.tag = req.tag;

        lock.lock();
        // slot may not be erased while started and not done: wait/poll only erase done slots
        slot & s = slots[id];
        s.comp   = std::move(c);
        s.done   = true;
        done_cv.notify_all();
    }
}

} // namespace gravity
