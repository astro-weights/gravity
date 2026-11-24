#pragma once

// gravity streaming: pread thread pool with per-request status and cancel

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace gravity {

// RAII aligned heap buffer; size rounds up to alignment. ok() false on bad alignment or alloc failure.
class aligned_buffer {
  public:
    aligned_buffer() = default;
    aligned_buffer(size_t size, size_t alignment);
    ~aligned_buffer();

    aligned_buffer(aligned_buffer && other) noexcept;
    aligned_buffer & operator=(aligned_buffer && other) noexcept;
    aligned_buffer(const aligned_buffer &)             = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;

    bool   ok() const { return ptr != nullptr; }
    void * data() const { return ptr; }
    size_t size() const { return n; }
    size_t alignment() const { return align; }

  private:
    void * ptr   = nullptr;
    size_t n     = 0;
    size_t align = 0;
};

using read_id = uint64_t; // 0 = invalid

// Queued demand reads start before queued prefetch reads (FIFO within each class). Running reads are never preempted.
// Priority only changes timing, never which bytes land in dst.
// Prefetch can starve under continuous demand; bounded by scheduler max_prefetch_inflight + drop_prefetch.
enum class read_priority {
    demand,
    prefetch,
};

struct read_req {
    int           fd            = -1;
    uint64_t      offset        = 0;
    size_t        len           = 0;
    void *        dst           = nullptr;
    uint64_t      tag           = 0;
    size_t        alignment     = 0;     // > 0 for a direct fd: offset, len and dst must be multiples (checked at submit)
    bool          allow_partial = false; // EOF before len counts as ok with fewer bytes
    read_priority priority      = read_priority::demand; // default keeps every caller's behavior unchanged
};

enum class read_status {
    ok,
    error,
    cancelled,
};

const char * read_status_name(read_status st);

struct read_completion {
    uint64_t    tag    = 0;
    size_t      bytes  = 0;
    read_status status = read_status::error;
    int         err    = 0; // errno, 0 if none
    std::string error;
};

struct submit_result {
    read_id     id = 0;
    std::string error;

    bool ok() const { return id != 0; }
};

// pread-like: bytes read, 0 at EOF, or -1 with errno set. Tests inject a fake.
using pread_fn = int64_t (*)(int fd, void * dst, size_t len, uint64_t offset);

// One blocking read with the same rules as the pool workers: retries EINTR and short reads.
read_completion read_full(const read_req & req);
read_completion read_full_with(const read_req & req, pread_fn pr);

// Empty string if req is valid.
std::string validate_read_req(const read_req & req);

using read_fn = std::function<read_completion(const read_req &)>;

// Every id returned by submit, cancelled ones too, must be collected with wait() or poll(); else its completion stays in memory.
// Calling shutdown() or the destructor from inside a read_fn is allowed: that worker thread is detached, not joined.
class read_pool {
  public:
    // read is the OS read; tests inject a fake. Default read_full.
    explicit read_pool(int n_threads, read_fn read = read_full);
    ~read_pool(); // shutdown()

    read_pool(const read_pool &)             = delete;
    read_pool & operator=(const read_pool &) = delete;

    // Thread safe. Bad request, pool shut down or out of memory -> id 0 and error text.
    submit_result submit(const read_req & req);

    // Block until done, then forget the id. Unknown or already collected id -> error completion.
    read_completion wait(read_id id);

    // Non-blocking. true + fill out (and forget the id) when done; unknown id also returns true with an error.
    bool poll(read_id id, read_completion & out);

    // Best effort: true only if the request had not started; it then completes as cancelled.
    bool cancel(read_id id);

    // Queued requests complete as cancelled, running ones finish, threads join. Idempotent.
    // Only the first caller waits for running reads; a concurrent second call can return before they finish.
    void shutdown();

    size_t n_threads() const;

  private:
    struct slot {
        read_req        req;
        bool            started = false;
        bool            done    = false;
        read_completion comp;
    };

    // shared with the workers, so a worker that outlives the pool object (destroyed from inside read_fn) stays valid
    struct state {
        read_fn                           fn;
        std::mutex                        mtx;
        std::condition_variable           work_cv;
        std::condition_variable           done_cv;
        std::deque<read_id>               queue;          // demand
        std::deque<read_id>               prefetch_queue; // only served when queue is empty
        std::unordered_map<read_id, slot> slots;
        std::vector<std::thread>          workers;
        read_id                           next_id  = 1;
        bool                              stopping = false;

        bool take_done(read_id id, read_completion & out); // lock held
        bool has_queued() const { return !queue.empty() || !prefetch_queue.empty(); }
    };

    static void worker(const std::shared_ptr<state> & st);

    std::shared_ptr<state> st;
};

} // namespace gravity
