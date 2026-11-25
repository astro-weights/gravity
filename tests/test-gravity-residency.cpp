// gravity residency test: LRU budget logic, and which OS advice really drops file-backed pages (mincore)
#include "residency.h"

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#endif

static int n_fail = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_fail++;                                                    \
        }                                                                \
    } while (0)

using namespace gravity;

static void test_inner() {
    const residency_range a = residency_inner(4096, 8192, 4096);
    CHECK(a.addr == 4096 && a.len == 8192);
    const residency_range b = residency_inner(100, 8192, 4096);
    CHECK(b.addr == 4096 && b.len == 4096);
    const residency_range c = residency_inner(100, 4000, 4096); // no full page inside
    CHECK(c.len == 0);
    const residency_range d = residency_inner(16384, 16383, 16384);
    CHECK(d.len == 0);
}

static void test_add_errors() {
    residency_lru lru(0, 4096);
    CHECK(!lru.add_tensor(-1, 0, 4, 4096).ok);
    CHECK(!lru.add_tensor(0, 0, 0, 4096).ok);
    CHECK(!lru.add_tensor(0, 0, 4, 0).ok);
    CHECK(lru.add_tensor(0, 0, 4, 4096).ok);
    CHECK(!lru.add_tensor(0, 1 << 20, 8, 4096).ok); // n_expert differs in layer
    lru.mark_used(0, 0);
    CHECK(!lru.add_tensor(1, 1 << 20, 4, 4096).ok); // after first use
    lru.mark_used(0, 4);
    lru.mark_used(3, 0);
    lru.mark_used(-1, 0);
    CHECK(lru.n_bad_ids() == 3);
}

static void test_lru() {
    const size_t P = 4096;
    // 2 layers x 4 experts, 2 tensors per layer, slice = 2 pages: unit = 4 pages
    residency_lru lru(/*budget*/ 3 * 4 * P, P);
    CHECK(lru.add_tensor(0, 0x100000, 4, 2 * P).ok);
    CHECK(lru.add_tensor(0, 0x200000, 4, 2 * P).ok);
    CHECK(lru.add_tensor(1, 0x300000, 4, 2 * P).ok);
    CHECK(lru.add_tensor(1, 0x400000, 4, 2 * P).ok);
    CHECK(lru.total_bytes() == 8 * 4 * P);
    CHECK(lru.resident_bytes() == lru.total_bytes()); // starts all resident

    // step 1 uses (0,1) (1,2): drop 5 units, keep the 2 used + 1 oldest-lowest unit
    lru.mark_used(0, 1);
    lru.mark_used(1, 2);
    std::vector<residency_range> r = lru.end_step();
    CHECK(lru.resident_bytes() == 3 * 4 * P);
    CHECK(lru.stats().n_evicted == 5);
    CHECK(r.size() == 10); // 2 tensors per unit
    // ties at last_use 0 go by unit index: (0,0) (0,2) (0,3) (1,0) (1,1) dropped, (1,3) kept
    CHECK(r[0].addr == 0x100000 && r[0].len == 2 * P);
    CHECK(r[1].addr == 0x200000);
    CHECK(r[2].addr == 0x100000 + 2 * 2 * P);
    for (const auto & x : r) {
        CHECK(!(x.addr == 0x100000 + 1 * 2 * P)); // used unit never dropped
    }

    // under budget: nothing dropped
    lru.mark_used(0, 1);
    r = lru.end_step();
    CHECK(r.empty());

    // step 3 uses 3 new units: resident 6 > 3, drop the 3 old ones not used now
    lru.mark_used(0, 0);
    lru.mark_used(0, 2);
    lru.mark_used(0, 3);
    CHECK(lru.stats().n_refault_units == 3);
    r = lru.end_step();
    CHECK(lru.resident_bytes() == 3 * 4 * P);
    CHECK(r.size() == 6);

    // a step that uses more units than the budget keeps all of them (never drops units in use)
    for (int e = 0; e < 4; ++e) {
        lru.mark_used(0, e);
        lru.mark_used(1, e);
    }
    r = lru.end_step();
    CHECK(lru.resident_bytes() == 8 * 4 * P);
    CHECK(r.empty());
    CHECK(lru.stats().n_steps == 4);
}

static void test_evict_used() {
    const size_t P = 4096;
    // 1 layer x 4 experts, 1 page each, budget 1 unit: a step that uses all 4 still ends at the budget
    residency_lru lru(1 * P, P);
    CHECK(lru.add_tensor(0, 0x100000, 4, P).ok);
    // first step (MoE cache: nothing read yet) drops down to the budget, oldest-lowest first
    std::vector<residency_range> r = lru.end_step(true);
    CHECK(lru.resident_bytes() == P);
    CHECK(r.size() == 3);
    CHECK(r[0].addr == 0x100000 && r[1].addr == 0x100000 + P && r[2].addr == 0x100000 + 2 * P);

    // all used in one step (prefill): units used earlier this step order by index, all but one go
    for (int e = 0; e < 4; ++e) {
        lru.mark_used(0, e);
    }
    CHECK(lru.stats().n_refault_units == 3);
    r = lru.end_step(true);
    CHECK(lru.resident_bytes() == P);
    CHECK(r.size() == 3);
    CHECK(lru.stats().n_evicted == 6);

    // unused-this-step units go before used ones: use 1 and 2, (3) kept from before is older -> dropped first
    lru.mark_used(0, 1);
    lru.mark_used(0, 2);
    r = lru.end_step(true);
    CHECK(lru.resident_bytes() == P);
    CHECK(r.size() == 2);
    CHECK(r[0].addr == 0x100000 + 3 * P); // the old unit
    CHECK(r[1].addr == 0x100000 + 1 * P); // then the lower used one, (2) kept

    // under budget: nothing dropped
    r = lru.end_step(true);
    CHECK(r.empty());
}

static void test_unaligned_slices() {
    const size_t P = 4096;
    // slice 1.5 pages at odd base: only full pages inside each slice are dropped
    residency_lru lru(0, P);
    CHECK(lru.add_tensor(0, 0x100000 + 100, 2, P + P / 2).ok);
    std::vector<residency_range> r = lru.end_step();
    for (const auto & x : r) {
        CHECK(x.addr % P == 0 && x.len % P == 0);
        CHECK(x.addr >= 0x100000 + 100 && x.addr + x.len <= 0x100000 + 100 + 2 * (P + P / 2));
    }
    CHECK(lru.resident_bytes() == 0);
}

#if defined(__APPLE__) || defined(__linux__)
// Real file mmap: touch every page, apply advice, count pages in core. Reports what works on this OS.
static void test_os_advice() {
    const size_t ps = os_page_size();
    const size_t N  = 256 * ps;
    const std::string path = (std::filesystem::temp_directory_path() / ("test-gravity-residency-" + std::to_string(getpid()) + ".bin")).string();
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0);
    if (fd < 0) {
        return;
    }
    std::vector<char> buf(N);
    for (size_t i = 0; i < N; ++i) {
        buf[i] = (char) (i * 31 + 7);
    }
    CHECK(write(fd, buf.data(), N) == (ssize_t) N);
    fsync(fd);

    const drop_method methods[] = { drop_method::madv_dontneed, drop_method::posix_dontneed, drop_method::madv_pageout, drop_method::msync_invalidate };
    bool default_works = false;
    for (drop_method m : methods) {
        void * p = mmap(nullptr, N, PROT_READ, MAP_SHARED, fd, 0);
        CHECK(p != MAP_FAILED);
        if (p == MAP_FAILED) {
            continue;
        }
        volatile unsigned long sum = 0;
        for (size_t i = 0; i < N; i += ps) {
            sum = sum + (unsigned char) ((const char *) p)[i];
        }
        const int64_t before = resident_pages(p, N);
        // drop the middle half only, the rest must stay
        const int     rc    = drop_pages((char *) p + N / 4, N / 2, m);
        const int64_t after = resident_pages(p, N);
        // contents must be unchanged after the drop (pages come back from the file)
        const bool same = memcmp(p, buf.data(), N) == 0;
        CHECK(same);
        const bool works = rc == 0 && after >= 0 && before - after >= (int64_t) (N / 2 / ps) * 9 / 10;
        printf("os advice: %-36s rc=%d in-core pages %lld -> %lld of %zu: %s\n", drop_method_name(m), rc, (long long) before, (long long) after, N / ps,
               rc == ENOSYS ? "not on this OS" : works ? "DROPS pages" : "no effect on mincore");
        if (m == default_drop_method()) {
            default_works = works;
        }
        munmap(p, N);
    }
#    if defined(__APPLE__)
    // measured on macOS 27: only MS_INVALIDATE drops clean file pages; this guards the default
    CHECK(default_works);
#    else
    // Linux mincore reports the page cache, which MADV_DONTNEED keeps: RSS drops, mincore may not
    if (!default_works) {
        printf("os advice: default %s shows no mincore effect here (page cache kept); skip\n", drop_method_name(default_drop_method()));
    }
#    endif
    close(fd);
    unlink(path.c_str());
}
#endif

// LLAMA_GRAVITY_RESIDENCY_MODEL=<MoE gguf>: greedy tokens with a tiny budget must equal budget 0 (off).
// Pages dropped from the mmap fault back in from the same file, so the bytes the model reads never change.
static std::vector<llama_token> greedy(const char * path, int ngl, int budget_mib, int n_gen) {
    std::vector<llama_token> out;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = ngl;
    mp.load_mode          = LLAMA_LOAD_MODE_MMAP;
    llama_model * model   = llama_model_load_from_file(path, mp);
    if (!model) {
        return out;
    }
    llama_context_params cp  = llama_context_default_params();
    cp.n_ctx                 = 256;
    cp.n_batch               = 256;
    cp.gravity_residency_mib = budget_mib;
    llama_context * ctx      = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return out;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char *        text  = "Once upon a time";
    std::vector<llama_token> toks(64);
    const int n = llama_tokenize(vocab, text, (int) strlen(text), toks.data(), (int) toks.size(), true, false);
    toks.resize(n > 0 ? n : 0);
    llama_batch batch = llama_batch_get_one(toks.data(), (int) toks.size());
    llama_token tok   = 0;
    for (int i = 0; i < n_gen; ++i) {
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int     nv     = llama_vocab_n_tokens(vocab);
        tok                  = 0;
        for (int v = 1; v < nv; ++v) {
            if (logits[v] > logits[tok]) {
                tok = v;
            }
        }
        out.push_back(tok);
        batch = llama_batch_get_one(&tok, 1);
    }
    llama_free(ctx);
    llama_model_free(model);
    return out;
}

static void test_model_bit_identical() {
    const char * path = getenv("LLAMA_GRAVITY_RESIDENCY_MODEL");
    if (path == nullptr || path[0] == '\0') {
        printf("model test: LLAMA_GRAVITY_RESIDENCY_MODEL not set, skipped\n");
        return;
    }
    llama_backend_init();
    // ngl 0: CPU_Mapped experts, pages really drop and refault; ngl 99: GPU-wrapped experts turn residency off at init
    for (int ngl : { 0, 99 }) {
        const std::vector<llama_token> off = greedy(path, ngl, 0, 64);
        const std::vector<llama_token> on  = greedy(path, ngl, 1, 64);
        CHECK(off.size() == 64);
        CHECK(on == off);
        printf("model test: ngl %d, %zu tokens, residency 1 MiB vs off: %s\n", ngl, off.size(), on == off ? "identical" : "DIFFER");
    }
    llama_backend_free();
}

int main() {
    test_inner();
    test_add_errors();
    test_lru();
    test_evict_used();
    test_unaligned_slices();
#if defined(__APPLE__) || defined(__linux__)
    test_os_advice();
#else
    printf("os advice: no mmap on this OS, skipped\n");
#endif
    test_model_bit_identical();
    if (n_fail) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("all residency tests passed\n");
    return 0;
}
