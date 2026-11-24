// gravity streaming test: swap policy (usage decay, hysteresis, saved ranking)
#include "swap-policy.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_fail++;                                                    \
        }                                                                \
    } while (0)

using namespace gravity;

static std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static void spit(const std::string & p, const std::string & s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(s.data(), (std::streamsize) s.size());
}

static void test_decay() {
    usage_tracker t(2, 4);
    t.record(0, 1, 3.0);
    for (int n = 1; n <= 40; n++) {
        t.step();
        CHECK(t.use(0, 1) == 3.0 * std::pow(0.7, (double) n)); // exact, not tolerance
    }
    CHECK(t.use(1, 1) == 0.0);

    // record after steps: old value decays, new weight is full
    usage_tracker u(1, 2);
    u.record(0, 0, 1.0);
    u.step();
    u.step();
    u.record(0, 0, 1.0);
    CHECK(std::fabs(u.use(0, 0) - (1.0 + std::pow(0.7, 2.0))) < 1e-12);

    // custom decay
    usage_tracker v(1, 1, { 0.5, 2.0, 1.5 });
    v.record(0, 0, 8.0);
    v.step();
    v.step();
    v.step();
    CHECK(v.use(0, 0) == 1.0);
}

static void test_bounds() {
    usage_tracker t(2, 3);
    CHECK(!t.record(-1, 0));
    CHECK(!t.record(2, 0));
    CHECK(!t.record(0, -1));
    CHECK(!t.record(0, 3));
    CHECK(!t.record(0, 0, std::nan("")));
    CHECK(!t.record(0, 0, INFINITY));
    CHECK(t.use(-1, 0) == 0.0 && t.use(0, 3) == 0.0 && t.use(5, 5) == 0.0);
    CHECK(t.use(0, 0) == 0.0);

    CHECK(!t.should_swap(2, 0, { 1 }).error.empty());
    CHECK(!t.should_swap(0, 3, { 1 }).error.empty());
    CHECK(!t.should_swap(0, -1, { 1 }).error.empty());
    CHECK(!t.should_swap(0, 0, { 1, 9 }).error.empty());
    CHECK(!t.should_swap(0, 0, { -1 }).error.empty());
    CHECK(!t.should_swap(0, 0, { 1, 9 }).swap);

    usage_tracker z(0, 0);
    CHECK(!z.record(0, 0));
    CHECK(z.top_k(3).experts.empty());
    usage_tracker neg(-5, -5);
    CHECK(neg.n_layers() == 0 && neg.n_experts() == 0);
}

static void test_hysteresis() {
    // all values set by record without step, so they are exact
    auto run = [](double cand, double victim) {
        usage_tracker t(1, 4);
        t.record(0, 0, cand);
        t.record(0, 1, victim);
        t.record(0, 2, victim + 5.0);
        return t.should_swap(0, 0, { 1, 2 });
    };

    CHECK(!run(1.9, 0.0).swap);   // below swap_min_use even against an idle victim
    CHECK(!run(1.9, 0.5).swap);
    CHECK(run(2.0, 0.0).swap);    // at min use, victim idle
    CHECK(run(2.0, 0.0).victim == 1);
    CHECK(run(2.0, 1.3).swap);    // needs 1.95
    CHECK(!run(2.0, 1.4).swap);   // needs 2.1
    CHECK(!run(2.0, 1.5).swap);   // needs 2.25
    CHECK(run(3.0, 2.0).swap);    // exactly ratio * victim
    CHECK(!run(2.99, 2.0).swap);
    CHECK(run(2.25, 1.5).swap);   // exactly 2.25

    swap_decision d = run(1.9, 0.0);
    CHECK(d.victim == -1 && d.error.empty());

    // candidate already resident, or nothing resident: no swap, no error
    usage_tracker t(1, 4);
    t.record(0, 0, 10.0);
    swap_decision r = t.should_swap(0, 0, { 0, 1 });
    CHECK(!r.swap && r.error.empty());
    r = t.should_swap(0, 0, {});
    CHECK(!r.swap && r.error.empty());

    // custom config
    usage_tracker c(1, 3, { 0.7, 1.0, 2.0 });
    c.record(0, 0, 1.0);
    c.record(0, 1, 0.5);
    CHECK(c.should_swap(0, 0, { 1 }).swap);
    c.record(0, 1, 0.1);
    CHECK(!c.should_swap(0, 0, { 1 }).swap); // needs 1.2
}

static void test_victim_choice() {
    usage_tracker t(2, 6);
    t.record(0, 5, 9.0);
    t.record(0, 1, 1.0);
    t.record(0, 3, 1.0);
    t.record(0, 2, 4.0);
    // 1 and 3 tie at 1.0: lowest id wins, whatever the order in the resident list
    CHECK(t.should_swap(0, 5, { 3, 2, 1 }).victim == 1);
    CHECK(t.should_swap(0, 5, { 1, 2, 3 }).victim == 1);
    // layer 1 is independent
    CHECK(!t.should_swap(1, 5, { 1, 2 }).swap);
    // zero-use ties: lowest id
    CHECK(t.should_swap(0, 5, { 4, 0, 2 }).victim == 0);
}

static void test_determinism() {
    auto build = [] {
        usage_tracker t(3, 8);
        for (int i = 0; i < 200; i++) {
            t.record(i % 3, (i * 7) % 8, 1.0 + (i % 5) * 0.25);
            if (i % 4 == 0) {
                t.step();
            }
        }
        return t;
    };
    usage_tracker a = build();
    usage_tracker b = build();
    for (int l = 0; l < 3; l++) {
        for (int e = 0; e < 8; e++) {
            CHECK(a.use(l, e) == b.use(l, e));
            auto da = a.should_swap(l, e, { 0, 1, 2 });
            auto db = b.should_swap(l, e, { 0, 1, 2 });
            CHECK(da.swap == db.swap && da.victim == db.victim);
        }
    }
    CHECK(a.top_k(4) == b.top_k(4));

    const std::string pa = "test-gravity-swap-det-a.bin";
    const std::string pb = "test-gravity-swap-det-b.bin";
    CHECK(save_ranking(pa, "abc123", a.top_k(4)).ok);
    CHECK(save_ranking(pb, "abc123", b.top_k(4)).ok);
    CHECK(!slurp(pa).empty() && slurp(pa) == slurp(pb));
    std::filesystem::remove(pa);
    std::filesystem::remove(pb);
}

static void test_ranking() {
    usage_tracker t(2, 5);
    t.record(0, 3, 5.0);
    t.record(0, 1, 2.0);
    t.record(0, 4, 2.0); // ties with 1: lower id first
    t.record(1, 2, 1.0);
    ranking r = t.top_k(3);
    CHECK(r.n_layers == 2 && r.n_experts == 5 && r.k == 3);
    CHECK((r.experts[0] == std::vector<int>{ 3, 1, 4 }));
    CHECK((r.experts[1] == std::vector<int>{ 2, 0, 1 })); // zero-use fill by id
    CHECK(t.top_k(99).k == 5);
    CHECK(t.top_k(-3).k == 0);
    CHECK(t.top_k(0).experts.size() == 2);
}

static void test_roundtrip_and_refusals() {
    const std::string p   = "test-gravity-swap-rank.bin";
    const std::string sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    usage_tracker t(3, 6);
    for (int i = 0; i < 50; i++) {
        t.record(i % 3, (i * 5) % 6, 1.0);
    }
    ranking r = t.top_k(4);

    CHECK(save_ranking(p, sha, r).ok);
    CHECK(!std::filesystem::exists(p + ".tmp")); // tmp renamed away

    ranking_result ok = load_ranking(p, sha);
    CHECK(ok.ok && ok.value == r);
    CHECK(load_ranking(p, sha, 3, 6).ok);
    CHECK(load_ranking(p, sha, 3, -1).ok);

    // empty ranking (k = 0) round trips too
    ranking e = t.top_k(0);
    CHECK(save_ranking(p + ".e", sha, e).ok);
    ranking_result eok = load_ranking(p + ".e", sha);
    CHECK(eok.ok && eok.value == e);
    std::filesystem::remove(p + ".e");

    // overwrite existing file
    ranking r2 = t.top_k(2);
    CHECK(save_ranking(p, sha, r2).ok);
    CHECK(load_ranking(p, sha).value == r2);
    CHECK(save_ranking(p, sha, r).ok);

    // wrong sha
    std::string sha2 = sha;
    sha2[0]          = 'f';
    CHECK(!load_ranking(p, sha2).ok);
    CHECK(!load_ranking(p, "").ok);
    CHECK(!load_ranking(p, sha + "0").ok);
    CHECK(!load_ranking(p, sha.substr(1)).ok);

    // wrong shape
    CHECK(!load_ranking(p, sha, 4, 6).ok);
    CHECK(!load_ranking(p, sha, 3, 7).ok);

    // missing file
    CHECK(!load_ranking("test-gravity-swap-nope.bin", sha).ok);

    // every truncation point refused, none crash
    const std::string good = slurp(p);
    for (size_t n = 0; n < good.size(); n++) {
        spit(p, good.substr(0, n));
        CHECK(!load_ranking(p, sha).ok);
    }

    // every single byte flip refused (header, sha, ids or checksum)
    for (size_t i = 0; i < good.size(); i++) {
        std::string bad = good;
        bad[i]          = (char) (bad[i] ^ 0x01);
        spit(p, bad);
        CHECK(!load_ranking(p, sha).ok);
    }

    // trailing garbage
    spit(p, good + "x");
    CHECK(!load_ranking(p, sha).ok);

    // bad checksum only
    {
        std::string bad = good;
        bad[bad.size() - 1] = (char) (bad[bad.size() - 1] ^ 0x80);
        spit(p, bad);
        ranking_result x = load_ranking(p, sha);
        CHECK(!x.ok && x.error.find("checksum") != std::string::npos);
    }

    // garbage / empty file
    spit(p, "");
    CHECK(!load_ranking(p, sha).ok);
    spit(p, std::string(1000, 'z'));
    CHECK(!load_ranking(p, sha).ok);

    // save refuses malformed input
    ranking bad = r;
    bad.experts[0][0] = 99;
    CHECK(!save_ranking(p, sha, bad).ok);
    bad = r;
    bad.experts.pop_back();
    CHECK(!save_ranking(p, sha, bad).ok);
    bad = r;
    bad.experts[1].pop_back();
    CHECK(!save_ranking(p, sha, bad).ok);
    CHECK(!save_ranking(p, "", r).ok);
    CHECK(!save_ranking("/nonexistent-dir-gravity/x.bin", sha, r).ok);

    std::filesystem::remove(p);
}

int main() {
    test_decay();
    test_bounds();
    test_hysteresis();
    test_victim_choice();
    test_determinism();
    test_ranking();
    test_roundtrip_and_refusals();
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    return 0;
}
