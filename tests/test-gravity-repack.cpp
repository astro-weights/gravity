// gravity streaming test: expert repack (alignment, index, byte-equal tensors, idempotence, errors)
#include "repack.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace gravity;

static int n_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_fail++;                                                       \
        }                                                                   \
    } while (0)

static std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static const int N_EXPERT = 4;

struct tensor_spec {
    const char * name;
    ggml_type    type;
    int64_t      ne[3];
    int          n_dims;
};

static const tensor_spec SPECS[] = {
    { "token_embd.weight",              GGML_TYPE_F32,  { 16, 10, 1 }, 2 },
    { "blk.0.ffn_gate_exps.weight",     GGML_TYPE_F32,  { 32, 32, N_EXPERT }, 3 },
    { "blk.0.ffn_up_exps.weight",       GGML_TYPE_F16,  { 32, 16, N_EXPERT }, 3 },
    { "blk.0.ffn_down_exps.weight",     GGML_TYPE_Q8_0, { 64, 8, N_EXPERT }, 3 },
    { "blk.0.attn_norm.weight",         GGML_TYPE_F32,  { 7, 1, 1 }, 1 },
    { "blk.1.ffn_gate_up_exps.weight",  GGML_TYPE_F16,  { 32, 64, N_EXPERT }, 3 },
    { "blk.1.ffn_down_exps.weight",     GGML_TYPE_Q8_0, { 32, 3, N_EXPERT }, 3 },
};

// align 0 = writer default (32). gguf_set_val_u32(general.alignment) alone does not change how the
// writer pads, so a non-default alignment needs a ctx that was read back from a buffer carrying the key
static bool write_gguf(const std::string & path, const std::vector<tensor_spec> & specs, uint32_t align = 0) {
    ggml_init_params ip = { 1 << 20, nullptr, false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   g   = nullptr;
    if (align) {
        gguf_context * boot = gguf_init_empty();
        gguf_set_val_u32(boot, "general.alignment", align);
        std::vector<char> buf(gguf_get_meta_size(boot));
        gguf_get_meta_data(boot, buf.data());
        gguf_free(boot);
        gguf_init_params gp = { true, nullptr };
        g                   = gguf_init_from_buffer(buf.data(), buf.size(), gp);
    } else {
        g = gguf_init_empty();
    }
    gguf_set_val_str(g, "general.architecture", "testmoe");
    gguf_set_val_u32(g, "testmoe.expert_count", N_EXPERT);
    gguf_set_val_i32(g, "testmoe.neg", -5);
    gguf_set_val_f32(g, "testmoe.eps", 1e-5f);
    gguf_set_val_bool(g, "testmoe.flag", true);
    const char * toks[] = { "<s>", "a", "bc" };
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", toks, 3);
    const int32_t ids[] = { 3, 1, 4, 1, 5 };
    gguf_set_arr_data(g, "testmoe.ids", GGUF_TYPE_INT32, ids, 5);

    uint32_t seed = 12345;
    for (const tensor_spec & s : specs) {
        ggml_tensor * t = ggml_new_tensor(ctx, s.type, s.n_dims, s.ne);
        ggml_set_name(t, s.name);
        uint8_t * d = (uint8_t *) t->data;
        for (size_t i = 0; i < ggml_nbytes(t); i++) {
            seed = seed * 1664525u + 1013904223u;
            d[i] = (uint8_t) (seed >> 24);
        }
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

static std::vector<tensor_spec> specs_moe(bool moe) {
    std::vector<tensor_spec> v;
    for (const tensor_spec & s : SPECS) {
        if (moe || !strstr(s.name, "_exps")) {
            v.push_back(s);
        }
    }
    return v;
}

static bool write_synthetic(const std::string & path, bool moe) {
    return write_gguf(path, specs_moe(moe));
}

// no leftover temp files next to out
static bool no_tmp_files(const fs::path & dir, const std::string & out_name) {
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        if (e.path().filename().string().rfind(out_name + ".tmp", 0) == 0) {
            return false;
        }
    }
    return true;
}

struct loaded {
    gguf_context * g   = nullptr;
    ggml_context * ctx = nullptr;
    explicit loaded(const std::string & p) {
        gguf_init_params ip = { false, &ctx };
        g                   = gguf_init_from_file(p.c_str(), ip);
    }
    ~loaded() {
        gguf_free(g);
        ggml_free(ctx);
    }
};

static bool kv_equal(const gguf_context * a, int64_t ia, const gguf_context * b, int64_t ib) {
    const gguf_type t = gguf_get_kv_type(a, ia);
    if (t != gguf_get_kv_type(b, ib)) {
        return false;
    }
    switch (t) {
        case GGUF_TYPE_STRING: return !strcmp(gguf_get_val_str(a, ia), gguf_get_val_str(b, ib));
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(a, ia) == gguf_get_val_u32(b, ib);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(a, ia) == gguf_get_val_i32(b, ib);
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(a, ia) == gguf_get_val_f32(b, ib);
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(a, ia) == gguf_get_val_bool(b, ib);
        case GGUF_TYPE_ARRAY: {
            const gguf_type at = gguf_get_arr_type(a, ia);
            const size_t    n  = gguf_get_arr_n(a, ia);
            if (at != gguf_get_arr_type(b, ib) || n != gguf_get_arr_n(b, ib)) {
                return false;
            }
            if (at == GGUF_TYPE_STRING) {
                for (size_t i = 0; i < n; i++) {
                    if (strcmp(gguf_get_arr_str(a, ia, i), gguf_get_arr_str(b, ib, i))) {
                        return false;
                    }
                }
                return true;
            }
            return !memcmp(gguf_get_arr_data(a, ia), gguf_get_arr_data(b, ib), n * 4);
        }
        default: return false;
    }
}

static void test_names() {
    CHECK(is_moe_expert_tensor("blk.0.ffn_gate_exps.weight"));
    CHECK(is_moe_expert_tensor("blk.12.ffn_up_exps.weight"));
    CHECK(is_moe_expert_tensor("blk.3.ffn_down_exps.weight"));
    CHECK(is_moe_expert_tensor("blk.3.ffn_gate_up_exps.weight"));
    CHECK(!is_moe_expert_tensor("blk.0.ffn_gate_exps.bias"));
    CHECK(!is_moe_expert_tensor("blk.x.ffn_gate_exps.weight"));
    CHECK(!is_moe_expert_tensor("blk..ffn_gate_exps.weight"));
    CHECK(!is_moe_expert_tensor("blk.0.ffn_gate.weight"));
    CHECK(!is_moe_expert_tensor("blk.0.ffn_gate_shexp.weight"));
    CHECK(!is_moe_expert_tensor("token_embd.weight"));
    CHECK(!is_moe_expert_tensor("blk.0"));
    CHECK(!is_moe_expert_tensor(""));
}

static void test_roundtrip(const fs::path & dir, const char * tag, uint32_t in_align) {
    const std::string in  = (dir / (std::string(tag) + "_in.gguf")).string();
    const std::string out = (dir / (std::string(tag) + "_out.gguf")).string();
    const std::string out2 = (dir / (std::string(tag) + "_out2.gguf")).string();
    CHECK(write_gguf(in, specs_moe(true), in_align));
    {
        loaded chk(in);
        CHECK(chk.g && gguf_get_alignment(chk.g) == (in_align ? in_align : 32));
    }

    // dry run
    expert_index plan;
    status       ps = plan_repack(in, plan);
    CHECK(ps.ok());
    CHECK(plan.tensors.size() == 5);
    CHECK(plan.n_expert == N_EXPERT);
    size_t n_al = 0;
    for (const auto & e : plan.tensors) {
        n_al += e.stride % 4096 == 0;
    }
    CHECK(n_al == 2); // F32 32x32 gate and F16 32x64 fused gate_up

    status st = repack_gguf(in, out);
    CHECK(st.ok());
    if (!st.ok()) {
        fprintf(stderr, "repack error: %s\n", st.error.c_str());
        return;
    }
    CHECK(no_tmp_files(dir, fs::path(out).filename().string()));
    CHECK(fs::exists(out));

    loaded a(in);
    loaded b(out);
    CHECK(a.g && b.g);
    if (!a.g || !b.g) {
        return;
    }
    CHECK(gguf_get_alignment(b.g) == 4096);

    // all original KVs kept
    for (int64_t i = 0; i < gguf_get_n_kv(a.g); i++) {
        if (!strcmp(gguf_get_key(a.g, i), "general.alignment")) {
            continue;
        }
        const int64_t j = gguf_find_key(b.g, gguf_get_key(a.g, i));
        CHECK(j >= 0);
        if (j >= 0) {
            CHECK(kv_equal(a.g, i, b.g, j));
        }
    }
    CHECK(gguf_get_val_u32(b.g, gguf_find_key(b.g, "general.alignment")) == 4096);

    // tensors: same name, type, shape, bytes, order; data 4096-aligned
    CHECK(gguf_get_n_tensors(a.g) == gguf_get_n_tensors(b.g));
    for (int64_t i = 0; i < gguf_get_n_tensors(a.g); i++) {
        const char * name = gguf_get_tensor_name(a.g, i);
        CHECK(!strcmp(name, gguf_get_tensor_name(b.g, i)));
        const ggml_tensor * ta = ggml_get_tensor(a.ctx, name);
        const ggml_tensor * tb = ggml_get_tensor(b.ctx, name);
        CHECK(ta && tb);
        if (!ta || !tb) {
            continue;
        }
        CHECK(ta->type == tb->type);
        CHECK(!memcmp(ta->ne, tb->ne, sizeof(ta->ne)));
        CHECK(ggml_nbytes(ta) == ggml_nbytes(tb));
        CHECK(!memcmp(ta->data, tb->data, ggml_nbytes(ta)));
        CHECK((gguf_get_data_offset(b.g) + gguf_get_tensor_offset(b.g, i)) % 4096 == 0);
    }

    // index matches recomputed values, raw reads match original slices
    expert_index idx;
    CHECK(read_expert_index(out, idx).ok());
    CHECK(idx.version == 1);
    CHECK(idx.n_expert == N_EXPERT);
    CHECK(idx.tensors.size() == 5);
    const std::string raw = slurp(out);
    size_t            k   = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(a.g); i++) {
        const char * name = gguf_get_tensor_name(a.g, i);
        if (!is_moe_expert_tensor(name)) {
            continue;
        }
        CHECK(k < idx.tensors.size());
        if (k >= idx.tensors.size()) {
            break;
        }
        const expert_index_entry & e  = idx.tensors[k++];
        const ggml_tensor *        ta = ggml_get_tensor(a.ctx, name);
        CHECK(e.name == name);
        CHECK(e.offset == gguf_get_data_offset(b.g) + gguf_get_tensor_offset(b.g, i));
        CHECK(e.offset % 4096 == 0);
        CHECK(e.stride == ggml_nbytes(ta) / N_EXPERT);
        for (int x = 0; x < N_EXPERT; x++) {
            const uint64_t o = e.offset + (uint64_t) x * e.stride;
            CHECK(o + e.stride <= raw.size());
            CHECK(!memcmp(raw.data() + o, (const char *) ta->data + (size_t) x * e.stride, e.stride));
        }
    }
    CHECK(k == idx.tensors.size());

    // idempotent
    CHECK(repack_gguf(out, out2).ok());
    CHECK(slurp(out) == slurp(out2));
    // original unchanged
    CHECK(fs::exists(in));

    printf("[%s] test file size: before %ju after %ju\n", tag, (uintmax_t) fs::file_size(in), (uintmax_t) fs::file_size(out));
    printf("[%s] expert strides with %% 4096 == 0: %zu of %zu\n", tag, n_al, idx.tensors.size());
}

static void test_errors(const fs::path & dir) {
    const std::string in    = (dir / "e_in.gguf").string();
    const std::string dense = (dir / "e_dense.gguf").string();
    const std::string out   = (dir / "e_out.gguf").string();
    CHECK(write_synthetic(in, true));
    CHECK(write_synthetic(dense, false));

    auto no_tmp = [&](const std::string & o) {
        return no_tmp_files(fs::path(o).parent_path(), fs::path(o).filename().string()) && !fs::exists(o);
    };

    CHECK(!repack_gguf((dir / "missing.gguf").string(), out).ok());
    CHECK(no_tmp(out));

    status st = repack_gguf(dense, out);
    CHECK(!st.ok());
    CHECK(st.error.find("no MoE") != std::string::npos);
    CHECK(no_tmp(out));
    CHECK(!plan_repack(dense, *std::make_unique<expert_index>()).ok());

    const std::string before = slurp(in);
    CHECK(!repack_gguf(in, in).ok());
    CHECK(!repack_gguf(in, (dir / "." / "e_in.gguf").string()).ok());
    CHECK(slurp(in) == before);
    CHECK(no_tmp_files(dir, "e_in.gguf"));

    const std::string bad_out = (dir / "no_such_dir" / "o.gguf").string();
    CHECK(!repack_gguf(in, bad_out).ok());
    CHECK(!fs::exists(bad_out));

    // not a GGUF, truncated GGUF
    const std::string junk = (dir / "junk.gguf").string();
    {
        std::ofstream f(junk, std::ios::binary);
        f << "not a gguf file";
    }
    CHECK(!repack_gguf(junk, out).ok());
    CHECK(no_tmp(out));
    const std::string trunc = (dir / "trunc.gguf").string();
    {
        std::ofstream f(trunc, std::ios::binary);
        f.write(before.data(), (std::streamsize) (before.size() - 1000));
    }
    CHECK(!repack_gguf(trunc, out).ok());
    CHECK(no_tmp(out));

    // 2D *_exps tensor
    const std::string flat = (dir / "e_flat.gguf").string();
    CHECK(write_gguf(flat, { { "blk.0.ffn_gate_exps.weight", GGML_TYPE_F32, { 32, 8, 1 }, 2 } }));
    st = repack_gguf(flat, out);
    CHECK(!st.ok());
    CHECK(st.error.find("not 3D") != std::string::npos);
    CHECK(no_tmp(out));
    CHECK(!plan_repack(flat, *std::make_unique<expert_index>()).ok());

    // MoE tensors that disagree on n_expert
    const std::string mixed = (dir / "e_mixed.gguf").string();
    CHECK(write_gguf(mixed, { { "blk.0.ffn_gate_exps.weight", GGML_TYPE_F32, { 32, 8, 4 }, 3 },
                              { "blk.0.ffn_up_exps.weight", GGML_TYPE_F32, { 32, 8, 3 }, 3 } }));
    st = repack_gguf(mixed, out);
    CHECK(!st.ok());
    CHECK(st.error.find("disagree") != std::string::npos);
    CHECK(no_tmp(out));

    // an existing file at out + ".tmp" is not ours: left alone
    const std::string user_tmp = out + ".tmp";
    {
        std::ofstream f(user_tmp, std::ios::binary);
        f << "user data";
    }
    CHECK(!repack_gguf(dense, out).ok());
    CHECK(slurp(user_tmp) == "user data");
    CHECK(repack_gguf(in, out).ok());
    CHECK(slurp(user_tmp) == "user data");
    fs::remove(user_tmp);

    // index read on a file that was never repacked
    expert_index idx;
    CHECK(!read_expert_index(in, idx).ok());
    CHECK(!read_expert_index((dir / "missing.gguf").string(), idx).ok());
}

// quantize/split copy KVs with gguf_set_kv but lay tensors out with their own alignment: index must not be trusted
static void test_stale_index(const fs::path & dir) {
    const std::string in  = (dir / "s_in.gguf").string();
    const std::string rp  = (dir / "s_rp.gguf").string();
    const std::string bad = (dir / "s_bad.gguf").string();
    CHECK(write_synthetic(in, true));
    CHECK(repack_gguf(in, rp).ok());
    expert_index idx;
    CHECK(read_expert_index(rp, idx).ok());

    loaded a(rp);
    CHECK(a.g != nullptr);
    if (!a.g) {
        return;
    }
    gguf_context * g = gguf_init_empty(); // 32-aligned ctx, like llama-quantize / gguf-split
    gguf_set_kv(g, a.g);
    gguf_remove_key(g, "general.alignment"); // else the header claims 4096 and gguf_init rejects the file
    for (int64_t i = 0; i < gguf_get_n_tensors(a.g); i++) {
        gguf_add_tensor(g, ggml_get_tensor(a.ctx, gguf_get_tensor_name(a.g, i)));
    }
    CHECK(gguf_write_to_file(g, bad.c_str(), false));
    gguf_free(g);

    loaded b(bad);
    CHECK(b.g != nullptr);
    CHECK(gguf_find_key(b.g, "gravity.expert_index.offsets") >= 0); // keys survived the copy
    status st = read_expert_index(bad, idx);
    CHECK(!st.ok());
    CHECK(st.error.find("stale") != std::string::npos);
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("gravity-repack-test-" + std::to_string(std::hash<std::string>{}(std::to_string((uintmax_t) (uintptr_t) &n_fail))));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    test_names();
    test_roundtrip(dir, "default", 0);
    test_roundtrip(dir, "align64", 64);
    test_stale_index(dir);
    test_errors(dir);

    fs::remove_all(dir, ec);
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("ok\n");
    return 0;
}
