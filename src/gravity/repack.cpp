#include "repack.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>

namespace fs = std::filesystem;

namespace gravity {

namespace {

const char * const KEY_VERSION  = "gravity.expert_index.version";
const char * const KEY_TENSORS  = "gravity.expert_index.tensors";
const char * const KEY_OFFSETS  = "gravity.expert_index.offsets";
const char * const KEY_STRIDES  = "gravity.expert_index.strides";
const char * const KEY_N_EXPERT = "gravity.expert_index.n_expert";

struct gguf_deleter {
    void operator()(gguf_context * p) const { gguf_free(p); }
};
struct ggml_deleter {
    void operator()(ggml_context * p) const { ggml_free(p); }
};
using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using ggml_ptr = std::unique_ptr<ggml_context, ggml_deleter>;

struct loaded_gguf {
    gguf_ptr ctx;
    ggml_ptr data; // tensor descriptors only (no_alloc)
};

status open_gguf(const std::string & path, loaded_gguf & out) {
    ggml_context *    ctx_data = nullptr;
    gguf_init_params  params   = { /*no_alloc =*/ true, /*ctx =*/ &ctx_data };
    out.ctx.reset(gguf_init_from_file(path.c_str(), params));
    out.data.reset(ctx_data);
    if (!out.ctx) {
        return { "cannot read GGUF: " + path };
    }
    return {};
}

// arr: key must be an array of `type`, else a scalar of `type`
bool find_key(const gguf_context * ctx, const char * key, gguf_type type, bool arr, int64_t & id) {
    id = gguf_find_key(ctx, key);
    if (id < 0) {
        return false;
    }
    if (arr) {
        return gguf_get_kv_type(ctx, id) == GGUF_TYPE_ARRAY && gguf_get_arr_type(ctx, id) == type;
    }
    return gguf_get_kv_type(ctx, id) == type;
}

// gguf_set_val_u32(general.alignment) does not change ctx->alignment, only a file/buffer read does.
// So build the output context from a tiny buffer that carries the key.
gguf_ptr make_aligned_ctx() {
    gguf_ptr boot(gguf_init_empty());
    gguf_set_val_u32(boot.get(), GGUF_KEY_GENERAL_ALIGNMENT, REPACK_ALIGNMENT);
    std::vector<char> buf(gguf_get_meta_size(boot.get()));
    gguf_get_meta_data(boot.get(), buf.data());
    gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    return gguf_ptr(gguf_init_from_buffer(buf.data(), buf.size(), params));
}

struct moe_tensor {
    int64_t  id     = 0;
    uint64_t stride = 0;
};

// fills `moe` and n_expert from the tensor list
status collect_moe(const gguf_context * src, ggml_context * ctx_data, std::vector<moe_tensor> & moe, uint32_t & n_expert) {
    n_expert = 0;
    const int64_t n_tensors = gguf_get_n_tensors(src);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(src, i);
        if (!is_moe_expert_tensor(name)) {
            continue;
        }
        const ggml_tensor * t = ggml_get_tensor(ctx_data, name);
        if (!t) {
            return { std::string("tensor descriptor missing: ") + name };
        }
        if (ggml_n_dims(t) != 3 || t->ne[2] <= 0 || t->ne[2] > UINT32_MAX) {
            return { std::string("MoE tensor is not 3D: ") + name };
        }
        const uint64_t nbytes = gguf_get_tensor_size(src, i);
        const uint64_t ne2    = (uint64_t) t->ne[2];
        if (nbytes % ne2 != 0) {
            return { std::string("MoE tensor size not divisible by n_expert: ") + name };
        }
        if (n_expert == 0) {
            n_expert = (uint32_t) ne2;
        } else if (n_expert != ne2) {
            return { std::string("MoE tensors disagree on n_expert at: ") + name };
        }
        moe.push_back({ i, nbytes / ne2 });
    }
    if (moe.empty()) {
        return { "no MoE expert tensors (blk.N.ffn_{gate,up,down,gate_up}_exps.weight) found, nothing to repack" };
    }
    return {};
}

bool write_zeros(std::ofstream & f, uint64_t n) {
    static const char zeros[4096] = {};
    while (n > 0) {
        const uint64_t k = std::min<uint64_t>(n, sizeof(zeros));
        f.write(zeros, (std::streamsize) k);
        n -= k;
    }
    return f.good();
}

bool copy_bytes(std::ifstream & in, uint64_t pos, uint64_t n, std::ofstream & out, std::vector<char> & buf) {
    in.clear();
    in.seekg((std::streamoff) pos);
    while (n > 0) {
        const uint64_t k = std::min<uint64_t>(n, buf.size());
        in.read(buf.data(), (std::streamsize) k);
        if (!in.good()) {
            return false;
        }
        out.write(buf.data(), (std::streamsize) k);
        if (!out.good()) {
            return false;
        }
        n -= k;
    }
    return true;
}

status repack_to(const std::string & in_path, const std::string & tmp_path) {
    loaded_gguf src;
    if (status st = open_gguf(in_path, src); !st.ok()) {
        return st;
    }
    std::vector<moe_tensor> moe;
    uint32_t                n_expert = 0;
    if (status st = collect_moe(src.ctx.get(), src.data.get(), moe, n_expert); !st.ok()) {
        return st;
    }

    const int64_t n_tensors = gguf_get_n_tensors(src.ctx.get());
    const size_t  in_data   = gguf_get_data_offset(src.ctx.get());

    std::error_code ec;
    const uint64_t  in_size = fs::file_size(in_path, ec);
    if (ec) {
        return { "cannot stat input: " + ec.message() };
    }
    if (n_tensors > 0) {
        const int64_t last = n_tensors - 1;
        const uint64_t end = in_data + gguf_get_tensor_offset(src.ctx.get(), last) + gguf_get_tensor_size(src.ctx.get(), last);
        if (end > in_size) {
            return { "input is truncated: tensor data runs past end of file" };
        }
    }

    gguf_ptr dst = make_aligned_ctx();
    if (!dst || gguf_get_alignment(dst.get()) != REPACK_ALIGNMENT) {
        return { "cannot create aligned GGUF context" };
    }
    gguf_set_kv(dst.get(), src.ctx.get());
    // drop old index so repacking a repacked file gives the same bytes
    for (const char * key : { KEY_VERSION, KEY_TENSORS, KEY_OFFSETS, KEY_STRIDES, KEY_N_EXPERT }) {
        gguf_remove_key(dst.get(), key);
    }
    gguf_set_val_u32(dst.get(), GGUF_KEY_GENERAL_ALIGNMENT, REPACK_ALIGNMENT);
    for (int64_t i = 0; i < n_tensors; i++) {
        const ggml_tensor * t = ggml_get_tensor(src.data.get(), gguf_get_tensor_name(src.ctx.get(), i));
        if (!t) {
            return { "tensor descriptor missing" };
        }
        gguf_add_tensor(dst.get(), t);
    }

    // offsets depend on meta size, meta size on array lengths only: fill placeholders first
    const size_t n_moe = moe.size();
    std::vector<std::string> names(n_moe);
    std::vector<const char *> name_ptrs(n_moe);
    std::vector<uint64_t> offsets(n_moe, 0);
    std::vector<uint64_t> strides(n_moe, 0);
    for (size_t k = 0; k < n_moe; k++) {
        names[k]     = gguf_get_tensor_name(src.ctx.get(), moe[k].id);
        name_ptrs[k] = names[k].c_str();
        strides[k]   = moe[k].stride;
    }
    auto set_index = [&]() {
        gguf_set_val_u32(dst.get(), KEY_VERSION, EXPERT_INDEX_VERSION);
        gguf_set_arr_str(dst.get(), KEY_TENSORS, name_ptrs.data(), n_moe);
        gguf_set_arr_data(dst.get(), KEY_OFFSETS, GGUF_TYPE_UINT64, offsets.data(), n_moe);
        gguf_set_arr_data(dst.get(), KEY_STRIDES, GGUF_TYPE_UINT64, strides.data(), n_moe);
        gguf_set_val_u32(dst.get(), KEY_N_EXPERT, n_expert);
    };
    set_index();
    const size_t out_data = gguf_get_meta_size(dst.get()); // meta is padded to 4096, data starts here
    for (size_t k = 0; k < n_moe; k++) {
        offsets[k] = out_data + gguf_get_tensor_offset(dst.get(), moe[k].id);
    }
    set_index();
    std::vector<char> meta(out_data);
    gguf_get_meta_data(dst.get(), meta.data());
    if (gguf_get_meta_size(dst.get()) != out_data) {
        return { "internal error: meta size changed after filling index" };
    }

    std::ifstream fin(in_path, std::ios::binary);
    std::ofstream fout(tmp_path, std::ios::binary | std::ios::trunc);
    if (!fin.is_open()) {
        return { "cannot open input: " + in_path };
    }
    if (!fout.is_open()) {
        return { "cannot open output: " + tmp_path };
    }
    fout.write(meta.data(), (std::streamsize) meta.size());
    if (!fout.good()) {
        return { "write failed: " + tmp_path };
    }

    std::vector<char> buf(8u << 20);
    uint64_t          pos = out_data;
    for (int64_t i = 0; i < n_tensors; i++) {
        const uint64_t want  = out_data + gguf_get_tensor_offset(dst.get(), i);
        const uint64_t nbyte = gguf_get_tensor_size(dst.get(), i);
        if (want < pos || !write_zeros(fout, want - pos)) {
            return { "write failed: " + tmp_path };
        }
        const uint64_t from = in_data + gguf_get_tensor_offset(src.ctx.get(), i);
        if (!copy_bytes(fin, from, nbyte, fout, buf)) {
            return { std::string("copy failed for tensor ") + gguf_get_tensor_name(src.ctx.get(), i) };
        }
        pos = want + nbyte;
        // stock writer pads after every tensor, last one included
        const uint64_t padded = (pos + REPACK_ALIGNMENT - 1) / REPACK_ALIGNMENT * REPACK_ALIGNMENT;
        if (!write_zeros(fout, padded - pos)) {
            return { "write failed: " + tmp_path };
        }
        pos = padded;
    }
    fout.flush();
    fout.close();
    if (fout.fail()) {
        return { "write failed: " + tmp_path };
    }
    return {};
}

} // namespace

bool is_moe_expert_tensor(const std::string & name) {
    static const char * const bases[] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps" };
    if (name.compare(0, 4, "blk.") != 0) {
        return false;
    }
    size_t p = 4;
    while (p < name.size() && std::isdigit((unsigned char) name[p])) {
        p++;
    }
    if (p == 4 || p >= name.size() || name[p] != '.') {
        return false;
    }
    p++;
    for (const char * b : bases) {
        const std::string tail = std::string(b) + ".weight";
        if (name.compare(p, std::string::npos, tail) == 0) {
            return true;
        }
    }
    return false;
}

status repack_gguf(const std::string & in, const std::string & out) {
    if (in.empty() || out.empty()) {
        return { "input and output paths must not be empty" };
    }
    std::error_code ec;
    if (!fs::exists(in, ec)) {
        return { "input does not exist: " + in };
    }
    if (fs::path(in).lexically_normal() == fs::path(out).lexically_normal() ||
        (fs::exists(out, ec) && fs::equivalent(in, out, ec))) {
        return { "input and output are the same file" };
    }
    const std::string tmp = out + ".tmp";
    if (fs::exists(tmp, ec) && fs::equivalent(in, tmp, ec)) {
        return { "input is the temporary output file" };
    }

    status st = repack_to(in, tmp);
    if (!st.ok()) {
        fs::remove(tmp, ec);
        return st;
    }
    fs::rename(tmp, out, ec);
    if (ec) {
        const std::string msg = ec.message();
        fs::remove(tmp, ec);
        return { "rename to output failed: " + msg };
    }
    return {};
}

status plan_repack(const std::string & in, expert_index & idx) {
    idx = {};
    loaded_gguf g;
    if (status st = open_gguf(in, g); !st.ok()) {
        return st;
    }
    std::vector<moe_tensor> moe;
    if (status st = collect_moe(g.ctx.get(), g.data.get(), moe, idx.n_expert); !st.ok()) {
        return st;
    }
    idx.version = EXPERT_INDEX_VERSION;
    for (const moe_tensor & m : moe) {
        idx.tensors.push_back({ gguf_get_tensor_name(g.ctx.get(), m.id), 0, m.stride });
    }
    return {};
}

status read_expert_index(const std::string & path, expert_index & idx) {
    idx = {};
    loaded_gguf g;
    if (status st = open_gguf(path, g); !st.ok()) {
        return st;
    }
    const gguf_context * ctx = g.ctx.get();
    int64_t id_ver, id_names, id_offs, id_strides, id_n;
    if (!find_key(ctx, KEY_VERSION, GGUF_TYPE_UINT32, false, id_ver) ||
        !find_key(ctx, KEY_TENSORS, GGUF_TYPE_STRING, true, id_names) ||
        !find_key(ctx, KEY_OFFSETS, GGUF_TYPE_UINT64, true, id_offs) ||
        !find_key(ctx, KEY_STRIDES, GGUF_TYPE_UINT64, true, id_strides) ||
        !find_key(ctx, KEY_N_EXPERT, GGUF_TYPE_UINT32, false, id_n)) {
        return { "no gravity expert index in file (run llama-gravity-repack)" };
    }
    const uint32_t version = gguf_get_val_u32(ctx, id_ver);
    if (version != EXPERT_INDEX_VERSION) {
        return { "unsupported expert index version " + std::to_string(version) };
    }
    const size_t n = gguf_get_arr_n(ctx, id_names);
    if (gguf_get_arr_n(ctx, id_offs) != n || gguf_get_arr_n(ctx, id_strides) != n) {
        return { "expert index arrays have different lengths" };
    }
    const uint64_t * offs    = (const uint64_t *) gguf_get_arr_data(ctx, id_offs);
    const uint64_t * strides = (const uint64_t *) gguf_get_arr_data(ctx, id_strides);
    idx.version  = version;
    idx.n_expert = gguf_get_val_u32(ctx, id_n);
    for (size_t i = 0; i < n; i++) {
        idx.tensors.push_back({ gguf_get_arr_str(ctx, id_names, i), offs[i], strides[i] });
    }
    return {};
}

} // namespace gravity
