// gravity: llama-gguf-split --split / --merge on a 4096-aligned GGUF must produce files stock gguf_init accepts.
// The tool writes tensor data itself, so this covers the padding it picks, not just gguf_write_to_file.
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int n_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            n_fail++;                                                       \
        }                                                                   \
    } while (0)

static const int64_t NE[] = { 10, 100, 7, 1000, 33 }; // F32 elements, none a multiple of 4096 bytes

static std::vector<float> make_data(size_t i) {
    std::vector<float> v(NE[i]);
    for (size_t j = 0; j < v.size(); j++) {
        v[j] = (float) (i * 10000 + j);
    }
    return v;
}

static bool write_source(const std::string & path, size_t align) {
    ggml_init_params ip  = { 1 << 20, nullptr, false };
    ggml_context *   ctx = ggml_init(ip);
    gguf_context *   g   = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "test");
    if (align) {
        gguf_set_val_u32(g, "general.alignment", (uint32_t) align);
    }
    for (size_t i = 0; i < sizeof(NE) / sizeof(NE[0]); i++) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NE[i]);
        ggml_format_name(t, "t%zu", i);
        const std::vector<float> v = make_data(i);
        memcpy(t->data, v.data(), ggml_nbytes(t));
        gguf_add_tensor(g, t);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// loads `path` with stock gguf_init (data included), checks each tensor it holds against the source data
static int check_file(const std::string & path, size_t expect_align) {
    ggml_context *   ctx = nullptr;
    gguf_init_params ip  = { false, &ctx };
    gguf_context *   g   = gguf_init_from_file(path.c_str(), ip);
    if (!g) {
        fprintf(stderr, "FAIL: %s does not load\n", path.c_str());
        n_fail++;
        return -1;
    }
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(g, i);
        const size_t idx  = (size_t) atoi(name + 1);
        const ggml_tensor * t = ggml_get_tensor(ctx, name);
        const std::vector<float> ref = make_data(idx);
        CHECK(t && ggml_nbytes(t) == ref.size() * sizeof(float) && memcmp(t->data, ref.data(), ggml_nbytes(t)) == 0);
        if (expect_align) {
            CHECK(gguf_get_tensor_offset(g, i) % expect_align == 0);
        }
    }
    if (expect_align) {
        CHECK(gguf_get_alignment(g) == expect_align);
    }
    gguf_free(g);
    ggml_free(ctx);
    return (int) n;
}

static bool run(const std::string & tool, const std::string & args) {
    const std::string cmd = "\"" + tool + "\" " + args + " > /dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

static void test_split_merge(const std::string & tool, const fs::path & dir, size_t align) {
    const std::string tag = std::to_string(align);
    const fs::path    in  = dir / ("in" + tag + ".gguf");
    const fs::path    pfx = dir / ("sp" + tag);
    CHECK(write_source(in.string(), align));
    CHECK(check_file(in.string(), align) == (int) (sizeof(NE) / sizeof(NE[0])));

    // 2 tensors per split -> 3 splits; every split must load. Only split 1 carries the KVs.
    CHECK(run(tool, "--split --split-max-tensors 2 \"" + in.string() + "\" \"" + pfx.string() + "\""));
    int total = 0;
    for (int i = 1; i <= 3; i++) {
        char name[64];
        snprintf(name, sizeof(name), "-%05d-of-%05d.gguf", i, 3);
        const int n = check_file(pfx.string() + name, i == 1 ? align : 0);
        CHECK(n >= 1);
        total += n;
    }
    CHECK(total == (int) (sizeof(NE) / sizeof(NE[0])));

    const fs::path merged = dir / ("merged" + tag + ".gguf");
    CHECK(run(tool, "--merge \"" + pfx.string() + "-00001-of-00003.gguf\" \"" + merged.string() + "\""));
    CHECK(check_file(merged.string(), align) == (int) (sizeof(NE) / sizeof(NE[0])));
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s path/to/llama-gguf-split\n", argv[0]);
        return 1;
    }
    const std::string tool = argv[1];
    const fs::path    dir  = fs::temp_directory_path() / ("gguf-split-align-" + std::to_string((uintmax_t) (uintptr_t) &n_fail));
    std::error_code   ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    test_split_merge(tool, dir, 0);    // no key: default alignment, unchanged behavior
    test_split_merge(tool, dir, 4096); // what gravity-repack writes

    fs::remove_all(dir, ec);
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("ok\n");
    return 0;
}
