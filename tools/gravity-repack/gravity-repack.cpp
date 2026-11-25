// llama-gravity-repack: align MoE GGUF to 4096 and add a per-expert offset index
#include "repack.h"

#include <cstdio>
#include <cstring>
#include <string>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s IN OUT [--dry-run]\n"
        "\n"
        "Rewrite a MoE GGUF with general.alignment = 4096 and gravity.expert_index.* keys.\n"
        "Tensors keep name, type, shape, bytes and order: stock llama.cpp still loads OUT.\n"
        "Run it LAST: llama-quantize and gguf-split copy the 4096 alignment and index keys, so\n"
        "their output loads but the copied index goes stale when tensors change: quantize or split first, then repack.\n"
        "\n"
        "  --dry-run   print tensor, stride, 4096-aligned yes/no; write nothing\n"
        "\n"
        "Layout idea: Strata (Niko1221/Strata, MIT)\n",
        argv0);
}

static int dry_run(const std::string & in) {
    gravity::expert_index idx;
    if (gravity::status st = gravity::plan_repack(in, idx); !st.ok()) {
        fprintf(stderr, "error: %s\n", st.error.c_str());
        return 1;
    }
    size_t n_aligned = 0;
    printf("%-40s %14s %s\n", "tensor", "stride", "aligned");
    for (const auto & e : idx.tensors) {
        const bool ok = e.stride % gravity::REPACK_ALIGNMENT == 0;
        n_aligned += ok;
        printf("%-40s %14llu %s\n", e.name.c_str(), (unsigned long long) e.stride, ok ? "yes" : "no");
    }
    printf("%zu MoE tensors, %zu with 4096-aligned expert slices, n_expert %u\n", idx.tensors.size(), n_aligned, idx.n_expert);
    return 0;
}

int main(int argc, char ** argv) {
    std::string in, out;
    bool        dry = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_usage(argv[0]);
            return 0;
        }
        if (!strcmp(argv[i], "--dry-run")) {
            dry = true;
        } else if (in.empty()) {
            in = argv[i];
        } else if (out.empty()) {
            out = argv[i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (in.empty() || (out.empty() && !dry)) {
        print_usage(argv[0]);
        return 1;
    }
    if (dry) {
        return dry_run(in);
    }
    if (gravity::status st = gravity::repack_gguf(in, out); !st.ok()) {
        fprintf(stderr, "error: %s\n", st.error.c_str());
        return 1;
    }
    return 0;
}
