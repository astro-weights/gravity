#pragma once

// gravity streaming S3: rewrite a MoE GGUF with 4096 alignment + per-expert offset index.
// Tensors keep name/type/shape/bytes/order, so stock llama.cpp still loads the output.
// Layout idea: Strata (Niko1221/Strata, MIT). No Strata code is used.

#include <cstdint>
#include <string>
#include <vector>

namespace gravity {

constexpr uint32_t REPACK_ALIGNMENT     = 4096;
constexpr uint32_t EXPERT_INDEX_VERSION = 1;

// no exceptions across the API: error is empty on success
struct status {
    std::string error;
    bool ok() const { return error.empty(); }
};

// per MoE tensor, file order. Expert e starts at offset + e * stride
struct expert_index_entry {
    std::string name;
    uint64_t    offset = 0; // absolute file offset of tensor data
    uint64_t    stride = 0; // bytes per expert slice
};

struct expert_index {
    uint32_t                        version  = 0;
    uint32_t                        n_expert = 0;
    std::vector<expert_index_entry> tensors;
};

// true for blk.N.ffn_{gate,up,down,gate_up}_exps.weight
bool is_moe_expert_tensor(const std::string & name);

// refuses in == out and non-MoE input. Writes out + ".tmp" then renames; tmp removed on failure
status repack_gguf(const std::string & in, const std::string & out);

// dry run: fills name + stride + n_expert of the MoE tensors (offsets stay 0), same refusals as repack_gguf
status plan_repack(const std::string & in, expert_index & idx);

// reads the gravity.expert_index.* keys of a repacked file
status read_expert_index(const std::string & path, expert_index & idx);

} // namespace gravity
