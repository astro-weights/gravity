#include "file-pins.h"

#include "common.h"
#include "log.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <vector>

std::string common_parse_sha256_hex(const std::string & value) {
    if (value.size() != 64) {
        throw std::invalid_argument(string_format("invalid sha256 '%s': expected 64 hex digits, got %zu characters", value.c_str(), value.size()));
    }
    std::string out = value;
    for (auto & c : out) {
        if (!std::isxdigit((unsigned char) c)) {
            throw std::invalid_argument(string_format("invalid sha256 '%s': not a hex digit: '%c'", value.c_str(), c));
        }
        c = (char) std::tolower((unsigned char) c);
    }
    return out;
}

bool common_file_sha256(const std::string & path, std::string & out_hex, std::string & out_err, uint64_t * out_size) {
    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (!f) {
        out_err = string_format("cannot open file: %s", strerror(errno));
        return false;
    }

    // 8 MiB chunks: never hold the whole file in memory
    std::vector<unsigned char> buf(8u*1024*1024);
    sha256_t ctx;
    sha256_init(&ctx);

    uint64_t n_total = 0;
    size_t n = 0;
    while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) {
        sha256_update(&ctx, buf.data(), n);
        n_total += n;
    }
    const bool failed = ferror(f) != 0;
    const int  err    = errno;
    fclose(f);
    if (failed) {
        out_err = string_format("cannot read file: %s", strerror(err));
        return false;
    }

    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);

    static const char hex[] = "0123456789abcdef";
    out_hex.clear();
    out_hex.reserve(2*SHA256_DIGEST_SIZE);
    for (unsigned char b : digest) {
        out_hex += hex[b >> 4];
        out_hex += hex[b & 0xf];
    }
    if (out_size) {
        *out_size = n_total;
    }
    return true;
}

// a "<name>-00001-of-00004.gguf" shard: the loader also reads the other shards
static bool common_is_split_model_path(const std::string & path) {
    static const std::regex re("-[0-9]{5,}-of-[0-9]{5,}\\.gguf$", std::regex::icase);
    return std::regex_search(path, re);
}

bool common_verify_file_pins(const common_params & params, bool warn_unpinned) {
    const auto & loras     = params.lora_adapters;
    const auto & lora_pins = params.lora_sha256;

    if (!lora_pins.empty() && lora_pins.size() != loras.size()) {
        LOG_ERR("%s: got %zu --lora-sha256 values for %zu LoRA adapters, give one per --lora / --lora-scaled entry in the same order\n",
                __func__, lora_pins.size(), loras.size());
        return false;
    }

    struct file_pin {
        const char * kind;
        std::string  path;
        std::string  expected;
    };
    std::vector<file_pin> pins;

    if (params.model_sha256.empty()) {
        if (warn_unpinned) {
            LOG_WRN("%s: model not pinned (--model-sha256)\n", __func__);
        }
    } else {
        if (params.model.path.empty()) {
            LOG_ERR("%s: --model-sha256 is given but there is no local model file to check\n", __func__);
            return false;
        }
        // a pin on the first shard only would leave the other shards unchecked
        if (!params.model.shards.empty() || common_is_split_model_path(params.model.path)) {
            LOG_ERR("%s: --model-sha256 is not supported for split models (%s)\n", __func__, params.model.path.c_str());
            return false;
        }
        pins.push_back({ "model", params.model.path, params.model_sha256 });
    }

    for (size_t i = 0; i < loras.size(); ++i) {
        if (lora_pins.empty()) {
            if (warn_unpinned) {
                LOG_WRN("%s: LoRA adapter not pinned (--lora-sha256): %s\n", __func__, loras[i].path.c_str());
            }
        } else {
            pins.push_back({ "LoRA adapter", loras[i].path, lora_pins[i] });
        }
    }

    if (pins.empty()) {
        return true;
    }

    const auto t_start = std::chrono::steady_clock::now();
    uint64_t n_bytes = 0;

    for (const auto & pin : pins) {
        std::string actual;
        std::string err;
        uint64_t    size = 0;
        if (!common_file_sha256(pin.path, actual, err, &size)) {
            LOG_ERR("%s: cannot hash %s file '%s': %s\n", __func__, pin.kind, pin.path.c_str(), err.c_str());
            return false;
        }
        n_bytes += size;
        if (actual != pin.expected) {
            LOG_ERR("%s: sha256 mismatch for %s file '%s': expected %s, actual %s\n",
                    __func__, pin.kind, pin.path.c_str(), pin.expected.c_str(), actual.c_str());
            return false;
        }
    }

    const double t_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    LOG_INF("%s: sha256 of %zu pinned file(s) ok, %.2f GB in %.1f s\n", __func__, pins.size(), n_bytes/1e9, t_s);

    return true;
}
