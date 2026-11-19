#pragma once

#include <cstdint>
#include <string>

struct common_params;

// exactly 64 hex digits, case-insensitive; returns lower case, throws std::invalid_argument otherwise
std::string common_parse_sha256_hex(const std::string & value);

// sha256 of a file, read in chunks; on success out_hex is lowercase hex, on open/read failure returns false and sets out_err
bool common_file_sha256(const std::string & path, std::string & out_hex, std::string & out_err, uint64_t * out_size = nullptr);

// check the --model-sha256 / --lora-sha256 pins against the files on disk
// returns false (after LOG_ERR) on any error or mismatch; the caller must not load the model then
// warn_unpinned: log a warning for each file without a pin
bool common_verify_file_pins(const common_params & params, bool warn_unpinned = true);
