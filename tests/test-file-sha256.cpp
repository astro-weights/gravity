#include "common.h"

#include "hash/hash.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static void write_file(const std::string & path, const std::string & data) {
    std::ofstream f(path, std::ios::binary);
    f.write(data.data(), (std::streamsize) data.size());
    assert(f.good());
}

static std::string file_sha256(const std::string & path, uint64_t * size = nullptr) {
    std::string hex;
    std::string err;
    const bool ok = common_file_sha256(path, hex, err, size);
    if (!ok) {
        fprintf(stderr, "common_file_sha256(%s) failed: %s\n", path.c_str(), err.c_str());
    }
    assert(ok);
    assert(err.empty());
    return hex;
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "test-file-sha256";
    std::filesystem::create_directories(dir);

    // empty file
    {
        const std::string path = (dir / "empty.bin").string();
        write_file(path, "");
        uint64_t size = 1;
        assert(file_sha256(path, &size) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert(size == 0);
    }

    // "abc"
    {
        const std::string path = (dir / "abc.bin").string();
        write_file(path, "abc");
        uint64_t size = 0;
        assert(file_sha256(path, &size) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert(size == 3);
    }

    // larger than two 8 MiB chunks, with an odd tail
    {
        const size_t n = 2*8u*1024*1024 + 12345;
        std::string data(n, '\0');
        uint32_t x = 0x12345678;
        for (auto & c : data) {
            x = x*1664525u + 1013904223u;
            c = (char) (x >> 24);
        }
        const std::string path = (dir / "big.bin").string();
        write_file(path, data);
        uint64_t size = 0;
        const std::string actual = file_sha256(path, &size);
        assert(actual == hash_sha256_hex(data.data(), data.size()));
        assert(size == n);

        // one changed byte gives another hash
        data[n/2] ^= 1;
        write_file(path, data);
        assert(file_sha256(path) != actual);
    }

    // missing file
    {
        const std::string path = (dir / "does-not-exist.bin").string();
        std::filesystem::remove(path);
        std::string hex;
        std::string err;
        assert(!common_file_sha256(path, hex, err));
        assert(!err.empty());
        assert(hex.empty());
    }

    std::filesystem::remove_all(dir);
    printf("test-file-sha256: OK\n");
    return 0;
}
