#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

struct U64x {
    uint32_t lo;
    uint32_t hi;
};

__device__ __forceinline__ U64x make_u64x(uint32_t lo, uint32_t hi) {
    return U64x{lo, hi};
}

__device__ __forceinline__ U64x xor64(U64x a, U64x b) {
    return make_u64x(a.lo ^ b.lo, a.hi ^ b.hi);
}

__device__ __forceinline__ U64x and64(U64x a, U64x b) {
    return make_u64x(a.lo & b.lo, a.hi & b.hi);
}

__device__ __forceinline__ U64x not64(U64x a) {
    return make_u64x(~a.lo, ~a.hi);
}

__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return ((x & 0xff000000u) >> 24) |
           ((x & 0x00ff0000u) >> 8) |
           ((x & 0x0000ff00u) << 8) |
           ((x & 0x000000ffu) << 24);
}

__device__ __forceinline__ U64x rotl64(U64x x, uint32_t n) {
    if (n == 0) {
        return x;
    } else if (n < 32) {
        return make_u64x((x.lo << n) | (x.hi >> (32 - n)),
                         (x.hi << n) | (x.lo >> (32 - n)));
    } else if (n == 32) {
        return make_u64x(x.hi, x.lo);
    } else {
        uint32_t r = n - 32;
        return make_u64x((x.hi << r) | (x.lo >> (32 - r)),
                         (x.lo << r) | (x.hi >> (32 - r)));
    }
}

__constant__ uint32_t K_RC_LO[24] = {
    0x00000001u, 0x00008082u, 0x0000808au, 0x80008000u,
    0x0000808bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008au, 0x00000088u, 0x80008009u, 0x8000000au,
    0x8000808bu, 0x0000008bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800au, 0x8000000au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u
};

__constant__ uint32_t K_RC_HI[24] = {
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u
};

__constant__ uint32_t K_ROT[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

__device__ __forceinline__ void keccakf(U64x* s) {
#pragma unroll 24
    for (uint32_t round = 0; round < 24; round++) {
        U64x c0 = xor64(xor64(xor64(xor64(s[0], s[5]), s[10]), s[15]), s[20]);
        U64x c1 = xor64(xor64(xor64(xor64(s[1], s[6]), s[11]), s[16]), s[21]);
        U64x c2 = xor64(xor64(xor64(xor64(s[2], s[7]), s[12]), s[17]), s[22]);
        U64x c3 = xor64(xor64(xor64(xor64(s[3], s[8]), s[13]), s[18]), s[23]);
        U64x c4 = xor64(xor64(xor64(xor64(s[4], s[9]), s[14]), s[19]), s[24]);

        U64x d0 = xor64(c4, rotl64(c1, 1));
        U64x d1 = xor64(c0, rotl64(c2, 1));
        U64x d2 = xor64(c1, rotl64(c3, 1));
        U64x d3 = xor64(c2, rotl64(c4, 1));
        U64x d4 = xor64(c3, rotl64(c0, 1));

#pragma unroll
        for (int y = 0; y < 25; y += 5) {
            s[y + 0] = xor64(s[y + 0], d0);
            s[y + 1] = xor64(s[y + 1], d1);
            s[y + 2] = xor64(s[y + 2], d2);
            s[y + 3] = xor64(s[y + 3], d3);
            s[y + 4] = xor64(s[y + 4], d4);
        }

        U64x b[25];
#pragma unroll
        for (uint32_t x = 0; x < 5; x++) {
#pragma unroll
            for (uint32_t y = 0; y < 5; y++) {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(s[x + 5 * y], K_ROT[x + 5 * y]);
            }
        }

#pragma unroll
        for (uint32_t y = 0; y < 5; y++) {
            uint32_t row = 5 * y;
            U64x b0 = b[row + 0];
            U64x b1 = b[row + 1];
            U64x b2 = b[row + 2];
            U64x b3 = b[row + 3];
            U64x b4 = b[row + 4];
            s[row + 0] = xor64(b0, and64(not64(b1), b2));
            s[row + 1] = xor64(b1, and64(not64(b2), b3));
            s[row + 2] = xor64(b2, and64(not64(b3), b4));
            s[row + 3] = xor64(b3, and64(not64(b4), b0));
            s[row + 4] = xor64(b4, and64(not64(b0), b1));
        }

        s[0].lo ^= K_RC_LO[round];
        s[0].hi ^= K_RC_HI[round];
    }
}

__global__ void mine_kernel(
    const uint32_t* __restrict__ challenge,
    const uint32_t* __restrict__ target,
    const uint32_t* __restrict__ nonce_base,
    uint32_t* __restrict__ found,
    uint32_t* __restrict__ result,
    uint32_t nonces_per_thread
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t offset = tid * (uint64_t)nonces_per_thread;

    uint32_t base_lo = nonce_base[0] + (uint32_t)offset;
    uint32_t carry0 = base_lo < nonce_base[0] ? 1u : 0u;
    uint32_t offset_hi = (uint32_t)(offset >> 32);
    uint32_t base_hi_tmp = nonce_base[1] + offset_hi;
    uint32_t carry1a = base_hi_tmp < nonce_base[1] ? 1u : 0u;
    uint32_t base_hi = base_hi_tmp + carry0;
    uint32_t carry1b = base_hi < base_hi_tmp ? 1u : 0u;
    uint32_t carry1 = carry1a | carry1b;
    uint32_t base_nhi_lo = nonce_base[2] + carry1;
    uint32_t carry2 = base_nhi_lo < nonce_base[2] ? 1u : 0u;
    uint32_t base_nhi_hi = nonce_base[3] + carry2;

#pragma unroll 1
    for (uint32_t iter = 0; iter < nonces_per_thread; iter++) {
        if (*found != 0u) return;

        uint32_t nlo_lo = base_lo + iter;
        uint32_t c0 = nlo_lo < base_lo ? 1u : 0u;
        uint32_t nlo_hi = base_hi + c0;
        uint32_t c1 = nlo_hi < base_hi ? 1u : 0u;
        uint32_t nhi_lo = base_nhi_lo + c1;
        uint32_t c2 = nhi_lo < base_nhi_lo ? 1u : 0u;
        uint32_t nhi_hi = base_nhi_hi + c2;

        U64x s[25];
#pragma unroll
        for (int i = 0; i < 25; i++) s[i] = make_u64x(0, 0);

        s[0] = make_u64x(challenge[0], challenge[1]);
        s[1] = make_u64x(challenge[2], challenge[3]);
        s[2] = make_u64x(challenge[4], challenge[5]);
        s[3] = make_u64x(challenge[6], challenge[7]);

        s[6] = make_u64x(bswap32(nhi_hi), bswap32(nhi_lo));
        s[7] = make_u64x(bswap32(nlo_hi), bswap32(nlo_lo));
        s[8].lo = 0x00000001u;
        s[16].hi = 0x80000000u;

        keccakf(s);

        bool decided = false;
        bool less = false;
#pragma unroll
        for (int i = 0; i < 4 && !decided; i++) {
            uint32_t h_hi = bswap32(s[i].lo);
            uint32_t h_lo = bswap32(s[i].hi);
            uint32_t t_hi = target[i * 2];
            uint32_t t_lo = target[i * 2 + 1];
            if (h_hi < t_hi) {
                less = true;
                decided = true;
            } else if (h_hi > t_hi) {
                less = false;
                decided = true;
            } else if (h_lo < t_lo) {
                less = true;
                decided = true;
            } else if (h_lo > t_lo) {
                less = false;
                decided = true;
            }
        }

        if (less) {
            if (atomicCAS(found, 0u, 1u) == 0u) {
                result[0] = nlo_lo;
                result[1] = nlo_hi;
                result[2] = nhi_lo;
                result[3] = nhi_hi;
            }
            return;
        }
    }
}

static uint8_t hex_value(char c) {
    if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
    if (c >= 'a' && c <= 'f') return (uint8_t)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return (uint8_t)(c - 'A' + 10);
    throw std::runtime_error("bad hex digit");
}

static std::array<uint8_t, 32> parse_hex32(std::string s) {
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
    if (s.size() > 64) throw std::runtime_error("hex value longer than 32 bytes");
    if (s.size() % 2 != 0) s = "0" + s;
    while (s.size() < 64) s = "00" + s;

    std::array<uint8_t, 32> out{};
    for (size_t i = 0; i < 32; i++) {
        out[i] = (uint8_t)((hex_value(s[i * 2]) << 4) | hex_value(s[i * 2 + 1]));
    }
    return out;
}

static std::array<uint32_t, 8> le_words32(const std::array<uint8_t, 32>& bytes) {
    std::array<uint32_t, 8> out{};
    for (int i = 0; i < 8; i++) {
        out[i] = (uint32_t)bytes[i * 4] |
                 ((uint32_t)bytes[i * 4 + 1] << 8) |
                 ((uint32_t)bytes[i * 4 + 2] << 16) |
                 ((uint32_t)bytes[i * 4 + 3] << 24);
    }
    return out;
}

static std::array<uint32_t, 8> be_words32(const std::array<uint8_t, 32>& bytes) {
    std::array<uint32_t, 8> out{};
    for (int i = 0; i < 8; i++) {
        out[i] = ((uint32_t)bytes[i * 4] << 24) |
                 ((uint32_t)bytes[i * 4 + 1] << 16) |
                 ((uint32_t)bytes[i * 4 + 2] << 8) |
                 (uint32_t)bytes[i * 4 + 3];
    }
    return out;
}

static uint64_t rotl64_cpu(uint64_t x, unsigned n) {
    if (n == 0) return x;
    return (x << n) | (x >> (64 - n));
}

static void keccakf_cpu(uint64_t st[25]) {
    static const uint64_t rc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };
    static const unsigned rot[25] = {
         0,  1, 62, 28, 27,
        36, 44,  6, 55, 20,
         3, 10, 43, 25, 39,
        41, 45, 15, 21,  8,
        18,  2, 61, 56, 14
    };

    for (int round = 0; round < 24; round++) {
        uint64_t c[5], b[25];
        for (int x = 0; x < 5; x++) c[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (int x = 0; x < 5; x++) {
            uint64_t d = c[(x+4) % 5] ^ rotl64_cpu(c[(x+1) % 5], 1);
            for (int y = 0; y < 5; y++) st[x + 5*y] ^= d;
        }
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                b[y + 5 * ((2*x + 3*y) % 5)] = rotl64_cpu(st[x + 5*y], rot[x + 5*y]);
            }
        }
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                st[x + 5*y] = b[x + 5*y] ^ ((~b[((x+1) % 5) + 5*y]) & b[((x+2) % 5) + 5*y]);
            }
        }
        st[0] ^= rc[round];
    }
}

static std::array<uint8_t, 32> keccak_cpu(const std::array<uint8_t, 32>& challenge, unsigned __int128 nonce) {
    uint8_t buf[136]{};
    std::memcpy(buf, challenge.data(), 32);
    for (int i = 0; i < 16; i++) {
        buf[63 - i] = (uint8_t)(nonce & 0xff);
        nonce >>= 8;
    }
    buf[64] = 0x01;
    buf[135] ^= 0x80;

    uint64_t st[25]{};
    for (int i = 0; i < 17; i++) {
        uint64_t w = 0;
        for (int j = 0; j < 8; j++) w |= (uint64_t)buf[i * 8 + j] << (8 * j);
        st[i] ^= w;
    }
    keccakf_cpu(st);

    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; i++) {
        uint64_t w = st[i];
        for (int j = 0; j < 8; j++) out[i * 8 + j] = (uint8_t)((w >> (8 * j)) & 0xff);
    }
    return out;
}

static bool hash_lt_target(const std::array<uint8_t, 32>& hash, const std::array<uint8_t, 32>& target) {
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return false;
}

static std::string hex_encode(const std::array<uint8_t, 32>& bytes) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint8_t b : bytes) oss << std::setw(2) << (unsigned)b;
    return oss.str();
}

static uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::strtoull(v, nullptr, 10);
}

static double parse_bench_seconds(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        const char* prefix = "--bench-seconds=";
        size_t n = std::strlen(prefix);
        if (std::strncmp(argv[i], prefix, n) == 0) {
            return std::strtod(argv[i] + n, nullptr);
        }
    }
    return 0.0;
}

static bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], flag) == 0) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: cuda-miner <challenge-hex32> <difficulty-hex32> [--selftest] [--bench-seconds=N]\n");
        return 2;
    }

    auto challenge_bytes = parse_hex32(argv[1]);
    auto difficulty_bytes = parse_hex32(argv[2]);
    bool selftest = has_flag(argc, argv, "--selftest");
    double bench_seconds = parse_bench_seconds(argc, argv);

    int device_id = (int)env_u64("CUDA_DEVICE", 0);
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::fprintf(stderr, "device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::fprintf(stderr, "challenge:  0x%s\n", hex_encode(challenge_bytes).c_str());
    std::fprintf(stderr, "difficulty: 0x%s\n", hex_encode(difficulty_bytes).c_str());

    uint32_t block_threads = (uint32_t)env_u64("CUDA_BLOCK_THREADS", 256);
    uint32_t nonces_per_thread = (uint32_t)env_u64("CUDA_NONCES_PER_THREAD", 1);
    if (block_threads == 0) block_threads = 256;
    if (nonces_per_thread == 0) nonces_per_thread = 1;
    uint32_t batch_log2 = (uint32_t)env_u64("CUDA_BATCH_LOG2", prop.major >= 9 ? 29 : 28);
    if (batch_log2 < 12) batch_log2 = 12;
    if (batch_log2 > 32) batch_log2 = 32;
    uint64_t batch = 1ULL << batch_log2;
    uint64_t threads_per_dispatch = batch / nonces_per_thread;
    if (threads_per_dispatch < block_threads) threads_per_dispatch = block_threads;
    threads_per_dispatch -= threads_per_dispatch % block_threads;
    batch = threads_per_dispatch * nonces_per_thread;
    uint32_t grid_blocks = (uint32_t)(threads_per_dispatch / block_threads);
    std::fprintf(stderr, "blocks=%u threads/block=%u batch=%lluM nonces/thread=%u\n",
                 grid_blocks, block_threads, (unsigned long long)(batch / 1000000ULL), nonces_per_thread);

    auto challenge_words = le_words32(challenge_bytes);
    auto target_words = be_words32(difficulty_bytes);
    if (selftest) {
        difficulty_bytes.fill(0xff);
        target_words = be_words32(difficulty_bytes);
    }

    uint32_t *d_challenge = nullptr, *d_target = nullptr, *d_nonce_base = nullptr, *d_found = nullptr, *d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_challenge, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_target, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nonce_base, 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result, 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_challenge, challenge_words.data(), 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target_words.data(), 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    std::mt19937_64 rng((uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uint64_t nonce_lo = selftest ? 42ULL : rng();
    uint64_t nonce_hi = 0;

    uint64_t total = 0;
    auto start = std::chrono::steady_clock::now();
    auto last_report = start;

    while (true) {
        uint32_t found = 0;
        uint32_t result[4]{};
        uint32_t nonce_base[4] = {
            (uint32_t)nonce_lo,
            (uint32_t)(nonce_lo >> 32),
            (uint32_t)nonce_hi,
            (uint32_t)(nonce_hi >> 32),
        };
        CUDA_CHECK(cudaMemcpy(d_found, &found, sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nonce_base, nonce_base, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        if (selftest) {
            mine_kernel<<<1, 1>>>(d_challenge, d_target, d_nonce_base, d_found, d_result, 1);
        } else {
            mine_kernel<<<grid_blocks, block_threads>>>(d_challenge, d_target, d_nonce_base, d_found, d_result, nonces_per_thread);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t dispatched = selftest ? 1 : batch;
        total += dispatched;
        CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (found == 1) {
            CUDA_CHECK(cudaMemcpy(result, d_result, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint64_t nlo = (uint64_t)result[0] | ((uint64_t)result[1] << 32);
            uint64_t nhi = (uint64_t)result[2] | ((uint64_t)result[3] << 32);
            unsigned __int128 nonce = ((unsigned __int128)nhi << 64) | nlo;
            auto h = keccak_cpu(challenge_bytes, nonce);
            bool ok = hash_lt_target(h, difficulty_bytes);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            double rate = (double)total / elapsed / 1e9;
            std::fprintf(stderr, "FOUND in %.2fs (%llu MH, %.2f GH/s) -- hash=0x%s -- verify=%s\n",
                         elapsed, (unsigned long long)(total / 1000000ULL), rate,
                         hex_encode(h).c_str(), ok ? "OK" : "MISMATCH!");
            if (!ok) {
                std::fprintf(stderr, "Kernel produced bogus result -- refusing to submit\n");
                return 3;
            }
            if (selftest && nlo != 42ULL) {
                std::fprintf(stderr, "FAIL: expected nonce=42, got %llu\n", (unsigned long long)nlo);
                return 1;
            }
            if (selftest) {
                std::fprintf(stderr, "selftest OK\n");
            } else {
                if (nhi != 0) {
                    std::fprintf(stderr, "nonce high 64 bits are non-zero; decimal output is truncated\n");
                }
                std::cout << (unsigned long long)nlo << "\n";
            }
            return 0;
        }

        unsigned __int128 next = ((unsigned __int128)nonce_hi << 64) | nonce_lo;
        next += (unsigned __int128)batch;
        nonce_lo = (uint64_t)next;
        nonce_hi = (uint64_t)(next >> 64);

        auto now = std::chrono::steady_clock::now();
        double since_report = std::chrono::duration<double>(now - last_report).count();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (since_report >= 2.0) {
            double rate = (double)total / elapsed / 1e9;
            std::fprintf(stderr, "[%.1fs] %llu MH searched, %.2f GH/s\n",
                         elapsed, (unsigned long long)(total / 1000000ULL), rate);
            last_report = now;
        }
        if (bench_seconds > 0.0 && elapsed >= bench_seconds) {
            double rate = (double)total / elapsed / 1e9;
            std::fprintf(stderr, "BENCH %.2fs: %llu MH searched, %.3f GH/s\n",
                         elapsed, (unsigned long long)(total / 1000000ULL), rate);
            return 0;
        }
    }
}
