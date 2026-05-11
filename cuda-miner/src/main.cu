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
    return __byte_perm(x, 0, 0x0123);
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

template <uint32_t N>
__device__ __forceinline__ U64x rotl64c(U64x x) {
    if constexpr (N == 0) {
        return x;
    } else if constexpr (N < 32) {
        return make_u64x(__funnelshift_l(x.hi, x.lo, N),
                         __funnelshift_l(x.lo, x.hi, N));
    } else if constexpr (N == 32) {
        return make_u64x(x.hi, x.lo);
    } else {
        constexpr uint32_t R = N - 32;
        return make_u64x(__funnelshift_l(x.lo, x.hi, R),
                         __funnelshift_l(x.hi, x.lo, R));
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

        U64x d0 = xor64(c4, rotl64c<1>(c1));
        U64x d1 = xor64(c0, rotl64c<1>(c2));
        U64x d2 = xor64(c1, rotl64c<1>(c3));
        U64x d3 = xor64(c2, rotl64c<1>(c4));
        U64x d4 = xor64(c3, rotl64c<1>(c0));

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

__device__ __forceinline__ void keccakf_regs(
    U64x& a00, U64x& a10, U64x& a20, U64x& a30, U64x& a40,
    U64x& a01, U64x& a11, U64x& a21, U64x& a31, U64x& a41,
    U64x& a02, U64x& a12, U64x& a22, U64x& a32, U64x& a42,
    U64x& a03, U64x& a13, U64x& a23, U64x& a33, U64x& a43,
    U64x& a04, U64x& a14, U64x& a24, U64x& a34, U64x& a44
) {
#pragma unroll 24
    for (uint32_t round = 0; round < 24; round++) {
        U64x c0 = xor64(xor64(xor64(xor64(a00, a01), a02), a03), a04);
        U64x c1 = xor64(xor64(xor64(xor64(a10, a11), a12), a13), a14);
        U64x c2 = xor64(xor64(xor64(xor64(a20, a21), a22), a23), a24);
        U64x c3 = xor64(xor64(xor64(xor64(a30, a31), a32), a33), a34);
        U64x c4 = xor64(xor64(xor64(xor64(a40, a41), a42), a43), a44);

        U64x d0 = xor64(c4, rotl64c<1>(c1));
        U64x d1 = xor64(c0, rotl64c<1>(c2));
        U64x d2 = xor64(c1, rotl64c<1>(c3));
        U64x d3 = xor64(c2, rotl64c<1>(c4));
        U64x d4 = xor64(c3, rotl64c<1>(c0));

        a00 = xor64(a00, d0); a01 = xor64(a01, d0); a02 = xor64(a02, d0); a03 = xor64(a03, d0); a04 = xor64(a04, d0);
        a10 = xor64(a10, d1); a11 = xor64(a11, d1); a12 = xor64(a12, d1); a13 = xor64(a13, d1); a14 = xor64(a14, d1);
        a20 = xor64(a20, d2); a21 = xor64(a21, d2); a22 = xor64(a22, d2); a23 = xor64(a23, d2); a24 = xor64(a24, d2);
        a30 = xor64(a30, d3); a31 = xor64(a31, d3); a32 = xor64(a32, d3); a33 = xor64(a33, d3); a34 = xor64(a34, d3);
        a40 = xor64(a40, d4); a41 = xor64(a41, d4); a42 = xor64(a42, d4); a43 = xor64(a43, d4); a44 = xor64(a44, d4);

        U64x b00 = rotl64c<0>(a00);
        U64x b13 = rotl64c<36>(a01);
        U64x b21 = rotl64c<3>(a02);
        U64x b34 = rotl64c<41>(a03);
        U64x b42 = rotl64c<18>(a04);
        U64x b02 = rotl64c<1>(a10);
        U64x b10 = rotl64c<44>(a11);
        U64x b23 = rotl64c<10>(a12);
        U64x b31 = rotl64c<45>(a13);
        U64x b44 = rotl64c<2>(a14);
        U64x b04 = rotl64c<62>(a20);
        U64x b12 = rotl64c<6>(a21);
        U64x b20 = rotl64c<43>(a22);
        U64x b33 = rotl64c<15>(a23);
        U64x b41 = rotl64c<61>(a24);
        U64x b01 = rotl64c<28>(a30);
        U64x b14 = rotl64c<55>(a31);
        U64x b22 = rotl64c<25>(a32);
        U64x b30 = rotl64c<21>(a33);
        U64x b43 = rotl64c<56>(a34);
        U64x b03 = rotl64c<27>(a40);
        U64x b11 = rotl64c<20>(a41);
        U64x b24 = rotl64c<39>(a42);
        U64x b32 = rotl64c<8>(a43);
        U64x b40 = rotl64c<14>(a44);

        a00 = xor64(b00, and64(not64(b10), b20));
        a10 = xor64(b10, and64(not64(b20), b30));
        a20 = xor64(b20, and64(not64(b30), b40));
        a30 = xor64(b30, and64(not64(b40), b00));
        a40 = xor64(b40, and64(not64(b00), b10));

        a01 = xor64(b01, and64(not64(b11), b21));
        a11 = xor64(b11, and64(not64(b21), b31));
        a21 = xor64(b21, and64(not64(b31), b41));
        a31 = xor64(b31, and64(not64(b41), b01));
        a41 = xor64(b41, and64(not64(b01), b11));

        a02 = xor64(b02, and64(not64(b12), b22));
        a12 = xor64(b12, and64(not64(b22), b32));
        a22 = xor64(b22, and64(not64(b32), b42));
        a32 = xor64(b32, and64(not64(b42), b02));
        a42 = xor64(b42, and64(not64(b02), b12));

        a03 = xor64(b03, and64(not64(b13), b23));
        a13 = xor64(b13, and64(not64(b23), b33));
        a23 = xor64(b23, and64(not64(b33), b43));
        a33 = xor64(b33, and64(not64(b43), b03));
        a43 = xor64(b43, and64(not64(b03), b13));

        a04 = xor64(b04, and64(not64(b14), b24));
        a14 = xor64(b14, and64(not64(b24), b34));
        a24 = xor64(b24, and64(not64(b34), b44));
        a34 = xor64(b34, and64(not64(b44), b04));
        a44 = xor64(b44, and64(not64(b04), b14));

        a00.lo ^= K_RC_LO[round];
        a00.hi ^= K_RC_HI[round];
    }
}

template <uint32_t FIXED_NONCES_PER_THREAD, uint32_t FIXED_NONCE_STRIDE>
__global__ void mine_kernel(
    const uint32_t* __restrict__ challenge,
    const uint32_t* __restrict__ target,
    const uint32_t* __restrict__ nonce_base,
    uint32_t* __restrict__ found,
    uint32_t* __restrict__ result,
    uint32_t dynamic_nonces_per_thread,
    uint32_t dynamic_nonce_stride
) {
    constexpr bool fixed_nonces_per_thread = FIXED_NONCES_PER_THREAD != 0;
    constexpr bool fixed_nonce_stride = FIXED_NONCE_STRIDE != 0;
    const uint32_t nonces_per_thread = fixed_nonces_per_thread
        ? FIXED_NONCES_PER_THREAD
        : dynamic_nonces_per_thread;
    const uint32_t nonce_stride = fixed_nonce_stride
        ? FIXED_NONCE_STRIDE
        : dynamic_nonce_stride;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t offset = tid * (uint64_t)nonces_per_thread * (uint64_t)nonce_stride;

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
        if constexpr (FIXED_NONCES_PER_THREAD != 1) {
            if (*found != 0u) return;
        }

        uint64_t iter_offset = (uint64_t)iter * (uint64_t)nonce_stride;
        uint32_t nlo_lo = base_lo + (uint32_t)iter_offset;
        uint32_t c0 = nlo_lo < base_lo ? 1u : 0u;
        uint32_t iter_offset_hi = (uint32_t)(iter_offset >> 32);
        uint32_t nlo_hi_tmp = base_hi + iter_offset_hi;
        uint32_t c1a = nlo_hi_tmp < base_hi ? 1u : 0u;
        uint32_t nlo_hi = nlo_hi_tmp + c0;
        uint32_t c1b = nlo_hi < nlo_hi_tmp ? 1u : 0u;
        uint32_t c1 = c1a | c1b;
        uint32_t nhi_lo = base_nhi_lo + c1;
        uint32_t c2 = nhi_lo < base_nhi_lo ? 1u : 0u;
        uint32_t nhi_hi = base_nhi_hi + c2;

        U64x a00 = make_u64x(challenge[0], challenge[1]);
        U64x a10 = make_u64x(challenge[2], challenge[3]);
        U64x a20 = make_u64x(challenge[4], challenge[5]);
        U64x a30 = make_u64x(challenge[6], challenge[7]);
        U64x a40 = make_u64x(0, 0);
        U64x a01 = make_u64x(0, 0);
        U64x a11 = make_u64x(bswap32(nhi_hi), bswap32(nhi_lo));
        U64x a21 = make_u64x(bswap32(nlo_hi), bswap32(nlo_lo));
        U64x a31 = make_u64x(0x00000001u, 0);
        U64x a41 = make_u64x(0, 0);
        U64x a02 = make_u64x(0, 0);
        U64x a12 = make_u64x(0, 0);
        U64x a22 = make_u64x(0, 0);
        U64x a32 = make_u64x(0, 0);
        U64x a42 = make_u64x(0, 0);
        U64x a03 = make_u64x(0, 0);
        U64x a13 = make_u64x(0, 0x80000000u);
        U64x a23 = make_u64x(0, 0);
        U64x a33 = make_u64x(0, 0);
        U64x a43 = make_u64x(0, 0);
        U64x a04 = make_u64x(0, 0);
        U64x a14 = make_u64x(0, 0);
        U64x a24 = make_u64x(0, 0);
        U64x a34 = make_u64x(0, 0);
        U64x a44 = make_u64x(0, 0);

        keccakf_regs(
            a00, a10, a20, a30, a40,
            a01, a11, a21, a31, a41,
            a02, a12, a22, a32, a42,
            a03, a13, a23, a33, a43,
            a04, a14, a24, a34, a44
        );

        bool less = false;
        uint32_t t0 = target[0];
        if (t0 == 0u) {
            if (a00.lo == 0u) {
                uint32_t h1 = bswap32(a00.hi);
                if (h1 < target[1]) {
                    less = true;
                } else if (h1 == target[1]) {
                    uint32_t h2 = bswap32(a10.lo);
                    if (h2 < target[2]) {
                        less = true;
                    } else if (h2 == target[2]) {
                        uint32_t h3 = bswap32(a10.hi);
                        if (h3 < target[3]) {
                            less = true;
                        } else if (h3 == target[3]) {
                            uint32_t h4 = bswap32(a20.lo);
                            if (h4 < target[4]) {
                                less = true;
                            } else if (h4 == target[4]) {
                                uint32_t h5 = bswap32(a20.hi);
                                if (h5 < target[5]) {
                                    less = true;
                                } else if (h5 == target[5]) {
                                    uint32_t h6 = bswap32(a30.lo);
                                    if (h6 < target[6]) {
                                        less = true;
                                    } else if (h6 == target[6]) {
                                        uint32_t h7 = bswap32(a30.hi);
                                        less = h7 < target[7];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            uint32_t h0 = bswap32(a00.lo);
            if (h0 < t0) {
                less = true;
            } else if (h0 == t0) {
                uint32_t h1 = bswap32(a00.hi);
                if (h1 < target[1]) {
                    less = true;
                } else if (h1 == target[1]) {
                    uint32_t h2 = bswap32(a10.lo);
                    if (h2 < target[2]) {
                        less = true;
                    } else if (h2 == target[2]) {
                        uint32_t h3 = bswap32(a10.hi);
                        if (h3 < target[3]) {
                            less = true;
                        } else if (h3 == target[3]) {
                            uint32_t h4 = bswap32(a20.lo);
                            if (h4 < target[4]) {
                                less = true;
                            } else if (h4 == target[4]) {
                                uint32_t h5 = bswap32(a20.hi);
                                if (h5 < target[5]) {
                                    less = true;
                                } else if (h5 == target[5]) {
                                    uint32_t h6 = bswap32(a30.lo);
                                    if (h6 < target[6]) {
                                        less = true;
                                    } else if (h6 == target[6]) {
                                        uint32_t h7 = bswap32(a30.hi);
                                        less = h7 < target[7];
                                    }
                                }
                            }
                        }
                    }
                }
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

static std::string uint128_decimal(unsigned __int128 value) {
    if (value == 0) return "0";
    char buf[40]{};
    int pos = 0;
    while (value > 0) {
        buf[pos++] = (char)('0' + (uint32_t)(value % 10));
        value /= 10;
    }
    std::string out;
    out.reserve((size_t)pos);
    while (pos > 0) out.push_back(buf[--pos]);
    return out;
}

static uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::strtoull(v, nullptr, 0);
}

static double parse_bench_seconds(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        const char* bench_prefix = "--bench-seconds=";
        size_t bench_len = std::strlen(bench_prefix);
        if (std::strncmp(argv[i], bench_prefix, bench_len) == 0) {
            return std::strtod(argv[i] + bench_len, nullptr);
        }
        const char* timeout_prefix = "--timeout-seconds=";
        size_t timeout_len = std::strlen(timeout_prefix);
        if (std::strncmp(argv[i], timeout_prefix, timeout_len) == 0) {
            return std::strtod(argv[i] + timeout_len, nullptr);
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
        std::fprintf(stderr, "usage: cuda-miner <challenge-hex32> <difficulty-hex32> [--selftest] [--bench-seconds=N] [--timeout-seconds=N]\n");
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
    uint32_t nonce_stride = (uint32_t)env_u64("CUDA_NONCE_STRIDE", 1);
    if (block_threads == 0) block_threads = 256;
    if (nonces_per_thread == 0) nonces_per_thread = 1;
    if (nonce_stride == 0) nonce_stride = 1;
    uint32_t batch_log2 = (uint32_t)env_u64("CUDA_BATCH_LOG2", prop.major >= 9 ? 29 : 28);
    if (batch_log2 < 12) batch_log2 = 12;
    if (batch_log2 > 32) batch_log2 = 32;
    uint64_t batch = 1ULL << batch_log2;
    uint64_t threads_per_dispatch = batch / nonces_per_thread;
    if (threads_per_dispatch < block_threads) threads_per_dispatch = block_threads;
    threads_per_dispatch -= threads_per_dispatch % block_threads;
    batch = threads_per_dispatch * nonces_per_thread;
    uint32_t grid_blocks = (uint32_t)(threads_per_dispatch / block_threads);
    std::fprintf(stderr, "blocks=%u threads/block=%u batch=%lluM nonces/thread=%u stride=%u\n",
                 grid_blocks, block_threads, (unsigned long long)(batch / 1000000ULL),
                 nonces_per_thread, nonce_stride);

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
    uint64_t nonce_lo = selftest ? 42ULL : env_u64("CUDA_NONCE_LO", rng());
    uint64_t nonce_hi = selftest ? 0ULL : env_u64("CUDA_NONCE_HI", 0);

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
            mine_kernel<1, 1><<<1, 1>>>(d_challenge, d_target, d_nonce_base, d_found, d_result, 1, 1);
        } else if (nonces_per_thread == 1 && nonce_stride == 1) {
            mine_kernel<1, 1><<<grid_blocks, block_threads>>>(d_challenge, d_target, d_nonce_base, d_found, d_result, 1, 1);
        } else if (nonces_per_thread == 1) {
            mine_kernel<1, 0><<<grid_blocks, block_threads>>>(d_challenge, d_target, d_nonce_base, d_found, d_result, 1, nonce_stride);
        } else if (nonce_stride == 1) {
            mine_kernel<0, 1><<<grid_blocks, block_threads>>>(d_challenge, d_target, d_nonce_base, d_found, d_result, nonces_per_thread, 1);
        } else {
            mine_kernel<0, 0><<<grid_blocks, block_threads>>>(d_challenge, d_target, d_nonce_base, d_found, d_result, nonces_per_thread, nonce_stride);
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
                std::cout << uint128_decimal(nonce) << "\n";
            }
            return 0;
        }

        unsigned __int128 next = ((unsigned __int128)nonce_hi << 64) | nonce_lo;
        next += (unsigned __int128)batch * (unsigned __int128)nonce_stride;
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
