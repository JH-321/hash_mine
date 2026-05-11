#include <metal_stdlib>
using namespace metal;

struct U64 {
    uint lo;
    uint hi;
};

constant uint RC_LO[24] = {
    0x00000001u, 0x00008082u, 0x0000808au, 0x80008000u,
    0x0000808bu, 0x80000001u, 0x80008081u, 0x00008009u,
    0x0000008au, 0x00000088u, 0x80008009u, 0x8000000au,
    0x8000808bu, 0x0000008bu, 0x00008089u, 0x00008003u,
    0x00008002u, 0x00000080u, 0x0000800au, 0x8000000au,
    0x80008081u, 0x00008080u, 0x80000001u, 0x80008008u
};

constant uint RC_HI[24] = {
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x80000000u, 0x80000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u,
    0x80000000u, 0x80000000u, 0x00000000u, 0x80000000u
};

constant uint ROT[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

static inline U64 make_u64(uint lo, uint hi) {
    U64 v;
    v.lo = lo;
    v.hi = hi;
    return v;
}

static inline U64 xor64(U64 a, U64 b) {
    return make_u64(a.lo ^ b.lo, a.hi ^ b.hi);
}

static inline U64 and64(U64 a, U64 b) {
    return make_u64(a.lo & b.lo, a.hi & b.hi);
}

static inline U64 not64(U64 a) {
    return make_u64(~a.lo, ~a.hi);
}

static inline U64 rotl64(U64 x, uint n) {
    if (n == 0) {
        return x;
    } else if (n < 32) {
        return make_u64((x.lo << n) | (x.hi >> (32 - n)),
                        (x.hi << n) | (x.lo >> (32 - n)));
    } else if (n == 32) {
        return make_u64(x.hi, x.lo);
    } else {
        uint r = n - 32;
        return make_u64((x.hi << r) | (x.lo >> (32 - r)),
                        (x.lo << r) | (x.hi >> (32 - r)));
    }
}

static inline uint bswap32(uint x) {
    return ((x & 0xff000000u) >> 24) |
           ((x & 0x00ff0000u) >> 8) |
           ((x & 0x0000ff00u) << 8) |
           ((x & 0x000000ffu) << 24);
}

static void keccakf(thread U64* s) {
    for (uint round = 0; round < 24; round++) {
        U64 C[5];
#pragma unroll
        for (uint x = 0; x < 5; x++) {
            C[x] = xor64(xor64(xor64(xor64(s[x], s[x+5]), s[x+10]), s[x+15]), s[x+20]);
        }

#pragma unroll
        for (uint x = 0; x < 5; x++) {
            U64 D = xor64(C[(x+4) % 5], rotl64(C[(x+1) % 5], 1));
#pragma unroll
            for (uint y = 0; y < 5; y++) {
                s[x + 5*y] = xor64(s[x + 5*y], D);
            }
        }

        U64 B[25];
#pragma unroll
        for (uint x = 0; x < 5; x++) {
#pragma unroll
            for (uint y = 0; y < 5; y++) {
                B[y + 5*((2*x + 3*y) % 5)] = rotl64(s[x + 5*y], ROT[x + 5*y]);
            }
        }

#pragma unroll
        for (uint y = 0; y < 5; y++) {
#pragma unroll
            for (uint x = 0; x < 5; x++) {
                U64 a = B[x + 5*y];
                U64 b = B[((x+1) % 5) + 5*y];
                U64 c = B[((x+2) % 5) + 5*y];
                s[x + 5*y] = xor64(a, and64(not64(b), c));
            }
        }

        s[0].lo ^= RC_LO[round];
        s[0].hi ^= RC_HI[round];
    }
}

// One thread per nonce candidate.
//   nonce = nonce_base (uint128) + tid
//   buf  = challenge (32B) || nonce_as_uint256_be (32B)        // 64 bytes total
//   hash = keccak256(buf)   (Ethereum padding: 0x01 ... 0x80, rate=136)
//   if (uint256(hash) < target) atomically write nonce.
kernel void mine_kernel(
    constant uint*        challenge   [[buffer(0)]],  // 8 little-endian u32 state words
    constant uint*        target      [[buffer(1)]],  // 8 big-endian compare words
    constant uint*        nonce_base  [[buffer(2)]],  // [lo64.lo32, lo64.hi32, hi64.lo32, hi64.hi32]
    device   atomic_uint* found       [[buffer(3)]],
    device   uint*        result_lohi [[buffer(4)]],
    constant uint*        nonces_per_thread [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint npt = nonces_per_thread[0];
    uint tid_offset = tid * npt;
    uint base_lo = nonce_base[0] + tid_offset;
    uint base_carry0 = base_lo < nonce_base[0] ? 1u : 0u;
    uint base_hi = nonce_base[1] + base_carry0;
    uint base_carry1 = base_hi < nonce_base[1] ? 1u : 0u;
    uint base_nhi_lo = nonce_base[2] + base_carry1;
    uint base_carry2 = base_nhi_lo < nonce_base[2] ? 1u : 0u;
    uint base_nhi_hi = nonce_base[3] + base_carry2;

#pragma unroll
    for (uint iter = 0; iter < 16; iter++) {
        if (iter >= npt || atomic_load_explicit(found, memory_order_relaxed) != 0u) {
            return;
        }

        uint nlo_lo = base_lo + iter;
        uint carry0 = nlo_lo < base_lo ? 1u : 0u;
        uint nlo_hi = base_hi + carry0;
        uint carry1 = nlo_hi < base_hi ? 1u : 0u;
        uint nhi_lo = base_nhi_lo + carry1;
        uint carry2 = nhi_lo < base_nhi_lo ? 1u : 0u;
        uint nhi_hi = base_nhi_hi + carry2;

        U64 s[25];
#pragma unroll
        for (uint i = 0; i < 25; i++) s[i] = make_u64(0, 0);

        // challenge bytes 0..31 are pre-packed as little-endian Keccak state words.
        s[0] = make_u64(challenge[0], challenge[1]);
        s[1] = make_u64(challenge[2], challenge[3]);
        s[2] = make_u64(challenge[4], challenge[5]);
        s[3] = make_u64(challenge[6], challenge[7]);

        // nonce as 32 bytes BE -> bytes 32..63 of input.
        // Nonce uses only low 128 bits, so high 16 bytes (bytes 32..47) are zero.
        // In Keccak's little-endian lanes:
        //   s[6] bytes = nhi64.to_be_bytes()
        //   s[7] bytes = nlo64.to_be_bytes()
        s[6] = make_u64(bswap32(nhi_hi), bswap32(nhi_lo));
        s[7] = make_u64(bswap32(nlo_hi), bswap32(nlo_lo));

        // Keccak padding for 64-byte input, rate=1088 bits (136 bytes)
        s[8].lo  ^= 0x00000001u;          // 0x01 at byte 64
        s[16].hi ^= 0x80000000u;          // 0x80 at byte 135

        keccakf(s);

        // uint256(hash) < uint256(target). Keccak output bytes are state lanes read
        // little-endian, so the big-endian 32-bit word order is:
        //   bswap32(s[i].lo), bswap32(s[i].hi) for each output lane.
        bool decided = false;
        bool less = false;
#pragma unroll
        for (uint i = 0; i < 4 && !decided; i++) {
            uint h_hi = bswap32(s[i].lo);
            uint h_lo = bswap32(s[i].hi);
            uint t_hi = target[i * 2];
            uint t_lo = target[i * 2 + 1];
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
            if (atomic_exchange_explicit(found, 1u, memory_order_relaxed) == 0u) {
                result_lohi[0] = nlo_lo;
                result_lohi[1] = nlo_hi;
                result_lohi[2] = nhi_lo;
                result_lohi[3] = nhi_hi;
            }
            return;
        }
    }
}
