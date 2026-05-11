# CUDA HASH Miner

CUDA nonce finder for the HASH `mine(uint256 nonce)` proof:

```text
keccak256(abi.encode(challenge, nonce)) < difficulty
```

The Makefile builds one binary with native SASS for RTX 2080 Ti/Turing
(`sm_75`), A100 (`sm_80`), and H100 (`sm_90`), plus PTX for forward
compatibility.

## Build

```bash
cd cuda-miner
make
make 2080ti
make a100
make h100
```

Useful overrides:

```bash
make CUDA_ARCH_FLAGS="-gencode arch=compute_75,code=sm_75"   # RTX 2080 Ti only
make CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80"   # A100 only
make CUDA_ARCH_FLAGS="-gencode arch=compute_90,code=sm_90"   # H100 only
```

`make` writes `target/release/cuda-miner` with RTX 2080 Ti, A100, and H100 code.
`make 2080ti` writes `target/release/cuda-miner-2080ti`.
`make a100` writes `target/release/cuda-miner-a100`.
`make h100` writes `target/release/cuda-miner-h100`.

## Run

```bash
./target/release/cuda-miner <challenge-hex32> <difficulty-hex32>
./target/release/cuda-miner <challenge-hex32> <difficulty-hex32> --selftest
./target/release/cuda-miner <challenge-hex32> 0x00...00 --bench-seconds=10
```

## Tuning

- `CUDA_DEVICE`: GPU index, default `0`
- `CUDA_NONCE_LO`: low 64-bit starting nonce, default random
- `CUDA_NONCE_HI`: high 64-bit starting nonce, default `0`
- `CUDA_NONCE_STRIDE`: distance between searched nonces, default `1`
- `CUDA_BLOCK_THREADS`: threads per block, default `256`
- `CUDA_BATCH_LOG2`: hashes per dispatch as `2^N`, default `28`
- `CUDA_NONCES_PER_THREAD`: sequential nonces per CUDA thread, default `1`

Starting points:

```bash
# A100
CUDA_BLOCK_THREADS=256 CUDA_BATCH_LOG2=28 CUDA_NONCES_PER_THREAD=1 ./target/release/cuda-miner ...

# H100
CUDA_BLOCK_THREADS=256 CUDA_BATCH_LOG2=29 CUDA_NONCES_PER_THREAD=1 ./target/release/cuda-miner ...
```

## Multi-GPU

The top-level `mine.js` can launch one CUDA miner per GPU and partition nonce
searches by stride so workers do not overlap:

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 node mine.js --flashbots
```

For `N` devices, GPU `i` searches `base + i + k * N`.
