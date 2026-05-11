# CUDA HASH Miner

CUDA nonce finder for the HASH `mine(uint256 nonce)` proof:

```text
keccak256(abi.encode(challenge, nonce)) < difficulty
```

The Makefile builds one binary with native SASS for both A100 (`sm_80`) and H100
(`sm_90`), plus PTX for forward compatibility.

## Build

```bash
cd cuda-miner
make
make a100
make h100
```

Useful overrides:

```bash
make CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80"   # A100 only
make CUDA_ARCH_FLAGS="-gencode arch=compute_90,code=sm_90"   # H100 only
```

`make` writes `target/release/cuda-miner` with both A100 and H100 code.
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
