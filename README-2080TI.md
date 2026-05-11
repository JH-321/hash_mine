# RTX 2080 Ti 8-GPU HASH Mining

This guide runs one CUDA miner per RTX 2080 Ti and partitions nonce search so
the GPUs do not overlap.

## 1. Update and Build

```bash
cd ~/hash_mine
git pull

# Build a fat binary that includes RTX 2080 Ti sm_75 support.
make -C cuda-miner
```

RTX 2080 Ti-only build:

```bash
make -C cuda-miner clean
make -C cuda-miner 2080ti
```

If you use the 2080 Ti-only binary with `mine.js`, point `GPU_BIN` at it:

```bash
export GPU_BIN="$PWD/cuda-miner/target/release/cuda-miner-2080ti"
```

## 2. Check GPUs

```bash
nvidia-smi
```

You should see devices `0` through `7`.

## 3. Selftest

```bash
CUDA_DEVICE=0 ./cuda-miner/target/release/cuda-miner \
  0x0001020304050607080900010203040506070809000102030405060708090001 \
  0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --selftest
```

For an 2080 Ti-only binary:

```bash
CUDA_DEVICE=0 ./cuda-miner/target/release/cuda-miner-2080ti \
  0x0001020304050607080900010203040506070809000102030405060708090001 \
  0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --selftest
```

## 4. Single-GPU Benchmark

```bash
CUDA_DEVICE=0 \
CUDA_BLOCK_THREADS=256 \
CUDA_BATCH_LOG2=28 \
CUDA_NONCES_PER_THREAD=1 \
./cuda-miner/target/release/cuda-miner \
  0x0001020304050607080900010203040506070809000102030405060708090001 \
  0x0000000000000000000000000000000000000000000000000000000000000000 \
  --bench-seconds=30
```

Repeat with `CUDA_DEVICE=1`, `2`, ... if you want per-card numbers.

## 5. Run 8 GPUs

```bash
CUDA_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_BLOCK_THREADS=256 \
CUDA_BATCH_LOG2=28 \
CUDA_NONCES_PER_THREAD=1 \
npm run mine:flashbots
```

Expected startup line:

```text
CUDA multi-GPU: devices=0,1,2,3,4,5,6,7 stride=8
```

Nonce partitioning:

```text
GPU0: base + 0 + k*8
GPU1: base + 1 + k*8
GPU2: base + 2 + k*8
...
GPU7: base + 7 + k*8
```

So the 8 GPUs search the same challenge/difficulty without overlapping nonce
ranges. The first GPU to find a valid nonce wins, the parent process stops the
other miners, and only `mine.js` submits the transaction.

## 6. Epoch Safety

Default settings:

```bash
EPOCH_TIMEOUT_SAFETY_BLOCKS=2
BLOCK_WATCH_SECONDS=3
```

The miner will not start a new GPU search when the current epoch has only the
guard blocks left. While mining, `mine.js` watches `miningState()` and stops the
GPU processes when the epoch changes or the guard window is reached.

## 7. Useful Environment

```bash
export PRIORITY_GWEI=1
export MAX_TIP_GWEI=4
export GAS_LIMIT=200000
```

If logs are too noisy with 8 GPUs, lower reporting by running one card first for
benchmarks, then use the full `CUDA_DEVICES=0,1,2,3,4,5,6,7` command for live
mining.

