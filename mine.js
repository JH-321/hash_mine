import 'dotenv/config';
import os from 'node:os';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isMainThread, parentPort, workerData, Worker } from 'node:worker_threads';
import { ethers } from 'ethers';

const CONTRACT = '0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc';
const ABI = [
  'function mine(uint256 nonce) external',
  'function getChallenge(address miner) external view returns (bytes32)',
  'function currentDifficulty() external view returns (uint256)',
  'function mintsInBlock(uint256) external view returns (uint256)',
  'function usedProofs(bytes32) external view returns (bool)',
  'function totalMints() external view returns (uint256)',
  'function totalMiningMinted() external view returns (uint256)',
  'function MINING_SUPPLY() external view returns (uint256)',
  'function MAX_MINTS_PER_BLOCK() external view returns (uint256)',
  'function EPOCH_BLOCKS() external view returns (uint256)',
  'function genesisComplete() external view returns (bool)',
  'function miningState() external view returns (uint256 era,uint256 reward,uint256 difficulty,uint256 minted,uint256 remaining,uint256 epoch,uint256 epochBlocksLeft)',
];

// ─── Worker: brute-force nonces ────────────────────────────────────────────
function runWorker() {
  const { challenge, difficulty, startNonce, stride, workerId } = workerData;
  const diff = BigInt(difficulty);
  const challengeBytes = ethers.getBytes(challenge);
  const buf = new Uint8Array(64);
  buf.set(challengeBytes, 0);

  let nonce = BigInt(startNonce);
  const step = BigInt(stride);
  let hashes = 0n;
  let lastReport = Date.now();

  while (true) {
    // abi.encode(bytes32, uint256) = 32-byte challenge || 32-byte big-endian nonce
    let n = nonce;
    for (let i = 63; i >= 32; i--) {
      buf[i] = Number(n & 0xffn);
      n >>= 8n;
    }
    const result = BigInt(ethers.keccak256(buf));
    if (result < diff) {
      parentPort.postMessage({ type: 'found', nonce: nonce.toString(), workerId });
      return;
    }
    nonce += step;
    hashes++;
    if ((hashes & 0xffffn) === 0n) {
      const now = Date.now();
      if (now - lastReport > 5000) {
        parentPort.postMessage({ type: 'hashrate', hps: Number(hashes) / ((now - lastReport) / 1000), workerId });
        hashes = 0n;
        lastReport = now;
      }
    }
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────
const FLAGS = new Set(process.argv.slice(2));
const USE_FLASHBOTS = FLAGS.has('--flashbots');
const STATE_ONLY    = FLAGS.has('--state-only');
const ONCE          = FLAGS.has('--once');

class MiningTimedOut extends Error {
  constructor(seconds) {
    super(`mining timed out after ${seconds}s`);
    this.name = 'MiningTimedOut';
    this.seconds = seconds;
  }
}

class MiningWindowClosed extends Error {
  constructor(message) {
    super(message);
    this.name = 'MiningWindowClosed';
  }
}

function requireEnv(name) {
  const v = process.env[name];
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

async function main() {
  const PRIVATE_KEY = requireEnv('PRIVATE_KEY');
  const READ_RPC   = requireEnv('RPC_URL');
  const SEND_RPC   = USE_FLASHBOTS
    ? (process.env.FLASHBOTS_RPC || 'https://rpc.flashbots.net/fast')
    : (process.env.SEND_RPC_URL  || READ_RPC);
  const PRIORITY_GWEI = process.env.PRIORITY_GWEI || '3';
  const MAX_TIP_GWEI  = process.env.MAX_TIP_GWEI  || '50';
  const GAS_LIMIT     = BigInt(process.env.GAS_LIMIT || '200000');

  const readProvider = new ethers.JsonRpcProvider(READ_RPC);
  const sendProvider = new ethers.JsonRpcProvider(SEND_RPC);
  const wallet  = new ethers.Wallet(PRIVATE_KEY, sendProvider);
  const readC   = new ethers.Contract(CONTRACT, ABI, readProvider);

  console.log('━━━ Hash Miner ━━━');
  console.log('Miner :', wallet.address);
  console.log('RPC   :', READ_RPC);
  console.log('Send  :', USE_FLASHBOTS ? 'Flashbots Protect (private)' : SEND_RPC);
  console.log('Tip   :', `${PRIORITY_GWEI} gwei (cap ${MAX_TIP_GWEI} gwei)`);

  // Sanity probe
  const [bal, code, complete, state] = await Promise.all([
    readProvider.getBalance(wallet.address),
    readProvider.getCode(CONTRACT),
    readC.genesisComplete(),
    readC.miningState(),
  ]);
  if (code === '0x') throw new Error('Contract not found on this network');
  console.log('Balance:', ethers.formatEther(bal), 'ETH');
  console.log(`Difficulty: ${state.difficulty.toString(16).padStart(64, '0').slice(0, 16)}…`);
  console.log(`Reward    : ${ethers.formatEther(state.reward)} HASH (era ${state.era})`);
  console.log(`Remaining : ${ethers.formatEther(state.remaining)} / 18.9M HASH`);
  console.log(`Epoch     : ${state.epoch} (${state.epochBlocksLeft} blocks left)`);
  if (!complete) throw new Error('Genesis not complete — mining not yet open');
  if (STATE_ONLY) return;

  while (true) {
    const challenge  = await readC.getChallenge(wallet.address);
    const stateNow   = await readC.miningState();
    const difficulty = stateNow.difficulty;
    const blockNo    = await readProvider.getBlockNumber();
    const miningWindow = miningWindowFor(stateNow.epochBlocksLeft);
    console.log(`\n[block ${blockNo}] challenge=${challenge.slice(0,10)}… difficulty=2^${256 - difficulty.toString(2).length} headroom`);
    if (miningWindow.timeoutSeconds) {
      console.log(`→ mining window: ${miningWindow.timeoutSeconds}s (${miningWindow.blocksLeft} blocks left, guard ${miningWindow.safetyBlocks})`);
    }

    let nonce;
    try {
      nonce = await findNonce(challenge, difficulty, miningWindow, {
        readC,
        startEpoch: stateNow.epoch.toString(),
      });
    } catch (e) {
      if (e instanceof MiningTimedOut || e instanceof MiningWindowClosed) {
        console.log(`↻ ${e.message}; refreshing challenge`);
        if (ONCE) return;
        await new Promise(r => setTimeout(r, 1000));
        continue;
      }
      throw e;
    }

    console.log(`→ nonce found: ${nonce}`);

    // Build EIP-1559 tx with aggressive priority fee
    const block = await readProvider.getBlock('latest');
    const baseFee = block.baseFeePerGas || 0n;
    const tip   = ethers.parseUnits(PRIORITY_GWEI, 'gwei');
    const tipCap = ethers.parseUnits(MAX_TIP_GWEI, 'gwei');
    const maxPriorityFeePerGas = tip > tipCap ? tipCap : tip;
    const maxFeePerGas = baseFee * 2n + maxPriorityFeePerGas;

    const iface = new ethers.Interface(ABI);
    const data  = iface.encodeFunctionData('mine', [nonce]);

    const tx = {
      to: CONTRACT,
      data,
      value: 0n,
      gasLimit: GAS_LIMIT,
      maxFeePerGas,
      maxPriorityFeePerGas,
      type: 2,
      chainId: 1,
      nonce: await sendProvider.getTransactionCount(wallet.address, 'pending'),
    };

    const signed = await wallet.signTransaction(tx);
    console.log(`→ submitting (tip=${ethers.formatUnits(maxPriorityFeePerGas, 'gwei')} gwei, base=${ethers.formatUnits(baseFee, 'gwei')} gwei)`);

    try {
      const sent = await sendProvider.broadcastTransaction(signed);
      console.log(`→ tx: https://etherscan.io/tx/${sent.hash}`);
      const receipt = await sent.wait(1);
      if (receipt.status === 1) {
        console.log(`✓ MINED in block ${receipt.blockNumber} (gas used ${receipt.gasUsed})`);
      } else {
        console.log(`✗ tx reverted in block ${receipt.blockNumber} (likely BlockCapReached or ProofAlreadyUsed)`);
      }
    } catch (e) {
      console.error('✗ submission failed:', e.shortMessage || e.message);
    }

    if (ONCE) return;
    await new Promise(r => setTimeout(r, 1000));
  }
}

const ROOT_DIR = path.dirname(fileURLToPath(import.meta.url));
const CUDA_BIN = path.resolve(ROOT_DIR, 'cuda-miner/target/release/cuda-miner');
const METAL_BIN = path.resolve(ROOT_DIR, 'gpu-miner/target/release/gpu-miner');

function selectGpuBin() {
  if (process.env.GPU_BIN) return process.env.GPU_BIN;
  if (existsSync(CUDA_BIN)) return CUDA_BIN;
  return METAL_BIN;
}

const GPU_BIN = selectGpuBin();

function miningWindowFor(epochBlocksLeft) {
  const avgBlockSeconds = Number(process.env.BLOCK_SECONDS || '12');
  const safetyBlocks = Number(process.env.EPOCH_TIMEOUT_SAFETY_BLOCKS || '2');
  const blocksLeft = Number(epochBlocksLeft);
  const safeBlocks = Math.max(0, blocksLeft - safetyBlocks);
  const timeoutSeconds = process.env.MINING_TIMEOUT_SECONDS
    ? Math.max(1, Number(process.env.MINING_TIMEOUT_SECONDS))
    : Math.floor(safeBlocks * avgBlockSeconds);
  return {
    blocksLeft,
    safetyBlocks,
    timeoutSeconds: timeoutSeconds > 0 ? timeoutSeconds : 0,
  };
}

async function findNonce(challenge, difficulty, miningWindow, monitor) {
  if (existsSync(GPU_BIN)) return findNonceGPU(challenge, difficulty, miningWindow, monitor);
  console.log('GPU miner not found, falling back to CPU workers (slow).');
  console.log('Build CUDA with: cd cuda-miner && make');
  console.log('Build Metal with: cd gpu-miner && cargo build --release');
  return findNonceCPU(challenge, difficulty, miningWindow.timeoutSeconds);
}

function findNonceGPU(challenge, difficulty, miningWindow, monitor) {
  const diffHex = '0x' + difficulty.toString(16).padStart(64, '0');
  const timeoutSeconds = miningWindow.timeoutSeconds;
  return new Promise((resolve, reject) => {
    const args = [challenge, diffHex];
    if (timeoutSeconds) args.push(`--timeout-seconds=${timeoutSeconds}`);
    const proc = spawn(GPU_BIN, args, { stdio: ['ignore', 'pipe', 'inherit'] });
    let out = '';
    let done = false;
    let timer = null;
    let epochPoller = null;

    const cleanup = () => {
      if (timer) clearTimeout(timer);
      if (epochPoller) clearInterval(epochPoller);
    };
    const settle = (fn, value) => {
      if (done) return;
      done = true;
      cleanup();
      fn(value);
    };
    const stopMining = (err) => {
      if (proc.exitCode === null) proc.kill('SIGTERM');
      settle(reject, err);
    };

    if (timeoutSeconds) {
      timer = setTimeout(() => {
        stopMining(new MiningTimedOut(timeoutSeconds));
      }, timeoutSeconds * 1000);
    }

    const pollSeconds = Math.max(1, Number(process.env.BLOCK_WATCH_SECONDS || '3'));
    if (monitor?.readC && monitor?.startEpoch) {
      epochPoller = setInterval(async () => {
        try {
          const state = await monitor.readC.miningState();
          const epochChanged = state.epoch.toString() !== monitor.startEpoch;
          const tooCloseToBoundary = Number(state.epochBlocksLeft) <= miningWindow.safetyBlocks;
          if (epochChanged || tooCloseToBoundary) {
            stopMining(new MiningWindowClosed(
              `mining window closed at epoch ${state.epoch} (${state.epochBlocksLeft} blocks left)`
            ));
          }
        } catch {
          // Keep the local miner running; the wall-clock timeout is still a fallback.
        }
      }, pollSeconds * 1000);
    }

    proc.stdout.on('data', (chunk) => { out += chunk.toString(); });
    proc.on('error', (e) => settle(reject, e));
    proc.on('exit', (code) => {
      if (done) return;
      const nonce = out.trim().split('\n').pop();
      if (code !== 0) return settle(reject, new Error(`gpu-miner exited ${code}`));
      if ((!nonce || !/^\d+$/.test(nonce)) && timeoutSeconds) return settle(reject, new MiningTimedOut(timeoutSeconds));
      if (!nonce || !/^\d+$/.test(nonce)) return settle(reject, new Error(`bad nonce output: ${JSON.stringify(out)}`));
      settle(resolve, nonce);
    });
  });
}

async function findNonceCPU(challenge, difficulty, timeoutSeconds) {
  const numWorkers = Math.max(1, Number(process.env.WORKERS || os.cpus().length - 1));
  console.log(`mining with ${numWorkers} worker thread(s)…`);

  const self = fileURLToPath(import.meta.url);
  const baseNonce = (BigInt(Date.now()) << 32n) ^ (BigInt(Math.floor(Math.random() * 2 ** 32)));

  return new Promise((resolve, reject) => {
    const workers = [];
    let done = false;
    const stop = () => { done = true; workers.forEach(w => w.terminate().catch(() => {})); };
    let timer = null;
    if (timeoutSeconds) {
      timer = setTimeout(() => {
        if (!done) {
          stop();
          reject(new MiningTimedOut(timeoutSeconds));
        }
      }, timeoutSeconds * 1000);
    }

    for (let i = 0; i < numWorkers; i++) {
      const w = new Worker(self, {
        workerData: {
          challenge,
          difficulty: difficulty.toString(),
          startNonce: (baseNonce + BigInt(i)).toString(),
          stride: numWorkers.toString(),
          workerId: i,
        },
      });
      w.on('message', (msg) => {
        if (msg.type === 'found' && !done) {
          if (timer) clearTimeout(timer);
          stop();
          resolve(msg.nonce);
        }
        if (msg.type === 'hashrate') {
          process.stdout.write(`\r  worker ${msg.workerId}: ${(msg.hps/1000).toFixed(1)} kH/s    `);
        }
      });
      w.on('error', (e) => {
        if (!done) {
          if (timer) clearTimeout(timer);
          stop();
          reject(e);
        }
      });
      workers.push(w);
    }
  });
}

if (isMainThread) {
  main().catch((e) => { console.error('fatal:', e.message); process.exit(1); });
} else {
  runWorker();
}
