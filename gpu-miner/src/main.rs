use metal::*;
use std::env;
use std::ffi::c_void;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tiny_keccak::{Hasher, Keccak};

const SHADER_SRC: &str = include_str!("keccak.metal");
const DEFAULT_BATCH_LOG2: u32 = 22;

fn parse_hex32(s: &str) -> [u8; 32] {
    let s = s.trim_start_matches("0x");
    let padded = format!("{:0>64}", s);
    let bytes = hex_decode(&padded);
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    out
}

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("bad hex"))
        .collect()
}

fn keccak_cpu(challenge: &[u8; 32], nonce: u128) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(challenge);
    // nonce as 32 bytes big-endian (high 16 bytes zero since nonce ≤ u128::MAX)
    buf[32 + 16..].copy_from_slice(&nonce.to_be_bytes());
    let mut h = Keccak::v256();
    h.update(&buf);
    let mut out = [0u8; 32];
    h.finalize(&mut out);
    out
}

fn hash_be_lt_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    for i in 0..32 {
        match hash[i].cmp(&target[i]) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => continue,
        }
    }
    false
}

fn le_u32_words_from_bytes32(bytes: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        let mut chunk = [0u8; 4];
        chunk.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
        out[i] = u32::from_le_bytes(chunk);
    }
    out
}

fn be_u32_words_from_bytes32(bytes: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        let mut chunk = [0u8; 4];
        chunk.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
        out[i] = u32::from_be_bytes(chunk);
    }
    out
}

fn env_u64(name: &str) -> Option<u64> {
    env::var(name).ok().and_then(|v| v.parse::<u64>().ok())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gpu-miner <challenge-hex32> <difficulty-hex32> [--selftest]");
        std::process::exit(2);
    }
    let challenge = parse_hex32(&args[1]);
    let difficulty = parse_hex32(&args[2]);
    let selftest = args.iter().any(|a| a == "--selftest");
    let bench_seconds = args
        .iter()
        .find_map(|a| {
            a.strip_prefix("--bench-seconds=")
                .or_else(|| a.strip_prefix("--timeout-seconds="))
        })
        .and_then(|v| v.parse::<f64>().ok());

    let device = Device::system_default().expect("no Metal device");
    eprintln!("device: {}", device.name());
    eprintln!("challenge:  0x{}", hex_encode(&challenge));
    eprintln!("difficulty: 0x{}", hex_encode(&difficulty));

    let queue = device.new_command_queue();
    let library = device
        .new_library_with_source(SHADER_SRC, &CompileOptions::new())
        .expect("compile shader");
    let kernel = library
        .get_function("mine_kernel", None)
        .expect("kernel fn");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("pipeline");

    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let requested_tg = env_u64("MINER_THREADS").unwrap_or(64);
    let threadgroup_size = std::cmp::min(max_tg, requested_tg).max(1);
    let batch_log2 = env_u64("MINER_BATCH_LOG2")
        .unwrap_or(DEFAULT_BATCH_LOG2 as u64)
        .clamp(10, 30) as u32;
    let mut batch = 1u64 << batch_log2;
    batch -= batch % threadgroup_size;
    if batch == 0 {
        batch = threadgroup_size;
    }
    let nonces_per_thread = env_u64("MINER_NONCES_PER_THREAD")
        .unwrap_or(1)
        .clamp(1, 16) as u32;
    let threads_per_dispatch = (batch / nonces_per_thread as u64).max(threadgroup_size);
    let threads_per_dispatch = threads_per_dispatch - (threads_per_dispatch % threadgroup_size);
    let batch = threads_per_dispatch * nonces_per_thread as u64;
    eprintln!(
        "max_total_threads_per_threadgroup={} → using {}, batch={}M, nonces/thread={}",
        max_tg,
        threadgroup_size,
        batch / 1_000_000,
        nonces_per_thread
    );

    let challenge_words = le_u32_words_from_bytes32(&challenge);
    let target_words = be_u32_words_from_bytes32(&difficulty);

    let challenge_buf = device.new_buffer_with_data(
        challenge_words.as_ptr() as *const c_void,
        std::mem::size_of_val(&challenge_words) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let target_buf = device.new_buffer_with_data(
        target_words.as_ptr() as *const c_void,
        std::mem::size_of_val(&target_words) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let nonce_base_buf = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
    let found_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let result_buf = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
    let nonces_per_thread_buf = device.new_buffer_with_data(
        (&nonces_per_thread as *const u32).cast::<c_void>(),
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Random initial nonce base (low 64 bits varies; high stays 0 then increments)
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
    let mut nonce_lo: u64 = now.wrapping_mul(0x9e3779b97f4a7c15);
    let mut nonce_hi: u64 = 0;

    // Self-test: dispatch a tiny batch with target = u256::MAX so thread 0 wins,
    // then verify GPU's hash matches CPU keccak.
    if selftest {
        let max_target = [0xffu8; 32];
        let max_target_words = be_u32_words_from_bytes32(&max_target);
        let test_target_buf = device.new_buffer_with_data(
            max_target_words.as_ptr() as *const c_void,
            std::mem::size_of_val(&max_target_words) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            *(found_buf.contents() as *mut u32) = 0;
            let b = nonce_base_buf.contents() as *mut u64;
            *b = 42;
            *b.add(1) = 0;
        }
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&challenge_buf), 0);
        enc.set_buffer(1, Some(&test_target_buf), 0);
        enc.set_buffer(2, Some(&nonce_base_buf), 0);
        enc.set_buffer(3, Some(&found_buf), 0);
        enc.set_buffer(4, Some(&result_buf), 0);
        enc.set_buffer(5, Some(&nonces_per_thread_buf), 0);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let (nlo, _nhi) = unsafe {
            let p = result_buf.contents() as *const u64;
            (*p, *p.add(1))
        };
        let nonce = nlo as u128;
        let gpu_hash_would_be = keccak_cpu(&challenge, nonce);
        eprintln!("selftest: gpu produced nonce={}, CPU hash={}", nonce, hex_encode(&gpu_hash_would_be));
        if nonce != 42 {
            eprintln!("FAIL: expected nonce=42 (single-thread), got {}", nonce);
            std::process::exit(1);
        }
        eprintln!("selftest OK ✓");
        return;
    }

    let mut total: u64 = 0;
    let start = Instant::now();
    let mut last_report = start;

    loop {
        unsafe {
            *(found_buf.contents() as *mut u32) = 0;
            let b = nonce_base_buf.contents() as *mut u64;
            *b = nonce_lo;
            *b.add(1) = nonce_hi;
        }

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&challenge_buf), 0);
        enc.set_buffer(1, Some(&target_buf), 0);
        enc.set_buffer(2, Some(&nonce_base_buf), 0);
        enc.set_buffer(3, Some(&found_buf), 0);
        enc.set_buffer(4, Some(&result_buf), 0);
        enc.set_buffer(5, Some(&nonces_per_thread_buf), 0);

        let groups = threads_per_dispatch / threadgroup_size;
        enc.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(threadgroup_size, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        total += batch;
        let found = unsafe { *(found_buf.contents() as *const u32) };
        if found == 1 {
            let (nlo, nhi) = unsafe {
                let p = result_buf.contents() as *const u64;
                (*p, *p.add(1))
            };
            let nonce = ((nhi as u128) << 64) | (nlo as u128);

            // Verify on CPU
            let h = keccak_cpu(&challenge, nonce);
            let ok = hash_be_lt_target(&h, &difficulty);
            let elapsed = start.elapsed().as_secs_f64();
            let rate = total as f64 / elapsed / 1e9;
            eprintln!(
                "FOUND in {:.2}s ({} MH, {:.2} GH/s) — hash=0x{} — verify={}",
                elapsed,
                total / 1_000_000,
                rate,
                hex_encode(&h),
                if ok { "OK" } else { "MISMATCH!" }
            );
            if !ok {
                eprintln!("Kernel produced bogus result — refusing to submit");
                std::process::exit(3);
            }
            println!("{}", nonce);
            return;
        }

        let (new_lo, carry) = nonce_lo.overflowing_add(batch);
        nonce_lo = new_lo;
        if carry {
            nonce_hi = nonce_hi.wrapping_add(1);
        }

        if last_report.elapsed().as_secs_f64() > 2.0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = total as f64 / elapsed / 1e9;
            eprintln!(
                "[{:.1}s] {} MH searched, {:.2} GH/s",
                elapsed,
                total / 1_000_000,
                rate
            );
            last_report = Instant::now();
        }

        if let Some(limit) = bench_seconds {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed >= limit {
                let rate = total as f64 / elapsed / 1e9;
                eprintln!(
                    "BENCH {:.2}s: {} MH searched, {:.3} GH/s",
                    elapsed,
                    total / 1_000_000,
                    rate
                );
                return;
            }
        }
    }
}

fn hex_encode(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for &x in b {
        s.push_str(&format!("{:02x}", x));
    }
    s
}
