# Hash Coin Miner — 작업 노트

레퍼런스 tx: <https://etherscan.io/tx/0x09c817ab64d9ff970bea3e13635f1796110f0ed1338513bae3b9ef7822480593>

> **현재 상태 스냅샷** (2026-05-11)
> - Miner: `0x8868d6C68Eda4131a25826927E51102bdA7abFB7`
> - Balance: 0.0058 ETH (~$13.3 @ $2300/ETH)
> - Difficulty: `0x00000000003fffff…` → **상위 42 비트 0 필요**
> - Reward: 100 HASH/mint (era 0)
> - Remaining mining supply: ~17.85M / 18.9M HASH
> - Epoch: 250697 (challenge 70블록 유효)

---

## 1. 컨트랙트 분석

**Hash (HASH)** — `0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc`
- 메인넷 ERC20 + Uniswap V4 self-hook + PoW miner. 한 컨트랙트가 토큰/훅/마이너 셋 다.
- 풀소스 sourcify에서 확보 후 분석.

### 주요 상수
| 상수 | 값 | 의미 |
|---|---|---|
| `MAX_MINTS_PER_BLOCK` | **10** | 한 블록에 선착순 10개 mine() 성공 |
| `EPOCH_BLOCKS` | 100 | challenge 갱신 주기 (~20분) |
| `TARGET_BLOCKS_PER_MINT` | 5 | 목표 채굴 간격 |
| `ADJUSTMENT_INTERVAL` | 2,016 | 난이도 재조정 주기 |
| `BASE_REWARD` | 100e18 | 초기 mint당 100 HASH |
| `MINING_SUPPLY` | 18,900,000 HASH | 채굴 가능 총량 |

### 마이닝 로직
```solidity
challenge = keccak256(abi.encode(chainid, contract, miner, block.number/100))
result    = keccak256(abi.encode(challenge, nonce))
require(uint256(result) < currentDifficulty);
```
- challenge는 **miner 주소에 묶임** → 다른 지갑이 nonce 훔쳐도 무효
- challenge는 100블록(~20분) 동안 고정 → 미리 비축 가능
- 같은 (miner, nonce, epoch) 조합은 `usedProofs`로 한 번만 사용 가능

### Revert 케이스
- `BlockCapReached` — 그 블록에 이미 10개 채워짐
- `InsufficientWork` — hash >= difficulty
- `ProofAlreadyUsed` — 같은 nonce 재사용 시도
- `SupplyExhausted` — 채굴 가능량 소진

---

## 2. 비용 계산 (ETH = $2300 기준)

`gasUsed ≈ 84,000` (레퍼런스 tx 실측) × `(baseFee + priorityFee)` gwei

### 현재 메인넷 (한산함, baseFee ≈ 0.117 gwei)
| 시나리오 | 실효 가스 | ETH | USD |
|---|---|---|---|
| .env 기본 (`PRIORITY_GWEI=5`) | 5.12 gwei | 0.00043 | **~$1.0** |
| 좀 더 공격적 (`tip=10`) | 10.12 gwei | 0.00085 | ~$2.0 |
| 캡까지 (`tip=50`) | 50.12 gwei | 0.0042 | ~$9.7 |

### 혼잡 시 (baseFee 3~20 gwei)
| baseFee | 5 tip | 20 tip | 50 tip |
|---|---|---|---|
| 3 gwei | $1.5 | $4.4 | $10 |
| 20 gwei | $4.8 | $7.7 | $13.5 |

### 손익 분기
- 보상: 100 HASH × $0.24 = **$24/mint**
- 현재 환경 순익: **$23 / mint**
- HASH 단가 손익선: ~$0.025 (50 gwei 캡 기준)

---

## 3. Top-of-Block 전략

컨트랙트는 한 블록에 처음 10개 `mine()`만 통과. 빌더는 tx를 effective priority fee 내림차순으로 정렬 → **본질은 가스 경매**.

| 전략 | 효과 | 비고 |
|---|---|---|
| 높은 `PRIORITY_GWEI` | 빌더 정렬에서 상위 | 가장 직접적 |
| Flashbots Protect (`--flashbots`) | revert tx 자동 차단 → 실패해도 **가스 0** | 잔고 적을 때 필수 |
| Flashbots bundle + coinbase tip | 진짜 rank-1 보장 | bundle 작업 추가 필요 |
| Nonce 비축 | epoch 100블록 내 미리 탐색 | 현재 구현 안 됨 |

### Flashbots Protect 주의점
- Inclusion 보장 X — 입찰가 너무 낮으면 어떤 빌더도 안 가져감 → ~25블록 후 expire
- Wallet nonce 점유됨 — Flashbots 처리 중엔 다음 tx 못 보냄
- 시뮬에선 통과했는데 실제 빌드 시 state 바뀌어 revert → 빌더가 알아서 제외

---

## 4. GPU 마이너 구현

### 스택
- **Rust** + **Metal** (Mac 전용, `metal` crate v0.27)
- 컴퓨트 셰이더에서 keccak256 (이더리움 패딩: 0x01...0x80, rate=136)
- `tiny-keccak`로 CPU side verification

### 구조
```
gpu-miner/
├── Cargo.toml
└── src/
    ├── main.rs        # 디바이스 셋업, 디스패치 루프, CPU 검증
    └── keccak.metal   # 컴퓨트 셰이더 (25 라운드 keccak-f[1600])
```

### 검증 통과
- `--selftest`: 단일 스레드 디스패치, nonce=42 반환 확인 ✓
- 16/24/32비트 난이도 → GPU hash가 CPU keccak과 비트 단위 일치 ✓
- 모든 successful nonce에 대해 자동 CPU 재검증 후 송신

### 측정된 성능
**Apple M4 = 0.11 GH/s** (110 MH/s)

기대보다 낮음. Apple GPU는 32-bit native라 uint64 keccak ops가 비쌈.
- 비교: RTX 4090 ~12 GH/s, ASIC 100+ GH/s

### 현재 난이도(`0x00000000003fffff…` ≈ 2^42)에서 ETA
| 메트릭 | 값 |
|---|---|
| 필요 평균 해시 수 | ~4.4 × 10^12 (2^42) |
| 평균 nonce 탐색 시간 | **~11시간** |
| 5블록(60s) 내 탐색 성공 확률 | 0.15% |
| 1 epoch(20분) 내 성공 확률 | 3% |
| 시간당 기대 nonce 발견 | ~0.09개 |
| 시간당 기대 수익 (경쟁 없을 시 상한) | ~$2.2 |

⚠️ **난이도가 2비트 더 깎였음** (이전 2^40 → 현재 2^42, **4배 더 어려움**). 컨트랙트가 `_adjustDifficulty()` 최대치(÷4)로 자동 조정했다는 건 **마이닝 활동이 매우 활발**하다는 뜻.

⚠️ **경쟁자가 ASIC/GPU 리그면 못 이김.** 짧은 시간 동안 잔여량이 ~80,000 HASH (800+ mints) 줄어든 것을 보면 네트워크 활용도 ≥50%. 많은 블록이 10-slot 풀로 차고 있을 가능성 큼. `mintsInBlock(latest)` 직접 모니터링 권장.

### 최적화 여지 (필요 시)
- 32-bit slicing 재구현 → 3~5배 가능
- 라운드 unroll + register tuning
- Multi-GPU 디스패치

---

## 5. 파일 구조

```
hash_coin/
├── .env                         # PRIVATE_KEY, RPC_URL 등 (secret)
├── .env.example                 # 템플릿
├── package.json
├── mine.js                      # Node 메인 엔트리
├── node_modules/
├── gpu-miner/
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── src/main.rs
│   ├── src/keccak.metal
│   └── target/release/gpu-miner # 컴파일된 바이너리 (mine.js가 자동 감지)
└── NOTES.md                     # ← 이 파일
```

### .env (현재 설정)
```
PRIVATE_KEY=0x...
RPC_URL=https://eth-mainnet.g.alchemy.com/v2/9TjYyZCbsHY5nfAWikZoH
PRIORITY_GWEI=5
MAX_TIP_GWEI=50
```
- `SEND_RPC_URL` 미설정 → 일반 모드에서 `RPC_URL`로 폴백
- `--flashbots` 플래그 사용 시 자동으로 `https://rpc.flashbots.net/fast` 사용
- `WORKERS=` 는 CPU fallback 전용 → GPU 바이너리 자동 감지되므로 비워둬도 됨

---

## 6. 사용법

### 빌드 (1회)
```bash
npm install
cd gpu-miner && cargo build --release && cd ..
```

### 상태 확인
```bash
npm run state
```
출력: 잔고, 난이도, 현재 epoch, 잔여 보상

### 채굴
```bash
npm run mine              # public mempool, 높은 priority fee로 경쟁
npm run mine:flashbots    # Flashbots Protect (실패 시 가스 0) — 추천
```

### 단발성
```bash
node mine.js --once --flashbots
```

---

## 7. 동작 흐름

```
1. mine.js 시작
2. challenge = getChallenge(myAddr) on-chain 조회
3. currentDifficulty 조회
4. ./gpu-miner/target/release/gpu-miner spawn
   ↓
5. GPU 컴퓨트 셰이더로 nonce 브루트포스 (~0.1 GH/s)
6. valid nonce 발견 → CPU keccak 재검증
7. nonce stdout 출력 후 종료
   ↓
8. mine.js가 EIP-1559 tx 빌드 (maxPriorityFee = PRIORITY_GWEI)
9. Flashbots Protect로 송신 (또는 public RPC)
10. wait_until_completed
    - status=1 → "✓ MINED" + reward 100 HASH 수령
    - status=0 → 가스만 차감 (Flashbots면 0)
11. 다음 epoch challenge로 루프
```

---

## 8. 현실 체크 / 권장 사항

- **잔고 0.0058 ETH ($13)으로 시작 가능**: 현재 가스 환경에서 5~10번 시도분
- **첫 mint 성공 = +$23 순익** → 잔고 회수 + 증식
- **경쟁 모니터링**: 송신 전에 `mintsInBlock(latest)` 체크. 자주 10이면 ASIC 경쟁권. 0~3이면 기회.
- **GPU 성능이 부족하면** 외부 GPU 박스(RTX 30xx 이상) 또는 ASIC 검토. M4로는 운빨 채굴 수준.
- **Flashbots 무조건 추천** — 실패 비용 0이라 다운사이드 제한.

---

## 9. 검증된 사실 (테스트 로그)

- ✅ 레퍼런스 tx 입력값 재구성 → CPU/GPU keccak 동일 결과
- ✅ M4 Metal 컴퓨트 셰이더 정상 동작
- ✅ 32비트 난이도(4.8B 해시) 43초 내 탐색 완료
- ✅ Flashbots Protect endpoint 정상 (`rpc.flashbots.net/fast`)
- ✅ EIP-1559 tx 빌드 + 서명 + 송신 경로 검증

---

## 부록 A: 컨트랙트 mine() 함수 (verbatim)

```solidity
function mine(uint256 nonce) external nonReentrant {
    if (!genesisComplete)                                  revert GenesisNotComplete();
    if (totalMiningMinted >= MINING_SUPPLY)                revert SupplyExhausted();
    if (mintsInBlock[block.number] >= MAX_MINTS_PER_BLOCK) revert BlockCapReached();

    bytes32 result = keccak256(abi.encode(_challenge(msg.sender), nonce));
    if (uint256(result) >= currentDifficulty) revert InsufficientWork();

    bytes32 key = keccak256(abi.encode(msg.sender, nonce, _epoch()));
    if (usedProofs[key]) revert ProofAlreadyUsed();
    usedProofs[key] = true;

    mintsInBlock[block.number]++;
    totalMints++;
    // ... 난이도 조정, 보상 계산, 토큰 전송
}
```

## 부록 B: Etherscan 링크
- 컨트랙트: <https://etherscan.io/address/0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc>
- 레퍼런스 tx: <https://etherscan.io/tx/0x09c817ab64d9ff970bea3e13635f1796110f0ed1338513bae3b9ef7822480593>
- 가스 트래커: <https://etherscan.io/gastracker>
- 프로젝트: <https://hash256.org/>
