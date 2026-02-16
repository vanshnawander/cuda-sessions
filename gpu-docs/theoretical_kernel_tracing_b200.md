# Theoretical Kernel Performance Tracing — NVFP4 Group GEMM on B200

## How to Think About GPU Kernel Performance From First Principles

This document teaches you how to manually trace a kernel's execution time
using nothing but hardware specs and arithmetic. No profiler, no PyTorch —
just raw physics of data movement and computation.

---

## 1. B200 Hardware Constants (at 1.5 GHz clock, challenge spec)

### 1.1 Clock and SM Count

| Parameter | Value | How Derived |
|-----------|-------|-------------|
| SM count | 148 | 74 per die × 2 dies |
| Clock frequency (challenge) | 1.5 GHz | Given in challenge SOL analysis |
| Clock period | 0.667 ns | 1 / 1.5 GHz |
| Tensor Cores per SM | 4 (approx) | 640 total / 148 SMs ≈ 4.3 |

### 1.2 Memory Hierarchy — Latency (measured via microbenchmarks)

| Memory Level | Latency (cycles) | Latency (ns @ 1.5GHz) | Bandwidth per SM | Total GPU BW |
|-------------|-------------------|------------------------|------------------|--------------|
| **Registers** | 1 cycle | 0.67 ns | ~19 TB/s per SM | — |
| **TMEM (Tensor Memory)** | 11-14 cycles | 7.3-9.3 ns | 16 TB/s read, 8 TB/s write per SM | — |
| **Shared Memory (SMEM)** | ~20-30 cycles | 13-20 ns | ~8 TB/s per SM | — |
| **L1 Cache** | ~30 cycles | 20 ns | ~4 TB/s per SM | — |
| **L2 Cache** | ~200 cycles | 133 ns | — | ~12 TB/s aggregate |
| **HBM3e (DRAM)** | ~420 cycles | 280 ns | — | **8 TB/s** aggregate |

> **Source**: "Microbenchmarking NVIDIA's Blackwell Architecture" (arXiv:2512.02189)
> TMEM achieves 420 cycles for cache-miss, 58% reduction vs Hopper's 1000 cycles.
> tcgen05.mma latency: 11.0-11.4 cycles, nearly constant across tile sizes.

### 1.3 Compute Throughput (measured, 96.3% of theoretical peak)

| Precision | Peak TFLOPS (theoretical) | Measured TFLOPS | Per-SM TFLOPS |
|-----------|--------------------------|-----------------|---------------|
| **FP4** | ~8,000 | 7,702.5 | 52.04 |
| **FP8** | ~4,000 | 3,851.4 | 26.02 |
| **FP16 (FP16 acc)** | ~2,000 | 1,929.2 | 13.03 |
| **FP16 (FP32 acc)** | ~1,000 | 964.6 | 6.52 |
| **INT8** | ~4,000 | 3,927.1 | 26.53 |
| **FP64** | ~90 | 44.8 | 0.30 |

> FP4 achieves 96.3% of peak. Tensor Cores are NOT the bottleneck —
> memory bandwidth and kernel launch overhead are.

### 1.4 Memory Bandwidth

| Path | Bandwidth | Notes |
|------|-----------|-------|
| **HBM3e → L2** | 8 TB/s | 8 stacks, 8192-bit interface |
| **L2 → SMEM (TMA)** | ~12 TB/s | TMA bypasses L1, bulk async |
| **SMEM → Tensor Core** | ~8 TB/s per SM | Via tcgen05.mma operand fetch |
| **TMEM → Tensor Core** | 16 TB/s read per SM | Dedicated path, no L1/L2 contention |
| **SMEM capacity per SM** | 228 KB | compute capability 10.0 |
| **TMEM capacity per SM** | 256 KB | 512 columns × 128 lanes × 4 bytes |
| **L2 cache total** | 126 MB | Partitioned across 2 dies |

---

## 2. The Roofline Model — How to Determine the Bottleneck

For ANY kernel, performance is bounded by:

```
time = max(compute_time, memory_time)
```

### 2.1 Arithmetic Intensity

```
Arithmetic Intensity (AI) = FLOPs / Bytes_transferred

For GEMM: C[M,N] = A[M,K] × B[K,N]
  FLOPs = 2 × M × N × K   (multiply + accumulate per output element)
  
For NVFP4 GEMM:
  Bytes_A = M × K / 2       (FP4 = 0.5 bytes per element)
  Bytes_B = N × K / 2       (FP4 = 0.5 bytes per element)
  Bytes_C = M × N × 2       (FP16 = 2 bytes per element)
  Bytes_SFA = M × (K/16)    (FP8 scale factors, 1 byte each)
  Bytes_SFB = N × (K/16)    (FP8 scale factors, 1 byte each)
  
  Total_Bytes = M×K/2 + N×K/2 + M×N×2 + M×K/16 + N×K/16
              = (M+N)×K×(1/2 + 1/16) + M×N×2
              = (M+N)×K×(9/16) + M×N×2

  AI = 2×M×N×K / Total_Bytes
```

### 2.2 Roofline Crossover Point

```
AI_crossover = Peak_FLOPS / Peak_BW
             = 7,702.5 TFLOPS / 8 TB/s
             = 962.8 FLOPs/Byte

If AI < 962.8 → MEMORY BOUND (bandwidth limited)
If AI > 962.8 → COMPUTE BOUND (Tensor Core limited)
```

This is an EXTREMELY high crossover point. Almost all practical GEMM
sizes with small M are memory-bound on B200 for FP4.

---

## 3. Per-Problem Theoretical Analysis

### 3.1 Problem 1: G=8, M=[80..248], N=4096, K=7168, L=1

Let's trace one representative group: **M=128, N=4096, K=7168**

#### Step 1: Count FLOPs
```
FLOPs = 2 × 128 × 4096 × 7168 = 7,516,192,768 ≈ 7.52 GFLOPs
```

#### Step 2: Count Bytes (HBM traffic)
```
Bytes_A  = 128 × 7168 / 2           = 458,752 bytes  (448 KB)
Bytes_B  = 4096 × 7168 / 2          = 14,680,064 bytes (14 MB)
Bytes_C  = 128 × 4096 × 2           = 1,048,576 bytes (1 MB)
Bytes_SFA = 128 × (7168/16) × 1     = 57,344 bytes   (56 KB)
Bytes_SFB = 4096 × (7168/16) × 1    = 1,835,008 bytes (1.79 MB)
─────────────────────────────────────────────────────────────
Total = 18,079,744 bytes ≈ 17.24 MB
```

#### Step 3: Arithmetic Intensity
```
AI = 7,516,192,768 / 18,079,744 = 415.7 FLOPs/Byte

415.7 < 962.8 → MEMORY BOUND
```

#### Step 4: Compute Time (if compute were the bottleneck)
```
compute_time = FLOPs / Peak_TFLOPS
             = 7.52 × 10⁹ / (7,702.5 × 10¹²)
             = 0.000976 μs ≈ 0.001 μs per group
```

#### Step 5: Memory Time (the actual bottleneck)
```
memory_time = Total_Bytes / HBM_BW
            = 18,079,744 / (8 × 10¹²)
            = 2.26 μs per group
```

#### Step 6: All 8 Groups Combined

For all 8 groups with M=[80, 176, 128, 72, 64, 248, 96, 160]:
```
Per-group bytes (B is dominant since N×K >> M×K):
  Bytes_B per group = 4096 × 7168 / 2 = 14.0 MB  (SAME for all groups)
  Bytes_SFB per group = 4096 × 448 = 1.79 MB      (SAME for all groups)
  
  Bytes_A varies: M × 7168 / 2
  Bytes_C varies: M × 4096 × 2
  Bytes_SFA varies: M × 448

Group-by-group:
  M=80:  A=286K + B=14.0M + C=640K + SFA=35K + SFB=1.79M = 16.75 MB
  M=176: A=630K + B=14.0M + C=1.37M + SFA=77K + SFB=1.79M = 17.87 MB
  M=128: A=458K + B=14.0M + C=1.0M + SFA=56K + SFB=1.79M = 17.30 MB
  M=72:  A=258K + B=14.0M + C=576K + SFA=32K + SFB=1.79M = 16.66 MB
  M=64:  A=229K + B=14.0M + C=512K + SFA=28K + SFB=1.79M = 16.56 MB
  M=248: A=889K + B=14.0M + C=1.94M + SFA=109K + SFB=1.79M = 18.73 MB
  M=96:  A=344K + B=14.0M + C=768K + SFA=42K + SFB=1.79M = 16.94 MB
  M=160: A=573K + B=14.0M + C=1.25M + SFA=70K + SFB=1.79M = 17.68 MB

Total bytes all 8 groups = ~138.5 MB
```

#### Step 7: Theoretical Minimum Time
```
If all groups run sequentially (single stream):
  time = 138.5 MB / 8 TB/s = 17.31 μs

If B matrix is cached in L2 (126 MB L2 cache):
  B matrix = 14.0 MB per group, same across groups
  First load: 14.0 MB from HBM
  Subsequent 7 loads: from L2 (~12 TB/s)
  
  HBM traffic = 14.0 MB (first B) + 8 × (A+C+SFA+SFB varying) + 7 × 1.79 MB (SFB)
  ≈ 14.0 + 8×(avg 1.0 MB) + 12.53 = ~34.5 MB from HBM
  L2 traffic = 7 × 14.0 MB = 98 MB from L2
  
  time_hbm = 34.5 MB / 8 TB/s = 4.31 μs
  time_l2  = 98 MB / 12 TB/s = 8.17 μs
  
  These overlap, so: time ≈ max(4.31, 8.17) = 8.17 μs
  
  But B is different per group only if N differs. Here N=4096 for all.
  Actually B has SAME N and K → B matrix data is IDENTICAL across groups!
  So L2 caching of B is extremely effective.

Realistic estimate with L2 caching:
  Unique HBM reads = 1×B(14.0M) + 1×SFB(1.79M) + 8×A(avg 0.46M) + 8×C(avg 0.88M) + 8×SFA(avg 0.06M)
                   = 14.0 + 1.79 + 3.68 + 7.04 + 0.48 = 27.0 MB
  
  time = 27.0 MB / 8 TB/s = 3.37 μs
  
  But we also need to WRITE C back:
  C writes = 8 × (avg M × 4096 × 2) = 8 × 0.88 MB = 7.04 MB
  
  Total HBM traffic (read + write) = 27.0 + 7.04 = 34.0 MB
  time = 34.0 MB / 8 TB/s = 4.25 μs
```

**Challenge SOL target: 18.833 μs**

Wait — our theoretical minimum (4.25 μs) is much less than the SOL target (18.833 μs).
This means the SOL is computed differently. Let me re-read the challenge:

> "Speed of light analysis based on max(FP4 Tensor Core math throughput, DRAM memory throughput)"

The challenge SOL uses **per-group sequential execution without L2 caching**.
Let me recalculate:

```
Per-group compute time:
  avg_M = (80+176+128+72+64+248+96+160)/8 = 128
  FLOPs_per_group = 2 × 128 × 4096 × 7168 = 7.52 GFLOPs
  compute_time_per_group = 7.52e9 / 7702.5e12 = 0.000976 μs
  total_compute = 8 × 0.000976 = 0.0078 μs

Per-group memory time:
  bytes_per_group = (128+4096) × 7168 × 9/16 + 128 × 4096 × 2
                  = 4224 × 7168 × 0.5625 + 1,048,576
                  = 17,031,168 + 1,048,576 = 18,079,744 bytes
  memory_time_per_group = 18,079,744 / 8e12 = 2.26 μs
  total_memory = 8 × 2.26 = 18.08 μs

SOL = max(total_compute, total_memory) = max(0.0078, 18.08) = 18.08 μs
```

**This matches the challenge SOL of 18.833 μs!** (small difference from using exact M values)

Let me verify with exact M values:

```
Sum of all group bytes:
  For each group i: bytes_i = (M_i + 4096) × 7168 × 9/16 + M_i × 4096 × 2

  M=80:  (80+4096)×7168×0.5625 + 80×4096×2 = 16,832,640 + 655,360 = 17,488,000
  M=176: (176+4096)×7168×0.5625 + 176×4096×2 = 17,219,520 + 1,441,792 = 18,661,312
  M=128: (128+4096)×7168×0.5625 + 128×4096×2 = 17,026,080 + 1,048,576 = 18,074,656
  M=72:  (72+4096)×7168×0.5625 + 72×4096×2 = 16,800,384 + 589,824 = 17,390,208
  M=64:  (64+4096)×7168×0.5625 + 64×4096×2 = 16,768,128 + 524,288 = 17,292,416
  M=248: (248+4096)×7168×0.5625 + 248×4096×2 = 17,509,632 + 2,031,616 = 19,541,248
  M=96:  (96+4096)×7168×0.5625 + 96×4096×2 = 16,897,152 + 786,432 = 17,683,584
  M=160: (160+4096)×7168×0.5625 + 160×4096×2 = 17,155,008 + 1,310,720 = 18,465,728

  Total = 144,597,152 bytes = 137.9 MB
  
  time = 137.9 MB / 8 TB/s = 17.24 μs
```

Close to 18.833 μs. The remaining gap is from overhead (kernel launch, scale factor prep).

---

### 3.2 Problem 2: G=8, M=[40..196], N=7168, K=2048, L=1

Representative: **M=128, N=7168, K=2048**

```
FLOPs = 2 × 128 × 7168 × 2048 = 3,758,096,384 ≈ 3.76 GFLOPs

Bytes:
  A  = 128 × 2048 / 2 = 131,072 (128 KB)
  B  = 7168 × 2048 / 2 = 7,340,032 (7 MB)
  C  = 128 × 7168 × 2 = 1,835,008 (1.75 MB)
  SFA = 128 × 128 = 16,384 (16 KB)
  SFB = 7168 × 128 = 917,504 (896 KB)
  Total = 10,240,000 ≈ 9.77 MB

AI = 3.76e9 / 10.24e6 = 367 FLOPs/Byte → MEMORY BOUND

All 8 groups total bytes (using exact M values):
  Sum ≈ 8 × ~10 MB ≈ 80 MB  (B dominates, same for all groups)
  
  Exact: each group = (M_i + 7168) × 2048 × 9/16 + M_i × 7168 × 2
  
  M=40:  (40+7168)×2048×0.5625 + 40×7168×2 = 8,306,688 + 573,440 = 8,880,128
  M=76:  (76+7168)×2048×0.5625 + 76×7168×2 = 8,348,160 + 1,089,536 = 9,437,696
  M=168: (168+7168)×2048×0.5625 + 168×7168×2 = 8,454,144 + 2,408,448 = 10,862,592
  M=72:  (72+7168)×2048×0.5625 + 72×7168×2 = 8,343,552 + 1,032,192 = 9,375,744
  M=164: (164+7168)×2048×0.5625 + 164×7168×2 = 8,449,536 + 2,351,104 = 10,800,640
  M=148: (148+7168)×2048×0.5625 + 148×7168×2 = 8,431,104 + 2,121,728 = 10,552,832
  M=196: (196+7168)×2048×0.5625 + 196×7168×2 = 8,486,400 + 2,809,856 = 11,296,256
  M=160: (160+7168)×2048×0.5625 + 160×7168×2 = 8,445,312 + 2,293,760 = 10,739,072

  Total = 81,944,960 bytes = 78.15 MB
  
  time = 78.15 MB / 8 TB/s = 9.77 μs
```

**Challenge SOL: 10.667 μs** — our calculation gives 9.77 μs. Close! Gap is overhead.

---

### 3.3 Problem 3: G=2, M=[192,320], N=3072, K=4096, L=1

```
Group 1: M=192, N=3072, K=4096
  FLOPs = 2 × 192 × 3072 × 4096 = 4,831,838,208 ≈ 4.83 GFLOPs
  Bytes = (192+3072)×4096×9/16 + 192×3072×2
        = 3264×4096×0.5625 + 1,179,648
        = 7,520,256 + 1,179,648 = 8,699,904 (8.30 MB)

Group 2: M=320, N=3072, K=4096
  FLOPs = 2 × 320 × 3072 × 4096 = 8,053,063,680 ≈ 8.05 GFLOPs
  Bytes = (320+3072)×4096×9/16 + 320×3072×2
        = 3392×4096×0.5625 + 1,966,080
        = 7,815,168 + 1,966,080 = 9,781,248 (9.33 MB)

Total = 18,481,152 bytes = 17.62 MB
time = 17.62 MB / 8 TB/s = 2.20 μs
```

**Challenge SOL: 2.406 μs** — our calculation gives 2.20 μs. Very close!

---

### 3.4 Problem 4: G=2, M=[128,384], N=4096, K=1536, L=1

```
Group 1: M=128, N=4096, K=1536
  FLOPs = 2 × 128 × 4096 × 1536 = 1,610,612,736 ≈ 1.61 GFLOPs
  Bytes = (128+4096)×1536×9/16 + 128×4096×2
        = 4224×1536×0.5625 + 1,048,576
        = 3,650,616 + 1,048,576 = 4,699,192 (4.48 MB)

Group 2: M=384, N=4096, K=1536
  FLOPs = 2 × 384 × 4096 × 1536 = 4,831,838,208 ≈ 4.83 GFLOPs
  Bytes = (384+4096)×1536×9/16 + 384×4096×2
        = 4480×1536×0.5625 + 3,145,728
        = 3,870,720 + 3,145,728 = 7,016,448 (6.69 MB)

Total = 11,715,640 bytes = 11.18 MB
time = 11.18 MB / 8 TB/s = 1.40 μs
```

**Challenge SOL: 1.525 μs** — our calculation gives 1.40 μs. Close!

---

## 4. Kernel Execution Trace — What Happens Inside One GEMM

Here we trace the execution of a single NVFP4 GEMM tile on B200,
step by step, with cycle-level timing.

### 4.1 Warp-Specialized Persistent Kernel Architecture

The cuBLAS NVFP4 kernel on B200 uses **warp specialization**:

```
Thread Block = 192 threads = 6 warps:
  ┌─────────────────────────────────────────────┐
  │ Warp 0-3: Epilogue warps (store C to GMEM)  │
  │ Warp 4:   MMA warp (Tensor Core compute)    │
  │ Warp 5:   TMA warp (async data loading)     │
  └─────────────────────────────────────────────┘
```

### 4.2 Tile Dimensions

```
MMA tile: 128 × 128 × K_tile
  K_tile = mma_inst_k × 4 (4 MMA instructions per K-tile)
  For FP4: mma_inst_k = 64, so K_tile = 256

Tile sizes in bytes:
  A tile: 128 × 256 / 2 = 16,384 bytes = 16 KB (FP4)
  B tile: 128 × 256 / 2 = 16,384 bytes = 16 KB (FP4)
  SFA tile: 128 × (256/16) = 2,048 bytes = 2 KB (FP8)
  SFB tile: 128 × (256/16) = 2,048 bytes = 2 KB (FP8)
  C tile: 128 × 128 × 2 = 32,768 bytes = 32 KB (FP16)
  
  Total per tile load: 16 + 16 + 2 + 2 = 36 KB
```

### 4.3 Pipeline Stages (Double Buffering)

```
SMEM budget: 228 KB per SM
  A buffer (2 stages): 2 × 16 KB = 32 KB
  B buffer (2 stages): 2 × 16 KB = 32 KB
  SFA buffer (2 stages): 2 × 2 KB = 4 KB
  SFB buffer (2 stages): 2 × 2 KB = 4 KB
  C buffer (1 stage): 32 KB
  Barriers + metadata: ~4 KB
  ─────────────────────────────────
  Total SMEM: ~108 KB (fits in 228 KB with room for 3-4 stages)

With 3 stages:
  A+B+SFA+SFB: 3 × 36 KB = 108 KB
  C: 32 KB
  Total: 140 KB — fits easily in 228 KB
```

### 4.4 Cycle-by-Cycle Trace for One Output Tile

For M=128, N=4096, K=7168, tile 128×128:
```
Number of K-tiles = K / K_tile = 7168 / 256 = 28 tiles
Number of N-tiles = N / 128 = 4096 / 128 = 32 tiles
Number of M-tiles = M / 128 = 128 / 128 = 1 tile
Total output tiles = 1 × 32 = 32 tiles
```

**For ONE output tile (128×128), processing 28 K-tiles:**

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1: TMA Load (Warp 5, async)                                │
│                                                                  │
│   Load A[128×256] from HBM → SMEM: 16 KB                        │
│   Load B[128×256] from HBM → SMEM: 16 KB                        │
│   Load SFA[128×16] from HBM → SMEM: 2 KB                        │
│   Load SFB[128×16] from HBM → SMEM: 2 KB                        │
│   Total: 36 KB per K-tile                                        │
│                                                                  │
│   TMA bandwidth: ~8 TB/s (HBM limited)                           │
│   Time per K-tile load: 36 KB / 8 TB/s = 4.5 ns                 │
│   Time for 28 K-tiles: 28 × 4.5 = 126 ns                        │
│                                                                  │
│   But TMA is ASYNC — overlaps with compute!                      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Phase 2: MMA Compute (Warp 4, Tensor Core)                       │
│                                                                  │
│   Per K-tile: 4 × tcgen05.mma instructions                      │
│   Each MMA: 128×128×64 FP4 multiply-accumulate                  │
│   FLOPs per MMA = 2 × 128 × 128 × 64 = 2,097,152               │
│   FLOPs per K-tile = 4 × 2,097,152 = 8,388,608                  │
│                                                                  │
│   tcgen05.mma latency: ~11 cycles = 7.3 ns                      │
│   But throughput is 1 MMA per cycle when pipelined               │
│                                                                  │
│   Per-SM FP4 throughput: 52.04 TFLOPS                            │
│   Time per K-tile compute: 8.39e6 / 52.04e12 = 0.161 ns         │
│   Time for 28 K-tiles: 28 × 0.161 = 4.5 ns                      │
│                                                                  │
│   COMPUTE IS MUCH FASTER THAN MEMORY!                            │
│   Ratio: 126 ns (memory) / 4.5 ns (compute) = 28×               │
│   This tile is 28× memory-bound                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Phase 3: Epilogue Store (Warps 0-3)                              │
│                                                                  │
│   Convert FP32 accumulator → FP16 output                         │
│   Store C[128×128] from TMEM → SMEM → HBM via TMA               │
│   Bytes: 128 × 128 × 2 = 32 KB                                  │
│   Time: 32 KB / 8 TB/s = 4.0 ns                                 │
│                                                                  │
│   Overlaps with next tile's TMA load (pipelined)                 │
└──────────────────────────────────────────────────────────────────┘
```

### 4.5 Total Time for One Group (M=128, N=4096, K=7168)

```
32 output tiles, each taking:
  memory_time = 36 KB × 28 K-tiles / 8 TB/s = 126 ns (load)
              + 32 KB / 8 TB/s = 4 ns (store)
              = 130 ns per output tile

  compute_time = 4.5 ns per output tile (negligible)

Total = 32 × 130 ns = 4,160 ns = 4.16 μs

But this assumes perfect pipelining. Real overhead:
  - Kernel launch: ~5-10 μs (HUGE for small problems!)
  - TMA descriptor setup: ~1 μs
  - Barrier synchronization: ~0.5 μs per tile transition
  - Epilogue drain: ~0.5 μs
  
  Realistic: 4.16 + 5 + 1 + 0.5 = ~10.7 μs for one group
```

**This is why the challenge SOL is 18.8 μs for 8 groups** — kernel launch
overhead dominates for these small problem sizes!

---

## 5. Why Kernel Launch Overhead Matters

```
Kernel launch overhead on B200:
  - CUDA kernel launch: ~5-10 μs from Python
  - cuBLAS kernel selection: ~2-3 μs
  - TMA descriptor initialization: ~1 μs
  - Total per-group overhead: ~8-13 μs

For 8 groups sequentially: 8 × 10 μs = 80 μs overhead alone!

This is why optimizations matter:
  1. CUDA streams: overlap launch overhead with compute
  2. CUDA graphs: capture once, replay with ~1 μs overhead
  3. Persistent kernel: single launch for ALL groups (~5 μs total)
  4. CuTe DSL grouped GEMM: single kernel, persistent tile scheduling
```

---

## 6. Memory Access Pattern Deep Dive

### 6.1 TMA (Tensor Memory Accelerator) — B200's Secret Weapon

```
Traditional GPU memory access:
  Thread → Register → Shared Memory → L1 → L2 → HBM
  Each level adds latency and consumes bandwidth

TMA on B200:
  TMA Engine → HBM → L2 → SMEM (bypasses L1 entirely)
  - Async: doesn't block warps
  - Bulk: moves entire tiles (up to 128 bytes per transaction)
  - Aligned: requires 128-byte alignment for peak performance
  - Multicast: can broadcast to multiple SMs in a cluster

TMA throughput: limited by HBM bandwidth (8 TB/s)
TMA latency: ~420 cycles for first access, then pipelined
```

### 6.2 Scale Factor Access Pattern

```
NVFP4 block scaling: every 16 FP4 elements share one FP8 scale factor

For A[M=128, K=7168]:
  Scale factors: 128 × (7168/16) = 128 × 448 = 57,344 FP8 values
  = 57,344 bytes = 56 KB

Blocked layout for TMA:
  (M//128, K//16//4, 32, 4, 4) = (1, 112, 32, 4, 4)
  This ensures each TMA load fetches a contiguous 32×16 = 512 byte block
  of scale factors, matching the MMA tile's scale factor requirements.

Access pattern per MMA tile (128×256):
  SFA needed: 128 × (256/16) = 128 × 16 = 2,048 bytes
  SFB needed: 128 × (256/16) = 128 × 16 = 2,048 bytes
  Total SF per tile: 4,096 bytes = 4 KB
  
  This is ~10% of the A+B tile data (36 KB)
  Scale factors add ~11% memory overhead
```

### 6.3 Register Usage Analysis

```
Per-thread register budget for NVFP4 GEMM:

MMA warp (Warp 4):
  - Accumulator (FP32): 128×128 / 32 threads = 512 FP32 values = 512 registers
    But TMEM holds accumulator! So only ~32 registers for control flow.
  
TMA warp (Warp 5):
  - TMA descriptors: ~16 registers
  - Loop counters, addresses: ~16 registers
  - Total: ~32 registers

Epilogue warps (Warps 0-3):
  - FP32→FP16 conversion: ~32 registers per thread
  - Store addresses: ~8 registers
  - Total: ~40 registers

With 40 registers/thread, 192 threads/block:
  Total registers = 192 × 40 = 7,680 registers per block
  Available: 65,536 per SM
  Occupancy: could fit 8 blocks per SM (but limited by SMEM)
  
  With 140 KB SMEM per block: 228 / 140 = 1 block per SM
  → Register file is NOT the bottleneck (only 12% utilized)
  → SMEM is the occupancy limiter
```

---

## 7. Comparison: What If This Were Softmax Instead?

To teach the general methodology, let's compare GEMM with softmax:

### 7.1 Softmax: y_i = exp(x_i) / Σ exp(x_j)

```
For a vector of length N=4096 (FP16):

Step 1: Find max (for numerical stability)
  - Read N elements from HBM: 4096 × 2 = 8 KB
  - N comparisons: 4096 FLOPs
  - Store max in register: 1 register

Step 2: Compute exp(x_i - max) and sum
  - Read N elements again (or from SMEM if cached): 8 KB
  - N subtractions + N exponentials: ~4096 × 10 = 40,960 FLOPs (exp is ~10 FLOPs)
  - N additions for sum: 4096 FLOPs
  - Store partial results in SMEM: 8 KB

Step 3: Divide by sum
  - Read N partial results from SMEM: 8 KB
  - N divisions: 4096 FLOPs
  - Write N results to HBM: 8 KB

Total:
  HBM reads: 8 KB (can be 16 KB if not cached)
  HBM writes: 8 KB
  SMEM reads: 8 KB
  SMEM writes: 8 KB
  FLOPs: ~49,152

  Memory time: 16 KB / 8 TB/s = 2 ns (HBM)
  SMEM time: 16 KB / 8 TB/s per SM = 2 ns
  Compute time: 49,152 / 52.04e12 = 0.001 ns

  → Softmax is EXTREMELY memory-bound (AI = 49K / 32K = 1.5 FLOPs/Byte)
  → Dominated by HBM access latency (~280 ns) not bandwidth
  → This is why FlashAttention fuses softmax with GEMM!
```

### 7.2 Key Insight: Latency vs Bandwidth Bound

```
Small operations (softmax on 4096 elements):
  - Data fits in one cache line
  - Dominated by LATENCY (280 ns HBM, 20 ns SMEM)
  - Bandwidth is irrelevant

Large operations (GEMM with M=128, N=4096, K=7168):
  - Data is many MB
  - Dominated by BANDWIDTH (8 TB/s HBM)
  - Latency is hidden by pipelining

The transition point:
  Latency-bound when: data_size < BW × latency
  = 8 TB/s × 280 ns = 2.24 MB
  
  If your working set < 2.24 MB → latency-bound
  If your working set > 2.24 MB → bandwidth-bound
```

---

## 8. Summary: How to Do This Analysis for ANY Kernel

### Step-by-Step Recipe:

```
1. COUNT THE FLOPS
   - How many multiply-adds does your kernel perform?
   - For GEMM: 2×M×N×K
   - For softmax: ~10×N (exp is expensive)
   - For elementwise: N

2. COUNT THE BYTES
   - How many bytes read from HBM?
   - How many bytes written to HBM?
   - Include ALL tensors (inputs, outputs, scale factors, indices)
   - Remember: FP4=0.5B, FP8=1B, FP16=2B, FP32=4B

3. COMPUTE ARITHMETIC INTENSITY
   AI = FLOPs / Bytes
   
4. COMPARE WITH ROOFLINE CROSSOVER
   Crossover = Peak_TFLOPS / Peak_BW
   B200 FP4: 7702.5 / 8000 = 962.8 FLOPs/Byte
   B200 FP16: 1929.2 / 8000 = 241.2 FLOPs/Byte
   
5. DETERMINE BOTTLENECK
   If AI < crossover → MEMORY BOUND → time = Bytes / BW
   If AI > crossover → COMPUTE BOUND → time = FLOPs / Peak_TFLOPS

6. ADD OVERHEAD
   - Kernel launch: 5-10 μs
   - TMA setup: ~1 μs
   - Python dispatch: ~10-50 μs
   - Synchronization: ~1-5 μs

7. CONSIDER CACHING
   - Does data fit in L2 (126 MB)?
   - Is data reused across tiles/groups?
   - L2 BW (~12 TB/s) > HBM BW (8 TB/s)
```

### Quick Reference — B200 Roofline Crossover Points:

| Precision | Peak TFLOPS | HBM BW | Crossover (FLOPs/Byte) |
|-----------|-------------|--------|----------------------|
| FP4 | 7,702.5 | 8 TB/s | 962.8 |
| FP8 | 3,851.4 | 8 TB/s | 481.4 |
| FP16 (FP16 acc) | 1,929.2 | 8 TB/s | 241.2 |
| FP16 (FP32 acc) | 964.6 | 8 TB/s | 120.6 |
| FP32 | ~67 | 8 TB/s | 8.4 |
| FP64 | 44.8 | 8 TB/s | 5.6 |

> **Rule of thumb**: On B200, you need AI > 963 FLOPs/Byte to be compute-bound
> in FP4. That means K must be enormous relative to M and N.
> For the challenge problems (small M), ALL are memory-bound.

---

## 9. Challenge SOL Verification Table

| Problem | G | Avg M | N | K | Total Bytes | HBM Time (μs) | Compute Time (μs) | SOL = max() | Challenge SOL |
|---------|---|-------|------|------|-------------|----------------|-------------------|-------------|---------------|
| 1 | 8 | 128 | 4096 | 7168 | 138.5 MB | 17.3 | 0.008 | **17.3** | **18.833** |
| 2 | 8 | 128 | 7168 | 2048 | 78.2 MB | 9.8 | 0.004 | **9.8** | **10.667** |
| 3 | 2 | 256 | 3072 | 4096 | 17.6 MB | 2.2 | 0.002 | **2.2** | **2.406** |
| 4 | 2 | 256 | 4096 | 1536 | 11.2 MB | 1.4 | 0.001 | **1.4** | **1.525** |

All problems are **heavily memory-bound** (compute is 1000-4000× faster than memory).
The gap between our calculation and challenge SOL is kernel overhead (~10-15%).
