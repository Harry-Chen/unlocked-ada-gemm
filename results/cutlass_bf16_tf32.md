# CUTLASS BF16 / TF32 GEMM sweep — unlocked RTX 4090 (GPU 5)

- Machine: 8x RTX 4090 48GB (sm_89, Ada), driver 610.43.02, CUDA 13.3. All runs on GPU 5 (`CUDA_VISIBLE_DEVICES=5`).
- TCs unlocked: BF16/FP16 fp32-acc pure-issue peak **331.6 TF/s**, TF32 **165.6 TF/s** (2x stock).
- Source: `~/gemm-study/cutlass_bf16_tf32.cu`, binary `~/gemm-study/cutlass_bf16_tf32`.
- Build: `/usr/local/cuda-13.3/bin/nvcc -O3 -arch=sm_89 -I ~/cutlass/include cutlass_bf16_tf32.cu -o cutlass_bf16_tf32` (CUTLASS v4.2.1 headers, 2.x device API).
- Layout TN: A row-major MxK, B col-major KxN, C row-major MxN; fp32 accumulate, same-dtype out (bf16 out / tf32 out); alpha=1, beta=0. Warp tile 64x64x32 for all; instruction 16x8x16 (bf16) / 16x8x8 (tf32); swizzle = `GemmIdentityThreadblockSwizzle<N>` (N = log2 group size).
- Raw logs: `~/gemm-study/results/run_bf16_tf32_{4096,8192,16384}.log`.

## BF16 (fp32 acc) — TF/s

| TB tile | warp | stages | swizzle | 4096³ | 8192³ | 16384³ |
|---|---|---|---|---|---|---|
| 128x128x32 | 64x64 | 3 | 1 | 280.0 | 233.9 | 214.9 |
| 128x128x32 | 64x64 | 3 | 4 | 281.5 | 279.2 | 265.7 |
| 128x128x32 | 64x64 | 4 | 1 | 273.4 | 193.9 | 111.4 |
| 128x128x32 | 64x64 | 4 | 4 | 277.2 | 277.0 | 255.6 |
| **128x256x32** | 64x64 | **3** | 1 | 307.9 | 283.8 | 216.8 |
| **128x256x32** | 64x64 | **3** | **4** | 307.7 | 285.9 | **289.9** ★ |
| 128x256x32 | 64x64 | 4 | 1 | 307.0 | 287.8 | 217.4 |
| 128x256x32 | 64x64 | 4 | 4 | 307.0 | **287.9** ★ | 285.6 |
| 256x128x32 | 64x64 | 3 | 1 | **308.0** ★ | 287.7 | 220.5 |
| 256x128x32 | 64x64 | 3 | 4 | 295.4 | 285.6 | 287.0 |
| 256x128x32 | 64x64 | 4 | 1 | 289.5 | 286.5 | 177.9 |
| 256x128x32 | 64x64 | 4 | 4 | 290.0 | 286.5 | 285.0 |

**Best per shape:** 4096³: 256x128x32/st3/swz1 = 308.0 (128x256x32/st3/swz1 = 307.9, tied); 8192³: 128x256x32/st4/swz4 = 287.9; 16384³: **128x256x32/st3/swz4 = 289.9**.

**Efficiency vs 331.6 TF/s ceiling:** 4096³ 92.9% · 8192³ 86.8% · 16384³ 87.4%.

## TF32 (fp32 acc) — TF/s

| TB tile | warp | stages | swizzle | 4096³ | 8192³ | 16384³ |
|---|---|---|---|---|---|---|
| 128x128x32 | 64x64 | 3 | 1 | 124.0 | 103.9 | 54.3 |
| **128x128x32** | 64x64 | **3** | **4** | 123.1 | **141.9** ★ | **141.2** ★ |
| 128x128x32 | 64x64 | 4 | 1/4 | FAIL | FAIL | FAIL |
| 128x256x32 | 64x64 | 3 | 1/4 | FAIL | FAIL | FAIL |
| 128x256x32 | 64x64 | 4 | 1/4 | FAIL | FAIL | FAIL |
| 256x128x32 | 64x64 | 3 | 1/4 | FAIL | FAIL | FAIL |
| 256x128x32 | 64x64 | 4 | 1/4 | FAIL | FAIL | FAIL |
| 128x256x32 | 64x64 | 2 | 1 | 134.8 | 135.6 | 105.5 |
| **128x256x32** | 64x64 | **2** | **4** | **136.1** ★ | 138.7 | 140.7 |
| 256x128x32 | 64x64 | 2 | 1 | 124.1 | 122.6 | 105.0 |
| 256x128x32 | 64x64 | 2 | 4 | 124.6 | 123.3 | 124.4 |

FAIL = SMEM over the 99KB opt-in limit (4-byte tf32 tiles: 128x128 st4 = 128KB, 256-wide st3 = 144KB, st4 = 192KB; 128x128 st3 = 96KB and 256-wide st2 = 96KB fit). stages=2 rows are bonus configs outside the requested sweep so the 256-wide tf32 tiles could run at all.

**Best per shape:** 4096³: 128x256x32/st2/swz4 = 136.1; 8192³: 128x128x32/st3/swz4 = 141.9; 16384³: 128x128x32/st3/swz4 = 141.2 (128x256x32/st2/swz4 = 140.7, tied within noise).

**Efficiency vs 165.6 TF/s ceiling:** 4096³ 82.2% · 8192³ 85.7% · 16384³ 85.3%.

## Findings

- **BF16 matches the FP16 pattern**: 128x256x32/st3 + swizzle-4 wins at 16384³ (289.9 vs FP16's 276.7), and BF16 ≈ FP16 overall (same TC path) as expected. At 4096³ (wave-quantization-free small grid) 308 TF/s = 93% of the doubled pure-issue peak.
- **Swizzle=4 is mandatory at large shapes**: e.g. bf16 128x256 st3 at 16384³ goes 216.8 → 289.9; tf32 128x128 st3 at 16384³ goes 54.3 → 141.2. With swz1, deeper stages can even hurt badly (bf16 128x128 st4: 111.4 at 16384³).
- **TF32 winner**: `128x128x32/st3/swz4` (141.9 @8192³, 141.2 @16384³) — ~85% of the 165.6 ceiling and ≈ half of BF16, as expected (2x instruction issue per FLOP). The FP16-style 128x256 tile can't reach st≥3 in tf32 (SMEM), but the st2 variant + swz4 ties the winner (140.7 @16384³) and wins at 4096³.
- All failures were graceful (CUTLASS status → reported, sweep continued); no other GPUs touched.
