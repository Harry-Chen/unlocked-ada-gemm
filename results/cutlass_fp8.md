# CUTLASS FP8 (E4M3) GEMM on sm_89 (unlocked RTX 4090)

**Date:** 2026-11-16 · **Host:** the test host (8x RTX 4090 48GB, driver 610.43.02, CUDA 13.3) · **GPU used:** GPU 6 only (`CUDA_VISIBLE_DEVICES=6`) · **CUTLASS:** v4.2.1 headers @ `~/cutlass/include`

## TL;DR

- **2.x device API works for sm_89 FP8** — no 3.x needed. `cutlass::gemm::device::Gemm` + `arch::Sm89` + `OpClassTensorOp` + `InstructionShape=GemmShape<16,8,32>` emits `QMMA.16832.F32.E4M3.E4M3` (verified in SASS; zero HMMA).
- **Two operator modes, big trade-off:**
  - `OpMultiplyAdd` (CUTLASS default for sm89 FP8) = **staged accumulation**: numerically correct fp32-accumulate (maxrel ~0.0007-0.003 vs CPU fp64 ref at all K). Needs **small warp tiles (≤64x32)** to avoid register spills. Best: **505-507 TF/s** (flat across 4k/8k/16k).
  - `OpMultiplyAddFastAccum` = no staging: faster (**591-597 TF/s**) but accumulation error grows with K (up to 10-15% rel at K=16384) — Ada QMMA hardware truncates the fp32 accumulator on chained accumulation. **Not fp32-accumulate quality.**
- **Best staged config:** TB `128x128x64`, warp `64x32x64`, stages 4, swizzle log2=4-5 → **501.4 / 505.4 / 507.0 TF/s** @ 4096³/8192³/16384³ = **~77% of the 660.6 TF/s unlocked ceiling**, **beats cuBLAS best-algo at 16384³** (507.0 vs 493.2).
- **Best fastacc config:** TB `128x256x64`, warp `64x64x64`, stages 3, swizzle 4-5 → **581.4 / 591.2 / 597.4 TF/s** = **~90% of ceiling**; beats cuBLAS best-algo at both 8192³ (+0.6%) and 16384³ (+21%).

## Which code path

**2.x device API** (`cutlass::gemm::device::Gemm`, the same pattern as the existing `~/cutlass_bench.cu` FP16 driver). The 3.x path was investigated and is a **dead end for sm89**: `~/cutlass/include/cutlass/gemm/collective/builders/` contains only sm90/sm100/sm103/sm120 builders (no `sm89_umma_builder`); 3.x on sm89 would fall back to the sm80-style cp.async multistage collective driving the exact same QMMA atom, with much more boilerplate. 2.x already meets/beats cuBLAS, so 3.x was not pursued.

Key template parameters for the 2.x `Gemm` device op (TN layout: A row-major, B column-major):

```cpp
cutlass::gemm::device::Gemm<
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,      // A
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor,   // B
    cutlass::half_t,         cutlass::layout::RowMajor,    // C (fp16 out)
    float,                                                   // fp32 accumulate
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,     // sm89 REQUIRED (sm80 has no e4m3 atoms)
    cutlass::gemm::GemmShape<128,128,64>,                    // threadblock tile
    cutlass::gemm::GemmShape<64,32,64>,                      // warp tile (K must == TBK)
    cutlass::gemm::GemmShape<16,8,32>,                       // QMMA instruction
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<5>,
    /*Stages=*/4, /*AlignA=*/16, /*AlignB=*/16, /*SplitKSerial=*/false,
    /*Operator=*/cutlass::arch::OpMultiplyAdd>;              // staged; or OpMultiplyAddFastAccum
```

Gotcha encountered: the FP16 driver's `Arguments` passes `{B, N}` for B's leading dim; that only works for square problems. Correct col-major B ldb is **K** (`{B, K}`).

## Results — best-of-3 after 300 ms clock warmup (idle GPU ramps 210 MHz → boost; un-warmed runs read up to 20% low)

### Staged accumulation (`OpMultiplyAdd`) — the numerically correct fp32-accumulate path

| Config (TB tile x warp x stages x swizzle) | 4096³ | 8192³ | 16384³ |
|---|---:|---:|---:|
| **128x128x64 w64x32 st4 swz4** | **501.4** | 503.6 | 506.8 |
| **128x128x64 w64x32 st4 swz5** | **501.4** | **505.4** | **507.0** |
| 128x128x64 w32x32 st3 swz1 | 474.4 | 478.7 | 221.8 |
| 128x128x64 w32x32 st3 swz2 | 473.5 | 479.0 | 387.2 |
| 128x128x64 w32x32 st3 swz4 | 473.2 | 468.6 | 470.0 |
| 128x128x64 w32x32 st4 swz1 | 478.7 | 482.9 | 224.0 |
| 128x128x64 w32x32 st4 swz4 | 479.4 | 470.3 | 470.4 |
| 128x128x64 w64x32 st4 swz2 | — | — | 408.6 |
| 128x128x64 w64x64 st3 (REG SPILL) | — | 85-90 | — |
| 128x128x128 w32x32 st2 swz1 | 437.4 | 462.9 | 219.4 |
| 128x128x128 w32x32 st2 swz2 | — | — | 386.1 |

(`—` = not measured. Earlier un-warmed sweep: 546-573 @ 8192³ for small-warp staged configs.)

### FastAccum (`OpMultiplyAddFastAccum`) — fast but reduced precision (see below)

| Config | 4096³ | 8192³ | 16384³ |
|---|---:|---:|---:|
| 128x256x64 w64x64 st3 swz1 | 580.9 | 588.3 | 429.4 |
| 128x256x64 w64x64 st3 swz2 | 580.7 | 586.4 | 592.0 |
| **128x256x64 w64x64 st3 swz4/5/6** | **581.3** | **591.2** | **597.4** |
| 256x128x64 w64x64 st4 swz1 | 580.9 | 591.2 | 432.1 |
| 256x128x64 w64x64 st4 swz4 | 581.1 | 591.2 | 596.8 |
| 128x128x64 w64x64 st3 swz1 | 545.2 | 547.5 | 424.3 |
| 128x128x128 w64x64 st3 swz1 | 554.9 | 564.1 | 224.7 |
| 128x256x128 w64x64 st2 swz2 | — | 556.1 | 544.6 |

### Reference points

| | 4096³ | 8192³ | 16384³ |
|---|---:|---:|---:|
| cuBLAS default heuristic | — | 292.7 | — |
| cuBLAS best-of-32 algos | — | 587.6 | 493.2 |
| **CUTLASS staged best** | 501.4 | 505.4 | **507.0 (beats cuBLAS)** |
| **CUTLASS fastacc best** | 581.4 | 591.2 (beats cuBLAS) | **597.4 (+21% vs cuBLAS)** |
| Unlocked FP8 pure-issue ceiling | 660.6 | 660.6 | 660.6 |
| Staged efficiency | 75.9% | 76.5% | 76.7% |
| Fastacc efficiency | 88.0% | 89.5% | 90.4% |

## Why the default CUTLASS config is slow on sm_89 (and the fix)

The CUTLASS sm89 FP8 default (`DefaultGemmConfigurationSm89F8`) is TB 128x256x64, warp 64x64x64, `OpMultiplyAdd`. On sm_89, `arch::detail::is_sm89_staged_policy_v` (in `arch/mma_sm89.h`) is true for FP8+16x8x32+OpMultiplyAdd, so `MmaMultistage` runs **staged accumulation**: a second accumulator fragment `tmp_accum_` accumulates QMMA results and is periodically flushed into the real accumulator. This **doubles accumulator registers**:

- warp 64x64: 128 floats accum + 128 floats tmp = 256 regs/thread → `REG:255 STACK:336` → **local-memory spills → ~90 TF/s**.
- Fix: smaller warp tile. warp 64x32 → 64+64=128 regs → no spill → ~505 TF/s.

Constraint discovered: staged configs are limited to **128x128 TB tiles** — a 128x256 tile with 32-64-wide warps needs 16 warps (512 threads), and 512 × ~190 regs exceeds the 64K-reg/SM regfile → launch fails (`cudaErrorLaunchOutOfResources`).

## Precision (CPU fp64 reference, random e4m3 data, |v|≤0.469)

Measured on sampled output elements; "staged" = OpMultiplyAdd, "fastacc" = OpMultiplyAddFastAccum:

| Path | K=512 | K=2048 | K=8192 | K=16384 | K=16384 (mixed sign) |
|---|---|---|---|---|---|
| staged (TBK=64) | 0.0007 | 0.0007 | 0.0007 | 0.0007 | max 0.003 / avg 0.0004 |
| staged TBK=128 | — | — | — | **0.154 (FAIL)** | — |
| fastacc | 0.007 | 0.031 | 0.099 | 0.154 | max 0.101 / avg 0.006 |

Conclusions:
- **Staged with TBK=64 is the only fp32-accumulate-quality path** — error is at the fp16-output rounding level and does NOT grow with K.
- Staging flush frequency is tied to `kWarpGemmIterations = TBK/32`; TBK=128 doubles the unstaged chain (2→4 QMMAs) and precision collapses. **Do not use TBK=128 for staged FP8.**
- FastAccum (and any unstaged chain) degrades with K — the known Ada QMMA reduced-precision fp32 accumulator. Fine for inference-ish workloads with short K or tolerance for ~1-10% error; not fp32 semantics.

## Threadblock swizzle matters at 16384³

L2-locality rasterization (`GemmIdentityThreadblockSwizzle<log2>`) is decisive at 16384³ where tiles no longer fit L2:

| Config | swz1 | swz2 | swz3 | swz4 | swz5 |
|---|---:|---:|---:|---:|---:|
| staged 128x128 w32x32 st3 | 221.8 | 387.2 | 387.2 | 466.5 | 470.1 |
| staged 128x128 w64x32 st4 | 223.9 | 408.6 | 408.9 | 506.8 | 507.0 |
| fastacc 128x256 st3 | 429.4 | 592.0 | 592.0 | 597.4 | 597.4 |

At 4096³/8192³ swizzle is a wash (≤1%).

## Other findings / gotchas

- **SMEM opt-in is automatic** in CUTLASS v4.2.1's 2.x `device::Gemm` (`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` when ≥48KB). 72KB and 96KB (st4 128x256x64) configs launch fine. Configs over ~99-100KB fail at run() (status kError): e.g. 128x128x128 st3 (~96KB + epilogue padding), 256x256x64 st3 fastacc. Such configs are skipped and noted as FAILED(2).
- TBK (K-depth) for FP8 is in **elements**: TBK=64 = 64 bytes/row/stage. The sm89 default WarpShape K = TBK = 64 (kWarpGemmIterations = TBK/32 = 2).
- Failed/odd configs: 256x64x64 & 64x256x64 with warp 32x32 → compile-time static assert (512 threads can't map the 64-wide tile: "Number of iterations must be non-zero"). Warp 64x64 staged → spills. 16-warp configs with >128 regs/thread → launch fail (regfile).
- Memory values in perf runs: `cudaMemset(A, 0x11)` (e4m3 ≈ 0.0352); correctness runs use LCG random bytes.
- Timing: cudaEvent around N iters (50/20/10 for 4k/8k/16k), best of 3 reps, after a 300 ms warmup spin (clocks otherwise ramp from 210 MHz idle and early runs read low).

## Files (remote, `~/gemm-study/`)

| File | What |
|---|---|
| `cutlass_fp8_bench7.cu` / `cutlass_fp8_bench7` | **Final sweep driver** (champion configs x swizzle x {4096,8192,16384}, warmup + best-of-3). Source of the tables above. |
| `cutlass_fp8_check8.cu` / `cutlass_fp8_check8` | Mixed-sign precision validator @ K=16384. |
| `cutlass_fp8_bench3.cu` / `cutlass_fp8_bench3` | Precision-vs-K harness (staged vs fastacc, K=512..16384). |
| `cutlass_fp8_bench5.cu` / `cutlass_fp8_bench5` | Finalists sweep across shapes. |
| `cutlass_fp8_bench.cu`, `..._bench2.cu`, `..._bench4.cu`, `..._bench6.cu` | Earlier iterations (v1 has the `{B,N}` square-only ldb quirk). |
| Logs | `sweep_swz_all.log` (final tables), `sweep_16384_swz.log`, `sweep_final.log`, `check_k.log` (precision vs K), `check_mixed.log`, `run2_8192.log` |

## Reproduce

```bash
# on the remote
/usr/local/cuda-13.3/bin/nvcc -O3 -arch=sm_89 -I ~/cutlass/include \
    ~/gemm-study/cutlass_fp8_bench7.cu -o ~/gemm-study/cutlass_fp8_bench7
CUDA_VISIBLE_DEVICES=6 ~/gemm-study/cutlass_fp8_bench7 4096 8192 16384
```

## Bottom line

CUTLASS 2.x + sm_89 delivers a correct, fast FP8 GEMM: **~507 TF/s with full fp32-accumulate semantics (76.7% of the unlocked 660.6 TF/s ceiling)**, beating cuBLAS best-algo at 16384³. If the consumer can tolerate reduced accumulation precision (error grows ~linearly-ish with K, 10-15% at K=16384), `OpMultiplyAddFastAccum` with 128x256x64 tiles reaches **~597 TF/s (90.4% of ceiling)**, beating cuBLAS everywhere measured.
