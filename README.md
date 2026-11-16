# unlocked-ada-gemm

**中文摘要**：本项目是"解锁版 RTX 4090（P2P 解锁 + tensor core 全速解锁）的 GEMM 性能研究与工具包"。
核心结论：解锁后 cuBLAS 默认 GEMM"只快 12–34%"是**软件栈按 stock 半速模型选 kernel 的假象**；
用运行时实测选优（本仓库的 `cublaslt_autotune.hpp` shim）或正确的 CUTLASS 配置，
FP16/BF16 GEMM 端到端可达 ~1.5–1.9×、FP8 ~1.9×，逼近硬件实际上限。
完整调查过程与全部数据见 [docs/REPORT.zh.md](docs/REPORT.zh.md)，部署指南见
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

博客：<https://harrychen.xyz/2026/11/17/unlocked-rtx-4090-gemm-perf/>

注：本仓库几乎所有内容由大语言模型完成，因此语言风格可能引起不适，敬请谅解。

---

Unlocking the consumer Ada tensor cores (and PCIe P2P) is only half the story:
**the stock software stack leaves most of the gain on the table**, because cuBLASLt's
heuristic ranking is calibrated for the locked, half-rate hardware. This repo contains:

- **`src/cublaslt_autotune.hpp`** — a drop-in, header-only autotuning wrapper for
  `cublasLtMatmul`. First call per (dtype, shape, layout) measures all heuristic
  candidates on the real buffers and caches the winner; later calls just execute it.
  Never slower than the default pick, up to ~1.9× faster (measured).
- **CUTLASS config tables** for the unlocked card (fp16/bf16/tf32/fp8, 2048³–16384³)
  and the general rules behind them (see [docs/REPORT.zh.md](docs/REPORT.zh.md) §5.2.1).
- The **full investigation**: SASS-level root-cause analysis, a reverse-engineering
  of cuBLASLt's heuristic selection (incl. clock-invariance and LD_PRELOAD
  device-spoofing experiments), tensor-core accumulator precision checks
  (Ada accumulates fp32 at full 24-bit width with round-toward-zero — the A100-style
  narrow-accumulator problem does not apply), and a review of a hand-written GEMM.

Related blog article: <https://harrychen.xyz/2026/11/17/unlocked-rtx-4090-gemm-perf/>.

## Headline numbers (8× unlocked RTX 4090 48G, driver 610.43.02, CUDA 13.3)

FP16 GEMM (fp32 accumulate), square 8192³:

| path | TF/s |
|---|---:|
| pure-issue ceiling (measured) | 331.6 |
| cuBLASLt default heuristic pick | 182 |
| cuBLASLt best in-library algo | 274 |
| CUTLASS 128×256×32 / warp 64×64 / 3 stages | **293.7** |
| CUTLASS + L2 swizzle at 16384³ | 276.7 (vs 192 default) |
| FP8 8192³: default → best | 292.7 → **587.6** (of 660.6 ceiling) |

All 8 cards sustain ~290–302 TF/s concurrently (no power/thermal collapse; ~270–290 W per card
of a 450 W cap — the remaining gap to 331.6 is sustained-clock jitter, not memory or power).

## Quick start

```cpp
#include "src/cublaslt_autotune.hpp"
static ltat::Autotuner tuner;
// replace cublasLtMatmul(lt, desc, &alpha, ...) with:
tuner.run(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, D, Dd, ws, wsBytes, stream);
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for reproducing the unlock itself and building
the tools. `results/` holds the raw sweep tables.

## Repo layout

```
src/       the shim + all benchmark/probe sources (CUDA)
docs/      REPORT.zh.md (full investigation, zh) · DEPLOYMENT.md (deployment guide)
results/   sweep CSVs + CUTLASS result notes
```

## Upstream projects this builds on

- [JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) — Ada tensor-core unlock
- [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules) — consumer-GPU P2P patch
  (combining work by geohot/tinygrad, nimlgen, valdemardi)
- [cicuvc/nvidia-sass-document](https://github.com/cicuvc/nvidia-sass-document) — SASS reference we used for decoding
- NVIDIA CUTLASS (Apache-2.0)

## License & disclaimer

Code in this repo: MIT (see LICENSE). Report/docs: CC-BY-4.0.
Not affiliated with NVIDIA. Unlocking hardware behavior may violate warranty or license
terms — research use only, at your own risk. No NVIDIA binaries, debs, or firmware are
(or may be) redistributed here; the guides point at upstream sources.
