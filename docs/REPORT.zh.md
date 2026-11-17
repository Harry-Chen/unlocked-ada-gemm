# 解锁版 RTX 4090 的 GEMM 完整解决方案

> 机器：8× RTX 4090 48G（AD102, sm_89），驱动 610.43.02（P2P 补丁），CUDA 13.3，
> 内核 6.8.0-149，`iommu=pt`，gpu-unlock 1.1.1 全 8 卡生效（FP32 累加路径 TC 发射速率翻倍）。
> 所有测量在同一解锁状态下完成；stock 侧数字为指令级实测上限（FP16 165.6 / FP8 332.8 TF/s）
> 与文献值的推算，未回 stock 重启（用户要求）。

## TL;DR

解锁后"GEMM 只快 12–34%"**不是硬件限制，而是软件栈按 stock 半速模型选 kernel 的假象**。

- **通行办法（cuBLASLt 用户）**：用 `cublaslt_autotune.hpp`（本目录下）替换裸 `cublasLtMatmul` ——
  首次调用对每个 (dtype, 形状, 布局) 枚举候选实测选优，之后走缓存。永远不比默认差，常见形状快 1.1–1.9×。
- **CUTLASS 用户**：FP16/BF16 用 `TB128x256x32 / warp64x64 / stages3`（≥16384 加 swizzle log2=4）；
  TF32 用 `TB128x128x32 / st3 / swz4`；**FP8 要先选精度路径**（见 5.2：精度安全用 cuBLAS 调优或 CUTLASS staged，
  极限速度用 CUTLASS fastacc 但 K 大时降精度）。
- **PyTorch 用户**：`torch.compile(mode="max-autotune")`（inductor 会做 GEMM autotune），
  或调用本 shim 的自定义算子。
- **不用折腾的场景**：小 M（访存瓶颈，如 decode M≤64）、FP16 累加路径（本来就没被锁）、INT8、纯 FP32。

## 1. 背景与问题

gpu-unlock 在驱动加载前通过固件事务把 SM 的 speed-select 寄存器（fmla32 等）从"半速"改成"全速"，
纯指令级实测（mma_bench，寄存器直供）：FP16/BF16(FP32 acc) 165.6→331.6 TF/s，FP8 332.8→660.6，
INT8/dp4a/FFMA/FP16 累加不变（对照）。

但 cuBLAS 默认只快 12–34%。假设：CUDA 库/cuBLAS 的启发式按半速硬件模型选 kernel，
导致解锁后 TC 单元被饿死。本报告验证该假设并给出工程解法。

## 2. 调查方法

- `src/cublas_scan` / `src/cublaslt_sweep`：枚举 cublasLt heuristic 全部候选，
  解码 algo 配置（tile/stages/splitK，来自 cublasLt 头文件枚举），逐一实测
- `src/cutlass_bench.cu`（fp16）、`src/cutlass_bf16_tf32.cu`：CUTLASS 2.x 风格 device API 自测 tile/stages/swizzle（头文件为 CUTLASS v4.2.1；"2.x" 指 API 风格而非库版本，原因见 5.2 节末注）
- `~/mma_smem_bench`：纯 PTX 的 SMEM 供数微基准（ldmatrix + mma，调节操作数复用度）
- SASS 反汇编 + 控制字解码（参考 github.com/cicuvc/nvidia-sass-document 的 CONTROL_CODES.md 与 sm_89 数据库）
- 测量纪律：同窗口交错对比（消除时钟漂移）；多卡并发扫描仅用于排名，绝对值以单卡复测为准

## 3. 根因分析（调查原因）

### 3.1 硬件确实能翻倍

mma_bench SASS：干净的 HMMA.16816 背靠背指令流（8 条独立累加链、无 wait mask、无访存），
实测 331.6 TF/s。→ 指令发射侧翻倍是真实可用的。

### 3.2 SMEM 供数不是瓶颈（复用度够时）

微基准（每条 mma 前用 ldmatrix 从 SMEM 取操作数，R=复用度）：

| R (B/mma) | 0 (纯寄存器) | 1 (768B) | 2 (384B) | 4 (192B) | 8 (96B) |
|---|---:|---:|---:|---:|---:|
| TF/s | 302.6 | 200.1 | 300.3 | 301.7 | 298.7 |

按 sass-document 的模型核算（Ada：SMEM 128B/clk/SM，解锁后 TC 4clk/inst）：warp tile ≥64×64
时每条 mma 的供数需求 ≤128B，仍在带宽内。**只要 kernel 的 warp tile 够大（复用够高），SMEM 喂得动全速 TC。**

### 3.3 cuBLASLt 的 heuristic 排名在解锁硬件上是错的

例：FP16 8192³，heuristic 第 0 名 = algoCfgId 6（tile 128x64）实测 182 TF/s；
而第 1 名 = 273.6 TF/s（+50%）。FP8 更夸张（见全景表，最高 +93%）。
heuristic 的成本模型按 stock 的 TC 速率校准，解锁后 tile 小、复用低的 kernel 排名虚高。

NCU 实锤（fp16/fp8 8192³，GPU7，Nsight Compute）：

| | kernel | 时长 | SM 利用率 | Mem 利用率 |
|---|---|---:|---:|---:|
| fp16 默认 pick | `ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_tn` | 5.96ms（≈184 TF/s） | 56% | 87% |
| fp16 库内最优 | `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn` | 3.83–3.93ms（≈280–287 TF/s） | 84.7% | 91% |
| fp8 默认 pick | `sm89_xmma_gemm_..._tilesize64x64x64_stage4_warpsize2x2x1_tensor16x8x32_...` | 3.70ms（≈297 TF/s） | 44.8% | 93.8% |
| fp8 库内最优 | `sm89_xmma_gemm_..._tilesize128x128x64_stage3_warpsize4x2x1_...` | 1.94ms（≈567 TF/s） | 86.1% | 90.0% |

同一个库里面就有快 50–100% 的 kernel，heuristic 只是没选中它们。

### 3.4 SASS 证据：错误配置的两种死法

- **供数/流水不足**：tile 过小的 kernel（如 cuBLAS 默认 pick 的 128x64）在 4clk/HMMA 下 TC 被饿死。
- **寄存器溢出**：CUTLASS 256×256×32（warp 128×64）→ 每线程需 256 个累加寄存器超过上限，
  SASS 中出现 **523 条 LDL/STL（local memory 溢出）**，实测仅 34 TF/s；好的配置 0 条溢出。
- 正确配置的主循环（CUTLASS 128×256×32/st3）：8 条 LDSM 打头（软件流水预取），
  之后 HMMA/LDSM 1:1 交错，HMMA 的 wait mask 精确挂在 LDSM 的 scoreboard 上（如 `wait=000010` 等 SB1/2/4），
  操作数带 `.reuse` 标记走 RF 复用 cache —— 完全按全速 TC 的节奏供数。

### 3.5 剩余的差距是时钟，不是功耗墙

单卡持续 GEMM：功耗仅 270–290W（450W 上限远未触及），SM 频率在 2.0–2.76GHz 间波动。
纯发射峰值 331.6 是在 ~2.75GHz 测的；按持续频率折算实际上限 ≈ 290–300 TF/s。
**调优后的 CUTLASS 已达到 293.7（8192³）—— 即实际天花板**。8 卡同跑 8192³ GEMM 每张卡 290–302 TF/s，不降速。

## 3.6 逆向：cublasLt heuristic 到底怎么选（以及别的卡会怎么选）

**结论：它是一个按 (设备型号族 × 问题描述) 查的静态模型，不含任何能感知"解锁"的运行时信号。**

证据：

1. **时钟盲**：同一张 4090 上 `nvidia-smi -lgc 1000`（锁 1GHz）vs 全 boost，heuristic 的 pick **逐字节相同**
   （`results/heur_4090_unlocked.txt` 两次 diff 为空）。→ 不读时钟。
2. **跨架构对比（含修正）**：同一探针在 RTX 5090（sm_120, 170 SMs, a separate 8x RTX 5090 lab host）上的 pick 完全不同 —
   全部落在 NVJET 系（cfg67，tile 256x128/128x256/160x128…，stages=64xAUTO）；
   而 4090（sm_89）落在保守的 ampere_* 系（cfg5/6/21/30，tile 64x128/128x64/128x128，sliced1x2）。
   **注意：5090 并不是原生全速** —— 实测（a separate 8x RTX 5090 lab host, mma_bench sm_120）f16/f32 = 224.2 TF/s ≈
   170SM×512FLOP/clk×boost，f16/f16 = 448.1 正好 2×，与 4090 的半速锁定完全同构。
   因此跨架构对比的正确读法不是"全速卡配大 tile"，而是：
   sm_89 的 ampere_* 老表对"翻倍后的 4090"失配（本机内的同库不同 kernel 对比才是硬证据）；
   而 NVIDIA 在 sm_120 的 NVJET 新栈里默认就是大 tile —— 说明"全速 TC 需要大 tile"这一点
   他们自己是清楚的，只是 sm_89 的表没有、也不可能有为解锁后状态准备的调参。
   推论：5090 若解锁翻倍，其 heuristic 大概率同样失准（该场景同样不在其校准集内）。
   （结果文件：本机 `results/heur_4090_unlocked.txt`，a separate 8x RTX 5090 lab host `~/heur_5090.txt`）
3. **二进制证据**：`libcublasLt.so.13` 里有完整的启发式 LUT 机制符号
   （`cublasLtHeuristicsCache{Create,Get,Put,SetCapacity,Destroy}`、`cublasLtHeuristicLutSerializeEntry`）
   和环境变量 `CUBLASLT_HEURISTICS_LUT_FILE`、`CUBLASLT_HEURISTICS_CACHE_CAPACITY`。
   这对应 NVIDIA 内部的 AOT 调优流程（离线生成 LUT，运行时加载）。但实测在当前 13.3 build 上：
   strace 全程未观察到对 LUT 文件的任何 open 调用，置空文件/开日志均无反应 ——
   **该机制存在但此版本未对用户激活**（需要内部工具生成有效 LUT 才可能加载）。
4. 它能看到什么：CC、SM 数、显存带宽等静态属性 + 问题描述（dtype/shape/layout/workspace）。
   解锁不改变其中任何一项 → **对它完全隐形**。

**关于"专业卡会不会选得更好"（RTX 5000 Ada / RTX PRO 5000 之问）**：

命名陷阱先澄清：RTX 5000 **Ada**（AD102）= sm_89 ✓；RTX **PRO** 5000（Blackwell, GB202）= sm_120。
Ada 工作站卡（RTX 5000/6000 Ada）的 TC 是全速的（RTX 6000 Ada FP16/FP32acc ≈ 364 TF/s），
理论上它的调参才匹配我们解锁后的 4090。但逆向证据不支持这条路：

- libcublasLt.so 中**没有任何设备名/PCI ID 字符串** —— 不存在按 SKU 细分的查找表；
  kernel 集合是按 CC（sm_89）共享的（ampere_* 家族覆盖所有 sm_89）
- 启发式能区分的只剩运行时属性（SM 数 128 vs 100/142、显存 BW、L2 大小）——即便 RTX 6000 Ada
  因此得到不同的 rank0，也只是同一套失准校准里的另一个局部选择，而不是"为全速 TC 准备的另一套表"
- 而且无论哪张卡，模型都看不见解锁（时钟盲已证）

所以对"借专业卡的 pick"的回答是：无表可借；5090 的 NVJET 又是 sm_120 专用二进制搬不过来。

对"能不能骗它选对 kernel"的评估（含 spoofing 实测）：
- 伪装设备属性（hook cudaGetDeviceProperties 改 SM 数）理论上会换查表键，但 5090 的 NVJET kernel
  是 sm_120 专用二进制，无法移植；同族 sm_89 内换 SM 数只是微调，治不了本。
- 官方/半官方的 LUT 注入在本版本不可用（见上）。

**LD_PRELOAD 伪装实验**（LD_PRELOAD/filter-lib 伪装实验（代码未随仓库发布，方法见正文描述））：把 4090 伪装成 RTX 6000 Ada
（SM 142、96MB L2、384bit/20Gbps 显存、PCI ID 0x26b1），逐层尝试了所有可拦截面：
(a) 全局符号拦截 `cuDeviceGetAttribute`/`cudaDeviceGetAttribute`/`cudaGetDeviceProperties`；
(b) dlsym/dlvsym 拦截（应对 dlopen+dlsym）；(c) ELF filter-library 伪造整个 libcuda.so.1。
结果：**pick 逐字节不变；且三个层面的日志都证实启发式查询路径上设备属性查询次数为 0**。
解读：cublasLt 自包含（无 libcuda/libcudart 动态依赖，静态内嵌），启发式要么只按 CC+问题描述查表，
要么走静态内嵌 cudart/驱动内部通道读设备 —— 总之对外部不可见、不可注入。
- **所以"运行时实测选优"（本目录的 shim）是当前唯一可靠的通行办法**，这也正好绕开了对
  heuristic 内部逻辑的任何依赖。

## 3.7 累加精度：A100 的"22 位截断"问题在 4090 上不存在

（用户记忆中的 A100 问题：Fasi/Higham 等对 Volta/Ampere TC 的逆向发现 fp32 累加器是截断式窄尾数。
我们在指令级 + GEMM 级两层实测了本机（解锁态 4090）和 a separate 8x RTX 5090 lab host（stock 5090）——两者结果一致。）

**指令级探针**（`src/precision_probe.cu` / `src/tie_probe.cu`，wmma→单条 HMMA）：

| 探针 | 4090 (解锁) | 5090 (stock) |
|---|---|---|
| 1+2^-q 定向：q≤23 精确 | ✅ | ✅ |
| q=24（正好是中点 tie） | 归为 1.0 | 归为 1.0 |
| RN-vs-RZ 裁决：1+3·2^-25（0.75 ulp） | **截断为 1.0 → RZ**（IEEE RN 应给 1+2^-23） | 同 4090 |
| 双 tie 1+2^-24+2^-24（检验加法树是否中途丢小项） | 精确 1+2^-23 ✓ | ✓ |

→ 结论：**4090/5090 的 fp32 累加器是"全 24 位对齐 + 向零舍入（RZ）"**，不是 A100 那种窄尾数截断。
RZ 相对 RN 的偏差是 NVIDIA TC 的一贯特性（公开文档亦有注明），单向 mma 内多乘积先经加法树合并、
最终写回时才做一次 RZ。解锁只动速率寄存器，不累及精度数据通路（两机结果逐位一致也间接印证）。

**GEMM 级 K 扫描**（`src/gemm_precision.cu`，M=N=256，随机 ±0.5 输入，对 CPU fp64 参考）：

| dtype | K=1024 p50 |K=4096 p50 | K=16384 p50 | 随 K 是否劣化 |
|---|---|---|---|---|
| fp16 (cuBLASLt) | 1.69e-4 | 2.82e-4 | 2.82e-4 | 否（典型 fp32 累加行为；max 项是近零消减元素的正常现象） |
| fp8e4m3 (cuBLASLt COMPUTE_32F) | 2.12e-4 | 2.16e-4 | 2.17e-4 | 否 —— **cuBLAS FP8 是精度安全的**（非截断路径） |

对照：CUTLASS FP8 `OpMultiplyAddFastAccum`（不分段）的误差随 K 增长（K=16384 时 10–15%），
证明**截断发生在 QMMA 的链式累加上**，CUTLASS staged / cuBLAS 的分段累加正是规避它的手段。

## 4. 实测全景表（cuBLASLt，默认 pick → 库内最优）

> 4 卡并发扫描（用于排名）；绝对值与单卡有 ±10% 时钟漂移。加粗 = 增益 >15%。

| shape | fp16 | bf16 | tf32 | fp8 |
|---|---|---|---|---|
| 64×4096×4096 | 108 → 117 (1.09x) | 113 → 113 (1.00x) | 69 → 69 (1.00x) | **136 → 159 (1.17x)** |
| 64×8192×8192 | 57 → 57 (1.00x) | 54 → 56 (1.04x) | 29 → 30 (1.03x) | **238 → 280 (1.17x)** |
| 256×4096×4096 | 181 → 188 (1.04x) | **164 → 211 (1.29x)** | **96 → 116 (1.21x)** | **267 → 350 (1.31x)** |
| 1024×4096×4096 | 204 → 204 (1.00x) | 242 → 242 (1.00x) | **141 → 162 (1.14x)** | **379 → 507 (1.34x)** |
| 1024×8192×8192 | 220 → 225 (1.02x) | 199 → 203 (1.02x) | **141 → 169 (1.19x)** | **402 → 528 (1.31x)** |
| 2048×2048×2048 | **185 → 217 (1.17x)** | 245 → 245 (1.00x) | 145 → 145 (1.00x) | **349 → 463 (1.32x)** |
| 2048×14336×4096 | **194 → 254 (1.31x)** | **204 → 335 (1.64x)** | 166 → 166 (1.00x) | **394 → 539 (1.37x)** |
| 4096×4096×4096 | 258 → 276 (1.07x) | **196 → 307 (1.57x)** | 156 → 156 (1.00x) | **375 → 524 (1.40x)** |
| 8192×8192×8192 | **182 → 275 (1.51x)** | 194 → 194 (1.00x) | 160 → 160 (1.00x) | **292 → 563 (1.93x)** |
| 8192×14336×4096 | 206 → 226 (1.09x) | **205 → 303 (1.48x)** | 166 → 166 (1.00x) | **316 → 587 (1.86x)** |
| 8192×4096×14336 | 184 → 195 (1.06x) | 185 → 196 (1.06x) | 154 → 154 (1.00x) | 481 → 481 (1.00x) |
| 16384×16384×16384 | 194 → 194 (1.00x) | 189 → 189 (1.00x) | 142 → 147 (1.03x) | **405 → 525 (1.30x)** |

规律：
- **FP8 几乎处处需要调优**（heuristic 对 FP8 kernel 的排名在解锁后完全失准）
- FP16/BF16 在大 K、大 N 的非对称形状上增益最大（LLM FFN 形状 1.3–1.9x）
- 小 M（≤64，decode 型）访存瓶颈，无关 TC 速率
- 16384³ 受 L2/DRAM 限制更多，需 swizzle（见 5.2）；cuBLAS 内部无对应大 swizzle 配置故无增益

## 5. 解决方案

### 5.1 通行办法：`cublaslt_autotune.hpp`（本目录，header-only）

设计：
- 与 `cublasLtMatmul` 同签名；按 (desc 属性 × 布局 × workspace) 作 key，首次调用实测选优，之后查表执行
- 候选来自 `cublasLtMatmulAlgoGetHeuristic`（同一合法集合，无越权行为）
- 交错计时（抗时钟漂移）+ 失败候选跳过 + **"不劣于默认"保护**（赢家须超过 heuristic rank0 >5% 才替换）
- 调优开销：每个新形状一次性 ~0.3–0.5s；之后零开销

实测（GPU7 单卡，交错对比）：

| dtype | shape | 默认 | 调优 | 提升 |
|---|---|---:|---:|---:|
| fp16 | 8192³ | 175–184 | 262–272 | **1.43–1.55x** |
| fp16 | 4096³ | 257 | 273–277 | 1.06x |
| fp16 | 16384³ | 194–196 | 194 | 1.00x（默认已对） |
| fp16 | 8192×14336×4096 | 186–194 | 251–271 | **1.30–1.45x** |
| fp16 | 64×4096×4096 | 97–100 | 111–121 | 1.13–1.24x |
| fp8 | 8192³ | 303–307 | 568–591 | **1.85–1.99x** |
| bf16 | 8192³ | 202–214 | 202–203 | ~1.00x（候选本就挤在一起） |
| tf32 | 8192³ | 155–175 | 152–160 | ~1.00x |

用法（最小示例）：

```cpp
#include "cublaslt_autotune.hpp"
static ltat::Autotuner tuner;   // 全局一个
// 把原来的 cublasLtMatmul(...) 替换为：
tuner.run(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, D, Dd, ws, wsBytes, stream);
```

注意：beta≠0 且 C==D（原地累加）时，调优会反复写 D —— 该场景请自行拷贝 D 或避免调优（已在代码注释说明）。

### 5.2 CUTLASS 直接用户：推荐配置（解锁硬件实测）

> 术语说明：下文 "2.x device API" 指 `cutlass/gemm/device/gemm.h` 的**设备级 API 风格**，
> 头文件本身是 CUTLASS v4.2.1。不用 3.x/4.x 的 CuTe collective builder 的原因：
> `collective/builders/` 只为 sm_90（wgmma+TMA）/sm_100/sm_120（tcgen05）提供 builder，
> **没有 sm_89 的 builder**；强行用 3.x 只会回落到同一套 sm80 风格 cp.async mainloop、
> 驱动同一个 HMMA/QMMA 原子，SASS 完全相同而模板参数旋钮反而更难调。
> （上 Hopper/Blackwell 时才需要 4.x collective。）


| dtype | 推荐配置 | 4096³ | 8192³ | 16384³ |
|---|---|---:|---:|---:|
| fp16 | TB128x256x32 / warp64x64 / st3（16384³ 加 swizzle log2=4） | ~308 | **293.7** | **276.7** |
| bf16 | 同上（TB128x256x32/st3/swz4） | 307.7 | 287.9 | **289.9** |
| tf32 | TB128x128x32/st3/swz4（或 128x256x32/st2/swz4；256 宽 tile 在 st3 下 SMEM 超 99KB 不可行） | 136.1 | **141.9** | **141.2** |
| fp8 (e4m3→fp16，**精度安全**) | staged TB128x128x64 / warp64x32 / st4 / swz4-5 | 501.4 | **505.4** | **507.0** |
| fp8 (e4m3→fp16，快但降精度) | fastacc TB128x256x64 / warp64x64 / st3 / swz4-5 | 581.4 | **591.2** | **597.4** |

要点：
- 指令形状：fp16/bf16 = m16n8k16，tf32 = m16n8k8，fp8 = m16n8k32；warp K = TBK
- **swizzle（L2 友好的 block 栅格化）在 ≥16384 的形状上是决定性的**（fp16 +24%，fp8 420.7→592.0 = +41%）
- 不要盲目加大 tile：256×256×32（warp128×64）寄存器溢出 → 34 TF/s（SASS 见 523 条 spill）
- TBK=64 + 256 宽 tile 的 SMEM 超过默认 48KB 需 opt-in（2.x device API 未处理）→ 直接用 TBK=32
- FP8 有两条路径，**精度权衡必须先选**（来自 `results/cutlass_fp8.md` 的逐 K 校验）：
  - `OpMultiplyAdd`（staged 分段累加）：真 fp32 累加精度（对 CPU fp64 参考 maxrel ~0.0007–0.003，不随 K 增长），
    但寄存器紧张，warp tile 必须 ≤64x32（默认 64x64 会触发 tmp_accum_ 双倍累加寄存器 → 溢出到 local memory，
    实测掉到 ~90 TF/s；SASS 确认纯 QMMA.16832、零 HMMA）
  - `OpMultiplyAddFastAccum`：跳过分段，更快（~90% 上限），但 **Ada 的 QMMA 链式累加会截断 fp32 累加器，
    K=16384 时相对误差达 10–15%** —— 只适合对精度不敏感的场景（如部分量化推理）
  - TBK=128/st2 的 staged 配置实测数值错误（maxrel 0.154）——**配置错了不一定报错，必须带数值校验**
- 精度安全的 FP8 对比：cuBLAS 库内最优（COMPUTE_32F，全真累加）8192³ = 587.6 仍是最佳；
  CUTLASS staged 在 16384³ 与之打平到略优（507.0 vs 493.2–525.1，随时钟漂移互有胜负）

FP8 明细（CUTLASS 2.x device API，e4m3 in / fp16 out，TN，指令 16x8x32 = QMMA）：

| 配置 | 模式 | 4096³ | 8192³ | 16384³ |
|---|---|---:|---:|---:|
| TB128x128x64 w64x32 st4 swz4-5 | staged（精度安全） | 501.4 | 505.4 | 507.0 |
| TB128x128x64 w32x32 st3 swz4 | staged | 473.2 | 468.6 | 470.0 |
| TB128x128x64 w64x64 st3 | staged → **寄存器溢出** | — | 85–90 | — |
| TB128x256x64 w64x64 st3 swz4/5/6 | fastacc（K 大时降精度） | 581.3 | 591.2 | 597.4 |
| cuBLAS 默认 pick（对照） | — | ~375 | ~292 | ~405 |

（"—" = 该单元格无意义（对照行）或该形状未测 —— 寄存器溢出已证实后其余形状无需再测。）
| cuBLAS 库内最优（对照） | — | 524.3 | 587.6 | 493.2–525.1 |

### 5.2.1 一般规律（从全部扫描数据归纳）

**一句话：解锁后 TC 发射翻倍（8→4 clk/inst），kernel 配置的任务从"喂饱半速 TC"变成"别让全速 TC 挨饿"，
同时撞两条硬约束：SMEM 供数带宽 与 寄存器文件上限。**

| 规律 | 内容 | 反例证据 |
|---|---|---|
| L1 | **warp tile = 64×64 是铁律**（fp32 累加路径）。它同时是：(a) 寄存器安全线 —— 64×64=128 个累加寄存器/线程，再大（128×64=256）必溢出；(b) SMEM 供数安全线 —— 按 sass-document 的模型，64×64 warp tile 每条 mma 需 ~128B，正好在 128B/clk/SM 的可持续范围内 | 256×256×32/warp128×64 → 523 条 LDL/STL 溢出 → **34 TF/s**；fp8 staged 用 64×64 warp → 双倍累加寄存器溢出 → 90 TF/s |
| L2 | **TB tile 128×256（或 256×128）+ K 深度 32**（fp16/bf16/tf32）或 64（fp8） | 128×128 全面略低；128×256 在 8192³ fp16 = 293.7 vs 128×128 的 237 |
| L3 | **stages = 3**（fp16/bf16），fp8 可 3–4，tf32 因 4 字节 tile 的 SMEM 压力退到 2–3（256 宽 tile 在 st3 下超 99KB 直接起不来） | 128×128×32/st5 比 st3 低（201 vs 237 @8192³） |
| L4 | **形状 ≥16384 必须开 swizzle（log2=4 起步）**：L2 命中率在大形状成为瓶颈 | fp16 16384³: 222→277（+24%）；fp8 staged: 222→507（+128%） |
| L5 | **形状 ≤4096 不必纠结**：大 tile 已经接近天花板，swizzle 无感（283 vs 284） | fp16 4096³ swz1 vs swz4 = 308.8 vs 308.5 |
| L6 | **fp8 先选精度路径再调 tile**：`OpMultiplyAdd`（staged，精度安全，~505）vs `OpMultiplyAddFastAccum`（~597 但 K=16k 时 10–15% 相对误差）。TBK=128 的 staged 配置会算错且不报错 | maxrel 0.154 vs 0.0007（check_k.log） |
| L7 | 我们这套方法对**任何"形状×精度"的未知组合都适用**：与其查表，不如让 shim/CUTLASS sweep 实测 —— 静态表只对"驱动 610.43.02 + CUDA 13.3 + 本机时钟行为"负责 | — |

**推荐配置总表（解锁后实测最优）：**

| dtype | 配置 | 2048³ | 4096³ | 8192³ | 16384³ | 效率 vs 纯发射上限 |
|---|---|---:|---:|---:|---:|---:|
| fp16 | 128×256×32 / w64×64 / st3（16384 加 swz4） | 285.0 | 308.8 | 293.7 | 276.7 | 86–93% |
| bf16 | 同上（大形状 swz4） | 284.4 | 307.7 | 287.9 | 289.9 | 86–93% |
| tf32 | 128×128×32 / w64×64 / st3 / swz4（2048³ 用 128×256×32/st2/swz4） | 136.8 | 136.1* | 141.9 | 141.2 | 83–86% |
| fp8 staged（精度安全） | 128×128×64 / w64×32 / st4 / swz4-5 | 470.2 | 501.4 | 505.4 | 507.0 | ~71–77% |
| fp8 fastacc（降精度） | 128×256×64 / w64×64 / st3 / swz4-6 | 533.3 | 581.4 | 591.2 | 597.4 | ~81–90% |

（2048³ 列为 2026-11-17 服务器恢复后补测：小形状上 swizzle 无感（规律 L5），各候选差异有限；
tf32 的 2048³ 与 4096³ 一样由 256 宽 st2 配置占优。）

*tf32 4096³ 的 128×256/st3 因 SMEM 超限不可用，表中为 st2/swz4 的值；warp K 必须等于 TBK。

### 5.3 PyTorch / 框架用户

- `torch.compile(mode="max-autotune")`（inductor 自带 GEMM autotune，等价于本 shim 的思路）
- 或 `torch._inductor.config.max_autotune_gemm = True`
- 裸 `torch.matmul` 走 cuBLASLt 默认 heuristic —— 在 FP8/特定形状上损失最大，建议换成自定义算子包本 shim

### 5.4 不需要折腾的场景

- M ≤ ~64 的 decode 型 GEMM（访存瓶颈）
- FP16 累加（fp16→fp16）路径：解锁不涉及（指令级 333 TF/s 本来就没锁）
- INT8 / dp4a / 纯 FP32 FFMA：同上，不受影响

## 6. 已知限制与风险

- stock 侧"基准"是指令级实测上限（165.6/332.8）的推算值，本机未回 stock 状态实测 GEMM（用户要求不重启回 stock）
- 全景表为 4 卡并发扫描（排名可靠）；绝对值以单卡为准，存在 ±10% 时钟漂移
- 调优结果绑定硬件状态（时钟/温度）；shim 为进程内缓存，跨进程需重新调优（可扩展为磁盘缓存）
- 所有数字基于：驱动 610.43.02（P2P 补丁版）+ CUDA 13.3 + gpu-unlock 1.1.1；升级驱动/CUDA 后应重新验证
- FP8 输出为 FP16（E4M3→FP32 累加→FP16）；若需 E4M3 输出/缩放因子，cublasLt 的 FP8 快速路径约束更多

## 7. 工具与复现

本仓库（`src/` 与 `results/`）：

- `src/cublaslt_autotune.hpp` —— 通行 shim；`src/autotune_demo.cu` —— 验证 demo
- `src/cublaslt_sweep.cu` —— cuBLASLt 全量扫描（algo 配置解码）；`src/cublas_scan.cu` —— 单形状 algo 枚举
- `src/cutlass_bench.cu`（fp16 tile/swizzle）、`src/cutlass_bf16_tf32.cu`、`src/cutlass_fp8_bench7.cu`（fp8 含数值校验）
- `src/mma_smem_bench.cu` —— SMEM 供数微基准；`src/precision_probe.cu` / `src/tie_probe.cu` —— 累加精度探针；
  `src/gemm_precision.cu` —— GEMM 级 K 扫描精度
- `src/heur_probe.cu` —— cuBLASLt heuristic 只查询探针；`src/gen_lt_names.py` —— 从 cublasLt.h 生成枚举名表
- `src/p2p_test.cu` / `src/h2d_test.cu` —— P2P / host 带宽
- `results/` —— 4 精度扫描 CSV 与 CUTLASS 结果笔记

编译示例见 [DEPLOYMENT.md](DEPLOYMENT.md) 第 3 节。CUTLASS v4.2.1 头文件需自备（`git clone` 后取 `include/`）。

SASS 解码：本调查使用了 [cicuvc/nvidia-sass-document](https://github.com/cicuvc/nvidia-sass-document)
的 `CONTROL_CODES.md` 与 sm_89 指令/延迟数据库作为控制字与调度字段的解读依据；先用
`cuobjdump -sass <bin>` 转储，再按该文档的位段说明解析即可。
