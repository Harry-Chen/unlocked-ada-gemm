# 部署指南 / Deployment Guide

> 中文版为主（与报告一致）。English summary: this guide reproduces an 8× RTX 4090
> machine with (1) PCIe P2P unlocked via a patched open-kernel-module driver and
> (2) full-rate FP16/BF16/FP8 tensor-core accumulation via gpu-unlock-ada,
> then installs this repo's GEMM toolkit.

## 0. 前置条件 / 适用范围

- Ubuntu 24.04 类系统，内核 6.8（其它版本需自行验证 DKMS 编译）
- RTX 4090（含 48G 改版；AD102, 10de:2684）
- 裸金属、Secure Boot 关闭、有重启权限
- 风险自负：这两个解锁都改变了 NVIDIA 的默认硬件行为（可能违反保修/许可条款；仅用于研究）

## 1. P2P 解锁（驱动）

驱动栈来自公开的 P2P 补丁 —— NVIDIA open kernel modules 的社区修改版
（本项目实测使用 [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)
@ `14de73d8`，它合并了 geohot/tinygrad 的原始 mod、nimlgen 的 5090 支持与 valdemardi 的 NVLink 支持）。

步骤概要：

1. 卸载旧驱动（runfile 用 `nvidia-uninstall`，apt 包用 `apt remove`）
2. 添加 NVIDIA CUDA 仓库（国内把 `developer.download.nvidia.com` 换成 `.cn` 加速）：
   `cuda-keyring_1.1-1_all.deb` → 修改 `/etc/apt/sources.list.d/cuda-*.list` 的域名为 `.cn`
3. 安装与补丁版本精确匹配的分支：我们用的是 `nvidia-driver-open=610.43.02-*` 系列
   （配 `nvidia-driver-pinning-610.43.02` 锁定）
4. 用打过补丁的 `nvidia-kernel-source-open` + `nvidia-dkms-open` 覆盖安装（自行从上游补丁构建；
   **本项目不分发任何 NVIDIA 二进制/deb/固件**），然后 `apt-mark hold` 这两个包
5. 内核参数需要 IOMMU 直通：`GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on iommu=pt"`
   （AMD 平台相应替换），`update-grub`，重启
6. 验证：`nvidia-smi topo -p2p r` 全 OK；用 `src/p2p_test.cu`（或 cuda-samples 的
   `p2pBandwidthLatencyTest`，注意它对 write 路径）测带宽

要点与排雷：
- 该补丁 README 强调 IOMMU 必须为 passthrough，否则 P2P DMA 会失败
- 若个别卡对带宽异常，先查 ACS（`lspci -vvv | grep ACSCtl`）—— 我们实测 ACS 开关对结果无影响，
  但不同主板可能不同
- `apt-mark hold` 不可省：否则 `apt upgrade` 会用官方同版本包覆盖掉补丁

## 2. Tensor Core 全速解锁（gpu-unlock-ada）

上游：[JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada)（GPL-2.0）。

它的 DKMS 模块在 initramfs 阶段、nvidia 驱动加载之前完成固件事务，把 SM 的
FP32 累加速率选择器从半速改为全速。我们的构建注意事项：

1. `make deb` 需要 SHA-pin 的 NVIDIA 610.43.03 输入：内核源码 tar 包 + 一个 `nvidia-utils` 包。
   上游 manifest 写的 CachyOS 存档链接已失效；我们用 Arch Linux Archive 的同名同内容包替换
   （`archive.archlinux.org/packages/n/nvidia-utils/nvidia-utils-610.43.03-1-x86_64.pkg.tar.zst`），
   其 `gsp_ga10x.bin` 的 SHA-256 与上游 pin 完全一致，只需同步更新 manifest /
   `firmware/generate_v67.py` 的 `PACKAGE_SHA256` / `Makefile` 的 `NV_UTILS` 文件名 /
   PKGBUILD 与 rpm spec 里的 pin。固件校验链会兜底（内容不符会构建失败，不会静默出错）。
2. 产出 `gpu-unlock-dkms` + `gpu-unlock-firmware` 两个 deb，安装后 DKMS 编译 +
   `update-initramfs` 自动把模块和固件放进 initramfs（先于 nvidia 加载）。
3. 激活：`GRUB_CMDLINE_LINUX_DEFAULT` 加 `gpu_unlock.targets=0000:xx:00.0`（按 `lspci -Dnn` 的
   总线地址；多卡逗号分隔）。官方建议多卡机先单卡验证再全开。
4. 首次激活最好是完全断电的冷启动；我们实测本机**热重启即可**（取决于主板固件状态保持情况）。
5. 验证：`journalctl -k -b | grep "gpu_unlock: unlocked"`；算力验证用上游的
   `mma_bench`（FP16/FP32 应从 ~166 到 ~331 TF/s）。

回退：删掉 cmdline 参数 + `update-grub` + 重启即恢复（模块无参数时完全惰性）。

## 3. 本仓库的 GEMM 工具链

前提：CUDA toolkit（我们用 13.3）+ CUTLASS 头文件（v4.2.1，`git clone` 后只需 `include/`）。

```bash
# 通行方案：cublasLt 自动调优 shim（header-only）
# 把 cublasLtMatmul(...) 换成 tuner.run(...) 即可，见 src/cublaslt_autotune.hpp 头注释
nvcc -O2 -arch=sm_89 -std=c++17 -I src src/autotune_demo.cu -o autotune_demo -lcublasLt

# cuBLASLt 全量扫描（默认 pick vs 库内最优，含 tile/stages/splitK 解码）
python3 src/gen_lt_names.py /usr/local/cuda/include/cublasLt.h lt_enum_names.h
nvcc -O2 -arch=sm_89 -I . -I src src/cublaslt_sweep.cu -o cublaslt_sweep -lcublasLt

# CUTLASS 配置扫描（需 -I /path/to/cutlass/include）
nvcc -O3 -arch=sm_89 -I /path/to/cutlass/include src/cutlass_bench.cu -o cutlass_bench

# 精度探针
nvcc -O2 -arch=sm_89 -std=c++17 src/precision_probe.cu -o precision_probe
nvcc -O2 -arch=sm_89 -std=c++17 src/tie_probe.cu -o tie_probe
```

更多细节见 `docs/REPORT.zh.md`（调查过程、全部数据表、SASS 分析、一般规律 L1–L7）。

## 4. 免责与致谢

- 本项目与 NVIDIA 无关；解锁操作的风险与许可合规由使用者自负
- 上游：gpu-unlock-ada（tensor 解锁）、aikitoria/open-gpu-kernel-modules（P2P 补丁，基于
  geohot/tinygrad、nimlgen、valdemardi 的工作）、cicuvc/nvidia-sass-document（SASS 参考资料）、
  NVIDIA CUTLASS（Apache-2.0）
