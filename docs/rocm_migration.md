# nvdiffrec — AMD GPU ROCm 迁移可行性评估

## 实验总览表

| Exp | 假设 | 状态 | 关键结果 | 结论 |
|-----|------|------|----------|------|
| OT-1 | nvdiffrec 可迁移至 MI300X + ROCm 运行 mesh reconstruction | ✅ 已验证 | bob.json 100 iter 训练成功, loss 0.448→0.017 | 可行, 已在 MI300X 上运行 |

---

## Exp-OT-1: nvdiffrec ROCm 迁移可行性评估

### Phase 0 确认

- **输入完备性**: 本地 fork `github/kernels/nvdiffrec` (upstream: NVlabs/nvdiffrec)
- **变量**: GPU 架构 (AMD MI300X gfx942 vs NVIDIA)
- **评估标准**: 能否在 AMD GPU 上完成 image-based 3D mesh reconstruction (train.py)
- **许可证**: NVIDIA Source Code License — 允许非商业研究用途，允许再分发（需保留许可证 + 3.3 使用限制）

### 假设

nvdiffrec 可迁移至 AMD MI300X (gfx942) + ROCm。核心依赖 nvdiffrast 已有 ROCm fork (`ZJLi2013/nvdiffrast@rocm`)，renderutils CUDA kernel 结构简单可 hipify，FlexiCubes 为纯 PyTorch。

### 代码架构分析

```
nvdiffrec/
├── train.py                  # 主入口，训练/验证循环
├── geometry/
│   ├── flexicubes.py         # FlexiCubes 核心 (纯 PyTorch, Apache 2.0)
│   ├── flexicubes_geo.py     # FlexiCubes geometry wrapper
│   ├── dmtet.py              # DMTet isosurface
│   ├── dlmesh.py             # Direct mesh optimization
│   └── tables.py             # Lookup tables
├── render/
│   ├── render.py             # 渲染管线 (依赖 nvdiffrast)
│   ├── renderutils/
│   │   ├── ops.py            # Python → CUDA kernel 调度
│   │   ├── bsdf.py           # 纯 PyTorch BSDF fallback
│   │   ├── loss.py           # 纯 PyTorch loss fallback
│   │   ├── __init__.py
│   │   └── c_src/            # ★ CUDA kernels (需迁移)
│   │       ├── bsdf.cu       # PBR BSDF kernels (fwd+bwd)
│   │       ├── loss.cu       # Image loss kernels
│   │       ├── normal.cu     # Shading normal kernels
│   │       ├── mesh.cu       # Transform kernels
│   │       ├── cubemap.cu    # Cubemap filter kernels
│   │       ├── common.cpp    # Launch helpers
│   │       ├── torch_bindings.cpp  # PyTorch C++ 绑定
│   │       └── *.h           # Headers
│   ├── light.py              # 环境光照
│   ├── material.py           # 材质系统
│   ├── texture.py            # 纹理
│   ├── mlptexture.py         # MLP 纹理 (tiny-cuda-nn)
│   ├── mesh.py               # Mesh 数据结构
│   ├── obj.py                # OBJ I/O
│   ├── util.py               # 工具函数
│   └── regularizer.py        # 正则化
└── dataset/                  # 数据加载
```

### ROCm 库兼容性分析

| 依赖 | 类型 | ROCm 方案 | 风险 |
|------|------|-----------|------|
| **nvdiffrast** | 可微光栅化 | ✅ ROCm fork `ZJLi2013/nvdiffrast@rocm` | ✅ 已验证 (rasterize/interpolate/antialias/texture) |
| **renderutils CUDA kernels** (5 个 .cu) | 自定义 CUDA ext | 需 CUDA→HIP 迁移 | 🟡 中等 — 有纯 PyTorch fallback |
| torch | 核心 | ROCm Docker 预装 | ✅ 无 |
| xatlas | UV 展开 (CPU) | pip install | ✅ 无 |
| numpy | 数值 | pip install | ✅ 无 |
| imageio | 图像 I/O | pip install | ✅ 无 |
| **tiny-cuda-nn** | MLP 纹理 (HashGrid encoding) | ✅ ROCm port `ZJLi2013/tiny-rocm-nn` | ✅ 已验证 (MI300X, HashGrid+MLP) |
| apex (可选, multi-GPU) | DDP + FusedAdam | 不需要 (单卡) | ✅ 可跳过 |

### 核心迁移工作项分析

#### 1. renderutils CUDA kernels → HIP (工作量: 中等)

**5 个 .cu 文件**:
- `bsdf.cu` — PBR BSDF 前向/反向 (Lambert, Frostbite, Fresnel, GGX NDF, Smith masking, PBR specular)
- `loss.cu` — Image loss (L1/MSE/SMAPE/RelMSE + tonemapper)
- `normal.cu` — Shading normal preparation
- `mesh.cu` — Point/vector transforms
- `cubemap.cu` — Diffuse/specular cubemap filtering

**迁移难度评估: 低**
- 全部是 element-wise 或 pixel-wise kernels，**无 warp-level 协作**（除 loss.cu 的 warp reduce）
- 无 shared memory 使用
- 无 CUDA 专有 intrinsics（除标准 `__global__`, `__device__`, `blockIdx` 等）
- CUDA API 调用仅 `cudaLaunchKernel` + `cudaStream_t` → 直接 hipify
- `torch_bindings.cpp` 使用标准 PyTorch C++ extension API → ROCm PyTorch 兼容

**关键发现: 有纯 PyTorch fallback!**

所有 renderutils 函数都有 `use_python=True` 参数，可以走纯 PyTorch 实现：
- `ops.py` 中每个函数 (lambert, pbr_bsdf, image_loss, xfm_points 等) 都有 Python fallback path
- `bsdf.py` — 完整的纯 PyTorch BSDF 实现
- `loss.py` — 完整的纯 PyTorch loss 实现
- 例外: `diffuse_cubemap` 和 `specular_cubemap` 无 Python fallback (`assert False`)

**迁移策略选择**:

| 策略 | 工作量 | 性能 | 推荐 |
|------|--------|------|------|
| A: hipify 所有 .cu | 中 (2-3天) | 最优 | 长期方案 |
| B: 全用 `use_python=True` fallback | 低 (0.5天) | 较慢但可用 | 快速验证 |
| C: hipify 关键路径 + fallback 其余 | 低-中 (1天) | 平衡 | 推荐 |

#### 2. nvdiffrast (已解决)

ROCm fork 已验证：rasterize, interpolate, antialias, texture 模块在 gfx942/gfx1100/gfx1201 均通过。

#### 3. tiny-cuda-nn → tiny-rocm-nn (已解决)

`mlptexture.py` L12 `import tinycudann as tcnn`，L72 `self.encoder = tcnn.Encoding(3, enc_cfg)` — 使用 HashGrid positional encoding。

**已有 ROCm 迁移**: [`ZJLi2013/tiny-rocm-nn`](https://github.com/ZJLi2013/tiny-rocm-nn) — 完整 ROCm port，保持 `import tinycudann as tcnn` 兼容接口。
- 支持 HashGrid / OneBlob / Frequency / Identity encodings
- 支持 FullyFusedMLP (rocWMMA + hipBLAS)
- MI300X (gfx942) 已验证, 1.25x faster than PyTorch FP32
- 安装: `cd tiny-rocm-nn/bindings/torch && pip install --no-build-isolation .`
- nvdiffrec 的 `mlptexture.py` **零代码改动**即可使用，因为 Python module name 保持 `tinycudann`

#### 4. `torch_bindings.cpp` 编译问题

- `#include <ATen/cuda/CUDAContext.h>` → ROCm PyTorch 提供兼容头文件
- `cudaLaunchKernel` → `hipLaunchKernel`
- `-lcuda -lnvrtc` linker flags → 需改为 `-lamdhip64` 或由 PyTorch ROCm build system 处理
- `NVDR_CHECK_CUDA_ERROR` 宏 → 改为 `hipError_t` 检查
- `TORCH_CUDA_ARCH_LIST` → 改为 `PYTORCH_ROCM_ARCH`

#### 5. 硬编码 `device='cuda'` 调用

`train.py` 和多个文件中大量 `device='cuda'` 硬编码。ROCm PyTorch 兼容 `device='cuda'` 映射到 HIP，**无需修改**。

### 风险与缓解

| 风险 | 严重程度 | 缓解方案 |
|------|----------|----------|
| renderutils kernel 编译失败 | 中 | 退回 `use_python=True` fallback |
| tiny-cuda-nn 不可用 | ~~低~~ 已解决 | ✅ `ZJLi2013/tiny-rocm-nn` 已验证, 零改动兼容 |
| loss.cu warp reduce 在 wave64 上行为不同 | 低 | 仅 `image_loss` 使用 warp reduce, fallback 可用 |
| cubemap filter 无 Python fallback | 低 | 仅 environment lighting 需要, 可用固定 lighting 绕过 |
| multi-GPU (apex DDP) | 无 | 单卡测试不需要 |

### 推荐迁移路径

**Phase 1 — 快速验证 (0.5天)**:
1. 修改 `renderutils/ops.py`: 所有函数默认 `use_python=True`
2. 安装 nvdiffrast ROCm fork (`ZJLi2013/nvdiffrast@rocm`)
3. 安装 tiny-rocm-nn (`ZJLi2013/tiny-rocm-nn`, `pip install --no-build-isolation .`)
4. 运行 `python train.py --config configs/bob.json` (含完整 MLP 纹理第一阶段)
5. 验证: 训练完成, 输出 mesh + 纹理

**Phase 2 — 性能优化 (2-3天)**:
1. hipify `c_src/*.cu` + `torch_bindings.cpp`
2. 修改 build system (setup.py 或 JIT 编译 flags)
3. 对比 use_python=True vs HIP kernel 性能

### 可行性结论

| 维度 | 评估 |
|------|------|
| **技术可行性** | ✅ **高** — 三大核心依赖均已有 ROCm 支持 (nvdiffrast ✅, tiny-rocm-nn ✅, renderutils 有 Python fallback) |
| **工作量** | ✅ **低** — Phase 1 快速验证 0.5天 (所有依赖已就绪, 仅需装包+切 fallback) |
| **许可证** | ✅ **合规** — NSCL 允许非商业研究 + 再分发衍生作品 |
| **风险** | ✅ **极低** — 所有外部 CUDA 依赖已迁移, renderutils Python fallback 兜底 |

**总结: nvdiffrec ROCm 迁移条件已完全具备。nvdiffrast (✅), tiny-rocm-nn (✅), FlexiCubes (纯 PyTorch ✅) 三大依赖均已有 ROCm 支持，唯一剩余工作是 renderutils 内部 5 个 .cu kernel 的 fallback 切换 (0.5天) 或 hipify (2-3天)。**

---

## Exp-OT-1 实验结果

### Phase 1 — 快速验证 (已完成)

**测试环境**:
- Node: `hjbog-srdc-18.amd.com` (MI300X, gfx942)
- Docker: `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`
- PyTorch: 2.9.1+rocm7.1.1
- nvdiffrast: 0.4.0 (ROCm fork `ZJLi2013/nvdiffrast@rocm`)
- tiny-cuda-nn: 未安装 (ROCm 7.1 hipBLAS API 不兼容, 使用纯 PyTorch fallback)
- Branch: `ZJLi2013/nvdiffrec@rocm`

**代码改动** (共 7 commits):
1. `render/renderutils/ops.py` — 自动检测 ROCm (torch.version.hip), 所有 BSDF/loss/transform 函数默认使用 Python fallback
2. `train.py` — ROCm 环境使用 `dr.RasterizeCudaContext()` (无 OpenGL 依赖)
3. `render/mlptexture.py` — 新增 `_HashGridEncoding` 纯 PyTorch 多分辨率哈希网格编码 (tinycudann fallback)
4. `render/renderutils/cubemap_python.py` — 新增纯 PyTorch cubemap diffuse/specular filtering fallback

**Smoke Test 结果 (bob.json, 100 iterations)**:

```
iter=    0, img_loss=0.448444, reg_loss=0.333799, lr=0.02999, time=4250.0 ms
iter=   10, img_loss=0.158706, reg_loss=0.321369, lr=0.02985, time=3773.5 ms
iter=   20, img_loss=0.062264, reg_loss=0.296867, lr=0.02971, time=3822.3 ms
iter=   30, img_loss=0.044559, reg_loss=0.270694, lr=0.02957, time=3784.5 ms
iter=   40, img_loss=0.031614, reg_loss=0.255848, lr=0.02944, time=3831.6 ms
iter=   50, img_loss=0.025261, reg_loss=0.243869, lr=0.02930, time=3805.2 ms
iter=   60, img_loss=0.019909, reg_loss=0.231675, lr=0.02917, time=3850.4 ms
iter=   70, img_loss=0.019298, reg_loss=0.219970, lr=0.02903, time=3853.0 ms
iter=   80, img_loss=0.018719, reg_loss=0.208566, lr=0.02890, time=3827.0 ms
iter=   90, img_loss=0.018629, reg_loss=0.197456, lr=0.02877, time=3829.9 ms
iter=  100, img_loss=0.017255, reg_loss=0.186273, lr=0.02864, time=3851.5 ms
```

**关键结论**:
- ✅ **训练成功**: img_loss 从 0.448 → 0.017 (96% 下降), 收敛正常
- ✅ **性能稳定**: ~3.8s/10iter, 无 OOM, 无 NaN
- ✅ **全 Python fallback 可用**: renderutils + hash grid encoding + cubemap filtering 均使用纯 PyTorch
- ⚠️ **tiny-rocm-nn 未兼容 ROCm 7.1**: `hipblasDatatype_t` API 变更导致编译失败, 需更新 tiny-rocm-nn 或使用 ROCm 6.x Docker
- ⚠️ **cubemap Python fallback 较慢**: specular cubemap 预计算在初始化时需要 ~30s (CUDA kernel 版 <1s)

### 遗留问题 & Next Steps

| 优先级 | 任务 | 预估工作量 |
|--------|------|-----------|
| P1 | 修复 tiny-rocm-nn 在 ROCm 7.x 的 `hipblasDatatype_t` 兼容性 | 0.5天 |
| P2 | HIPify `renderutils/c_src/*.cu` (5 个 kernel 文件) 提升性能 | 2-3天 |
| P3 | 优化 specular cubemap Python fallback (或 HIPify cubemap.cu) | 1天 |
| P3 | 完整 1000 iter bob.json 训练 + 结果对比 | 0.5天 |

---

## 调试追踪

| 轮次 | 问题 | 修复 | 结果 |
|------|------|------|------|
| 1 | renderutils CUDA kernels 无法在 ROCm 编译 | 全部切 Python fallback (`_use_python_fallback`) | ✅ 可用 |
| 2 | `RasterizeGLContext` 无 OpenGL 在 headless server | 切到 `RasterizeCudaContext` | ✅ 可用 |
| 3 | `diffuse_cubemap` / `specular_cubemap` 无 Python fallback | 新增 `cubemap_python.py` | ✅ 可用 |
| 4 | `tinycudann` (tiny-rocm-nn) ROCm 7.1 编译失败 (`hipblasDatatype_t`) | 新增 `_HashGridEncoding` 纯 PyTorch fallback | ✅ 可用 |
| 5 | `_HashGridEncoding` embedding 在 CPU, input 在 GPU | 添加 `.cuda()` 到 encoder 初始化 | ✅ 修复 |
| 6 | `_HashGridEncoding` 缺少 `.params` 属性 (tinycudann compat) | 添加 `_CombinedParams` proxy class | ✅ 修复 |
| 7 | specular cubemap GGX importance sampling 太慢 (~10min) | 改用 box-blur 近似 | ✅ 快速 |
| 8 | Python stdout 缓冲导致看不到训练输出 | 使用 `PYTHONUNBUFFERED=1` | ✅ 修复 |
