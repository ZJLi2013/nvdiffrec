# nvdiffrec (ROCm)

> **Disclaimer**: This repository is a personal fork of [NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec)
> with modifications for AMD ROCm GPU compatibility.
> It is intended **solely for non-commercial academic research and personal study purposes**.
> The original code is licensed under the
> [NVIDIA Source Code License (NSCL)](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt),
> which **prohibits commercial use** without separate authorization from NVIDIA.
> All intellectual property rights remain with NVIDIA Corporation and its affiliates.
> If you wish to use nvdiffrec for commercial purposes, please contact
> [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

![Teaser image](https://nvlabs.github.io/nvdiffrec/assets/system.JPG "Teaser image")

Joint optimization of topology, materials and lighting from multi-view image observations
as described in the paper
[Extracting Triangular 3D Models, Materials, and Lighting From Images](https://nvlabs.github.io/nvdiffrec/).

This `rocm` branch adds support for running nvdiffrec on **AMD GPUs** with ROCm.

**Tested on**: MI300X (gfx942), ROCm 7.1.1, PyTorch 2.9.1

## Installation

Requires Linux, ROCm 6.x+, and PyTorch 2.x with ROCm support.

### Docker (recommended)

```bash
# Start a ROCm PyTorch container
docker run -d --name nvdiffrec-rocm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --shm-size=16g \
  -v /data:/data -w /data \
  rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 sleep infinity

# Enter the container
docker exec -it nvdiffrec-rocm bash

# Clone and install
git clone -b rocm https://github.com/ZJLi2013/nvdiffrec.git && cd nvdiffrec
pip install imageio trimesh tqdm matplotlib ninja xatlas numpy opencv-python-headless
pip install --no-build-isolation git+https://github.com/ZJLi2013/nvdiffrast.git@rocm
```

### Optional: tiny-rocm-nn (for faster hash grid encoding)

The pure PyTorch hash grid fallback works out of the box. For better performance, install
[tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) (currently requires ROCm 6.x):

```bash
git clone https://github.com/ZJLi2013/tiny-rocm-nn.git
cd tiny-rocm-nn && git submodule update --init --recursive
cd bindings/torch && pip install --no-build-isolation -e .
```

## Examples

Simple genus-1 reconstruction (bob):

```bash
PYTHONUNBUFFERED=1 python train.py --config configs/bob.json
```

Training output on MI300X (gfx942):

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

Included configs:

- `bob.json` — Simple genus-1 model
- `spot.json` — Geometry, materials, and lighting from image observations
- `spot_fixlight.json` — Same as above with fixed environment lighting
- `spot_metal.json` — Joint learning of materials and high-frequency environment lighting

Results are stored in the `out/` folder.

## What Changed for ROCm

| Component | Strategy |
|-----------|----------|
| `nvdiffrast` | ROCm fork ([ZJLi2013/nvdiffrast@rocm](https://github.com/ZJLi2013/nvdiffrast/tree/rocm)), uses `RasterizeCudaContext` instead of OpenGL |
| `renderutils` CUDA kernels | Auto-detected Python fallbacks via `torch.version.hip` |
| `tiny-cuda-nn` (hash grid) | Pure PyTorch `_HashGridEncoding` fallback (or install [tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn)) |
| Cubemap filtering | Pure PyTorch `cubemap_python.py` fallback |

## Known Limitations

- `tiny-rocm-nn` requires ROCm 6.x due to `hipblasDatatype_t` API change in 7.x; the PyTorch fallback is used on 7.x
- Cubemap pre-filtering uses approximate box-blur (slightly less accurate specular reflections)
- All `renderutils` operations use Python fallbacks (functional but slower than native CUDA kernels)
- Windows is not supported on this branch

See [docs/rocm_migration.md](docs/rocm_migration.md) for the full migration analysis and experiment log.

## Datasets

Configs `nerf_*.json` and `nerd_*.json` require third-party datasets. To download:

```bash
cd data
python download_datasets.py
```

See the [upstream README](https://github.com/NVlabs/nvdiffrec#datasets) for manual download instructions.

## Citation

```bibtex
@inproceedings{Munkberg_2022_CVPR,
    author    = {Munkberg, Jacob and Hasselgren, Jon and Shen, Tianchang and Gao, Jun and Chen, Wenzheng
                    and Evans, Alex and M\"uller, Thomas and Fidler, Sanja},
    title     = "{Extracting Triangular 3D Models, Materials, and Lighting From Images}",
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {8280-8290}
}
```

## License

Copyright &copy; 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the [NVIDIA Source Code License](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt).
For business inquiries, please visit [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).
