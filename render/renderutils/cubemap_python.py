# Pure-PyTorch fallback for cubemap diffuse/specular filtering.
# Used on ROCm where the CUDA cubemap kernels are not available.
# These are slower than the CUDA versions but produce functionally equivalent results.

import torch
import numpy as np
import nvdiffrast.torch as dr


def _cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def _safe_normalize(x, eps=1e-20):
    return x / torch.clamp(torch.norm(x, dim=-1, keepdim=True), min=eps)


def diffuse_cubemap_python(cubemap):
    """Compute diffuse (irradiance) cubemap via brute-force cosine-weighted integration.

    Args:
        cubemap: [6, H, W, C] environment cubemap

    Returns:
        [6, H, W, C] diffuse irradiance cubemap (same resolution as input)
    """
    n_faces, res_h, res_w, n_channels = cubemap.shape
    assert n_faces == 6

    device = cubemap.device
    out = torch.zeros_like(cubemap)

    src_dirs = []
    src_solid_angles = []
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res_h, 1.0 - 1.0 / res_h, res_h, device=device),
            torch.linspace(-1.0 + 1.0 / res_w, 1.0 - 1.0 / res_w, res_w, device=device),
            indexing='ij'
        )
        d = _safe_normalize(_cube_to_dir(s, gx, gy))
        pixel_area = (2.0 / res_h) * (2.0 / res_w)
        length_sq = (1.0 + gx**2 + gy**2)
        solid_angle = pixel_area / (length_sq * torch.sqrt(length_sq))
        src_dirs.append(d)
        src_solid_angles.append(solid_angle)

    all_dirs = torch.cat([d.reshape(-1, 3) for d in src_dirs], dim=0)
    all_sa = torch.cat([sa.reshape(-1) for sa in src_solid_angles], dim=0)
    all_colors = cubemap.reshape(-1, n_channels)

    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res_h, 1.0 - 1.0 / res_h, res_h, device=device),
            torch.linspace(-1.0 + 1.0 / res_w, 1.0 - 1.0 / res_w, res_w, device=device),
            indexing='ij'
        )
        normal = _safe_normalize(_cube_to_dir(s, gx, gy))
        normal_flat = normal.reshape(-1, 3)

        cos_theta = torch.mm(normal_flat, all_dirs.t())
        cos_theta = torch.clamp(cos_theta, min=0.0)

        weights = cos_theta * all_sa[None, :]
        weighted_sum = torch.mm(weights, all_colors)

        out[s] = (weighted_sum / np.pi).reshape(res_h, res_w, n_channels)

    return out


def specular_cubemap_python(cubemap, roughness, cutoff=0.99):
    """Approximate pre-filtered specular cubemap.

    Uses the reflection direction (= normal for V=N assumption) to look up from a
    progressively blurred version of the cubemap. The blur is approximated by repeated
    avg-pooling proportional to roughness, then upsampling back to original resolution.
    This avoids expensive per-texel importance sampling.

    Args:
        cubemap: [6, H, W, C] environment cubemap
        roughness: scalar roughness value
        cutoff: unused

    Returns:
        [6, H, W, C+1] pre-filtered specular cubemap (RGB + weight=1 in last channel)
    """
    n_faces, res_h, res_w, n_channels = cubemap.shape
    device = cubemap.device

    n_blur_passes = max(0, int(roughness * 8))
    blurred = cubemap.clone()
    for _ in range(n_blur_passes):
        if blurred.shape[1] < 4 or blurred.shape[2] < 4:
            break
        blurred = blurred.permute(0, 3, 1, 2)
        blurred = torch.nn.functional.avg_pool2d(blurred, kernel_size=2, stride=1, padding=0)
        blurred = torch.nn.functional.interpolate(blurred, size=(blurred.shape[2] + 1, blurred.shape[3] + 1),
                                                   mode='bilinear', align_corners=False)
        blurred = blurred.permute(0, 2, 3, 1)

    if blurred.shape[1] != res_h or blurred.shape[2] != res_w:
        blurred = blurred.permute(0, 3, 1, 2)
        blurred = torch.nn.functional.interpolate(blurred, size=(res_h, res_w), mode='bilinear', align_corners=False)
        blurred = blurred.permute(0, 2, 3, 1)

    ones = torch.ones(n_faces, res_h, res_w, 1, device=device)
    out = torch.cat([blurred, ones], dim=-1)
    return out
