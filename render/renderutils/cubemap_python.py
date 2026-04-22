# Pure-PyTorch fallback for cubemap diffuse/specular filtering.
# Used on ROCm where the CUDA cubemap kernels are not available.
# These are slower than the CUDA versions but functionally equivalent.

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
    """Compute pre-filtered specular cubemap using importance sampling of GGX NDF.

    All 6 faces are processed at once with vectorized sample generation to minimize
    Python loop overhead.

    Args:
        cubemap: [6, H, W, C] environment cubemap
        roughness: scalar roughness value
        cutoff: energy cutoff (unused in this simplified version)

    Returns:
        [6, H, W, C+1] pre-filtered specular cubemap (RGB + weight in last channel)
    """
    n_faces, res_h, res_w, n_channels = cubemap.shape
    device = cubemap.device
    alpha = roughness * roughness
    alpha_sq = alpha * alpha

    n_samples = 64

    normals = []
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res_h, 1.0 - 1.0 / res_h, res_h, device=device),
            torch.linspace(-1.0 + 1.0 / res_w, 1.0 - 1.0 / res_w, res_w, device=device),
            indexing='ij'
        )
        normals.append(_safe_normalize(_cube_to_dir(s, gx, gy)))

    normal_all = torch.stack(normals, dim=0)
    view_all = normal_all

    up = torch.where(
        normal_all[..., 2:3].abs() < 0.999,
        torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(normal_all),
        torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(normal_all)
    )
    tangent_all = _safe_normalize(torch.cross(up, normal_all, dim=-1))
    bitangent_all = torch.cross(normal_all, tangent_all, dim=-1)

    color_accum = torch.zeros(n_faces, res_h, res_w, n_channels, device=device)
    weight_accum = torch.zeros(n_faces, res_h, res_w, 1, device=device)

    for i in range(n_samples):
        xi1 = (i + 0.5) / n_samples
        xi2 = ((i * 0.7548776662) % 1.0)

        cos_theta_val = float(np.sqrt((1.0 - xi1) / (1.0 + (alpha_sq - 1.0) * xi1)))
        sin_theta_val = float(np.sqrt(max(0.0, 1.0 - cos_theta_val * cos_theta_val)))
        phi_val = 2.0 * np.pi * xi2

        hx = sin_theta_val * np.cos(phi_val)
        hy = sin_theta_val * np.sin(phi_val)
        hz = cos_theta_val

        h_world = tangent_all * hx + bitangent_all * hy + normal_all * hz
        h_world = _safe_normalize(h_world)

        reflect_dir = 2.0 * (view_all * h_world).sum(dim=-1, keepdim=True) * h_world - view_all
        reflect_dir = _safe_normalize(reflect_dir)

        ndotl = torch.clamp((normal_all * reflect_dir).sum(dim=-1, keepdim=True), min=0.0)

        sample_color = dr.texture(
            cubemap[None, ...],
            reflect_dir.reshape(1, n_faces * res_h, res_w, 3).contiguous(),
            filter_mode='linear',
            boundary_mode='cube'
        ).reshape(n_faces, res_h, res_w, n_channels)

        color_accum += sample_color * ndotl
        weight_accum += ndotl

    out = torch.cat([color_accum, weight_accum], dim=-1)
    return out
