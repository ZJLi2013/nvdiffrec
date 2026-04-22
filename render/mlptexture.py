# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import numpy as np

try:
    import tinycudann as tcnn
    _has_tcnn = True
except ImportError:
    _has_tcnn = False


class _HashGridEncoding(torch.nn.Module):
    """Pure-PyTorch multi-resolution hash grid encoding (fallback when tinycudann is unavailable)."""
    def __init__(self, n_input_dims, cfg):
        super().__init__()
        self.n_levels = cfg["n_levels"]
        self.n_features_per_level = cfg["n_features_per_level"]
        self.log2_hashmap_size = cfg["log2_hashmap_size"]
        self.base_resolution = cfg["base_resolution"]
        self.per_level_scale = cfg["per_level_scale"]
        self.n_input_dims = n_input_dims
        self.n_output_dims = self.n_levels * self.n_features_per_level
        self.hashmap_size = 2 ** self.log2_hashmap_size

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(self.hashmap_size, self.n_features_per_level)
            for _ in range(self.n_levels)
        ])
        for emb in self.embeddings:
            torch.nn.init.uniform_(emb.weight, -1e-4, 1e-4)

        primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.int64)
        self.register_buffer("primes", primes)

    def _hash_coords(self, coords_floor, level):
        hashed = torch.zeros(coords_floor.shape[0], dtype=torch.int64, device=coords_floor.device)
        for d in range(self.n_input_dims):
            hashed = hashed ^ (coords_floor[:, d].long() * self.primes[d])
        return hashed % self.hashmap_size

    def forward(self, x):
        outputs = []
        for level, emb in enumerate(self.embeddings):
            resolution = int(self.base_resolution * (self.per_level_scale ** level))
            scaled = x * resolution
            floor_coords = scaled.floor().long()

            n_verts = 2 ** self.n_input_dims
            interp = torch.zeros(x.shape[0], self.n_features_per_level, device=x.device)
            frac = scaled - floor_coords.float()

            for i in range(n_verts):
                offset = torch.tensor([(i >> d) & 1 for d in range(self.n_input_dims)],
                                      device=x.device, dtype=torch.long)
                corner = floor_coords + offset[None, :]
                idx = self._hash_coords(corner, level)
                weight = torch.ones(x.shape[0], device=x.device)
                for d in range(self.n_input_dims):
                    weight = weight * (frac[:, d] if offset[d] == 1 else (1.0 - frac[:, d]))
                interp = interp + weight[:, None] * emb(idx)
            outputs.append(interp)
        return torch.cat(outputs, dim=-1)

#######################################################################################################################################################
# Small MLP using PyTorch primitives, internal helper class
#######################################################################################################################################################

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()
        
        self.net.apply(self._init_weights)
        
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

#######################################################################################################################################################
# Outward visible MLP class
#######################################################################################################################################################

class MLPTexture3D(torch.nn.Module):
    def __init__(self, AABB, channels = 3, internal_dims = 32, hidden = 2, min_max = None):
        super(MLPTexture3D, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB
        self.min_max = min_max

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))

        enc_cfg =  {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale" : per_level_scale
	    }

        gradient_scaling = 128.0
        if _has_tcnn:
            self.encoder = tcnn.Encoding(3, enc_cfg)
        else:
            self.encoder = _HashGridEncoding(3, enc_cfg).cuda()
        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))

        # Setup MLP
        mlp_cfg = {
            "n_input_dims" : self.encoder.n_output_dims,
            "n_output_dims" : self.channels,
            "n_hidden_layers" : hidden,
            "n_neurons" : self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))

    # Sample texture at a given location
    def sample(self, texc):
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        
        p_enc = self.encoder(_texc.contiguous())
        out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]

        return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass

    def cleanup(self):
        if _has_tcnn:
            tcnn.free_temporary_memory()

