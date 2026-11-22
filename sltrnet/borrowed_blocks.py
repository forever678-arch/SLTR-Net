"""Four literature-inspired, modality-conditioned RF endpoint readout modules.

All branches operate after the preserved B1 fuse convolutions. Their final
projections are exactly zero at initialization. The wrapper owns the common
384->32 source projection and the residual multiplier (0.1); this module owns
neither. RF endpoint depths are not native image-pyramid resolutions.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _resize(x, size):
    return x if tuple(x.shape[-2:]) == tuple(size) else F.interpolate(
        x, size=size, mode="bilinear", align_corners=False)


def _norm(x):
    """Per-pixel RMS normalization, accumulating in FP32 without spatial padding."""
    xf = x.float()
    return (xf * (xf.square().mean(1, keepdim=True) + 1e-5).rsqrt()).to(x.dtype)


def _coverage(keep, size):
    if keep.ndim == 3:
        keep = keep[:, None]
    if tuple(keep.shape[-2:]) == tuple(size):
        return keep.float().clamp(0, 1)
    return F.interpolate(keep.float(), size=size, mode="area").clamp(0, 1)


def _pool(x, keep, output_size):
    m = _coverage(keep, x.shape[-2:])
    num = F.adaptive_avg_pool2d(x.float() * m, output_size)
    den = F.adaptive_avg_pool2d(m, output_size)
    return (num / den.clamp_min(1e-6)).to(x.dtype)


def _low(x, keep, kernel):
    m = _coverage(keep, x.shape[-2:])
    num = F.avg_pool2d(x.float() * m, kernel, 1, kernel // 2)
    den = F.avg_pool2d(m, kernel, 1, kernel // 2)
    return (num / den.clamp_min(1e-6)).to(x.dtype)


def _zero_output():
    out = nn.Conv2d(32, 128, 1)
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    return out


class _Query(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.decoder = nn.Conv2d(128, 32, 1)
        self.private = nn.Conv2d(channels, 32, 1, bias=False)
        self.hr = nn.Conv2d(64, 32, 1, bias=False)

    def forward(self, a, private, hr):
        size = a.shape[-2:]
        return F.gelu(_norm(self.decoder(_norm(a)) +
                           self.private(_resize(_norm(private), size)) +
                           self.hr(_resize(_norm(hr), size))))


class _Readout(nn.Module):
    def __init__(self, arm, stage):
        super().__init__()
        self.arm = arm
        self.stage = stage
        self.query = _Query({16: 96, 8: 64, 4: 32}[stage])
        self.output = _zero_output()
        if arm == "MS1":
            self.route = nn.Conv2d(32, 4, 1)
            self.reassemble = nn.ModuleList([
                nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, groups=32),
                              nn.GELU(), nn.Conv2d(32, 32, 1)) for _ in range(4)])
            self.mix = nn.Conv2d(64, 32, 1)
        elif arm == "MS2":
            self.route = nn.Conv2d(32, 4, 1)
            self.scale_route = nn.Conv2d(32, 3, 1)
            self.local = nn.Conv2d(32, 32, 3, padding=1, groups=32)
            self.mix = nn.Conv2d(96, 32, 1)
        elif arm == "MS3":
            self.route_low = nn.Conv2d(32, 4, 1)
            self.route_high = nn.Conv2d(32, 4, 1)
            self.low_kernel_route = nn.Conv2d(32, 2, 1)
            self.high_gate = nn.Conv2d(32, 32, 1)
            self.mix = nn.Conv2d(96, 32, 1)
        elif arm == "MS4":
            self.point_route = nn.Conv2d(32, 8, 1)
            self.offset = nn.Conv2d(32, 16, 1)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.mix = nn.Conv2d(64, 32, 1)
        else:
            raise ValueError(arm)

    def forward(self, a, private, hr, keep, sources):
        size = a.shape[-2:]
        q = self.query(a, private, hr)
        src = [_norm(x) * (_coverage(keep, x.shape[-2:]) > 0).to(x.dtype) for x in sources]
        if self.arm == "MS1":
            route = self.route(q).float().softmax(1)
            carry = None
            # Four endpoint depths are explicitly reassembled into an artificial
            # 4/8/16/32 spatial hierarchy, as in DPT's reassembly principle.
            for level in (3, 2, 1, 0):
                stride = (4, 8, 16, 32)[level]
                target = tuple(max(1, round(n * self.stage / stride)) for n in size)
                x = _resize(src[level], target)
                x = x * _resize(route[:, level:level + 1], target).to(x.dtype)
                if carry is not None:
                    x = x + _resize(carry, target)
                carry = self.reassemble[level](x)
            hidden = self.mix(torch.cat([q, _resize(carry, size)], 1))
        elif self.arm == "MS2":
            route = self.route(q).float().softmax(1)
            scales = self.scale_route(q).float().softmax(1)
            contexts = torch.zeros_like(q)
            local = torch.zeros_like(q)
            # Fixed 1/2/4 context bins: bounded memory, no global NxN attention.
            for level, x in enumerate(src):
                rl = route[:, level:level + 1].to(x.dtype)
                local = local + rl * _resize(x, size)
                multi = torch.zeros_like(q)
                for index, bins in enumerate((1, 2, 4)):
                    pooled = _pool(x, keep, (bins, bins))
                    multi = multi + scales[:, index:index + 1].to(x.dtype) * _resize(pooled, size)
                contexts = contexts + rl * multi
            hidden = self.mix(torch.cat([q, self.local(local), contexts], 1))
        elif self.arm == "MS3":
            low_route = self.route_low(q).float().softmax(1)
            high_route = self.route_high(q).float().softmax(1)
            kernels = self.low_kernel_route(q).float().softmax(1)
            low = torch.zeros_like(q)
            high = torch.zeros_like(q)
            for level, x in enumerate(src):
                smooth3 = _low(x, keep, 3)
                smooth5 = _low(x, keep, 5)
                lp = (kernels[:, :1].to(x.dtype) * _resize(smooth3, size) +
                      kernels[:, 1:].to(x.dtype) * _resize(smooth5, size))
                low = low + low_route[:, level:level + 1].to(x.dtype) * lp
                high = high + high_route[:, level:level + 1].to(x.dtype) * _resize(x - smooth3, size)
            high = high * self.high_gate(q).sigmoid()
            hidden = self.mix(torch.cat([q, low, high], 1))
            # Diagnostics: arithmetic mean of actual low/high endpoint weights;
            # two distinct spectral routes remain intact in computation.
            route = (low_route + high_route) * 0.5
        else:
            hidden, route = self._sparse(q, src, keep)
        delta = self.output(F.gelu(_norm(hidden)))
        delta = delta * (_coverage(keep, size) > 0).to(delta.dtype)
        return delta, route

    def _sparse(self, q, sources, keep):
        batch, _, height, width = q.shape
        offsets = self.offset(q).float().reshape(batch, 4, 2, 2, height, width)
        offsets = 2.0 * offsets.tanh()
        weights = self.point_route(q).float().softmax(1).reshape(batch, 4, 2, height, width)
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=q.device, dtype=torch.float32) + 0.5) / height * 2 - 1,
            (torch.arange(width, device=q.device, dtype=torch.float32) + 0.5) / width * 2 - 1,
            indexing="ij")
        base = torch.stack([xx, yy], -1)[None, :, :, None, :]
        sampled, coverages = [], []
        for level, x in enumerate(sources):
            sh, sw = x.shape[-2:]
            off = offsets[:, level].permute(0, 3, 4, 1, 2)
            fixed = off.new_tensor([[-0.5, 0.0], [0.5, 0.0]])
            grid = base + (off + fixed) * off.new_tensor([2.0 / sw, 2.0 / sh])
            grid = grid.reshape(batch, height, width * 2, 2)
            # CUDA grid_sample does not implement every BF16 path. Both input
            # and grid are explicitly FP32 inside a disabled-autocast region.
            with torch.autocast(device_type=q.device.type, enabled=False):
                val = F.grid_sample(x.float(), grid.float(), mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
                cov = F.grid_sample(_coverage(keep, (sh, sw)), grid.float(), mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            sampled.append(val.reshape(batch, 32, height, width, 2).permute(0, 1, 4, 2, 3))
            coverages.append(cov.reshape(batch, height, width, 2).permute(0, 3, 1, 2))
        coverage = torch.stack(coverages, 1)
        weights = weights * coverage
        den = weights.sum((1, 2), keepdim=True)
        weights = torch.where(den > 0, weights / den.clamp_min(1e-8), torch.full_like(weights, 1 / 8))
        value = sum((sampled[l] * weights[:, l:l + 1]).sum(2) for l in range(4))
        route = weights.sum(2)
        hidden = self.mix(torch.cat([q, value.to(q.dtype)], 1))
        return hidden, route


class BorrowedMultilevel(nn.Module):
    """Common stage-major interface: P16 A/B, P8 A/B, then P4 A/B.

    `plan` is uniform because these four methods compute their actual routes at
    each decoding stage; returned routes are the useful diagnostics. MS3 returns
    the mean of its actual low- and high-band level allocations.
    """
    def __init__(self, arm):
        super().__init__()
        if arm not in {"MS1", "MS2", "MS3", "MS4"}:
            raise ValueError(arm)
        self.arm = arm
        self.stages = nn.ModuleDict({str(s): nn.ModuleList([_Readout(arm, s), _Readout(arm, s)])
                                     for s in (16, 8, 4)})

    def plan(self, pa, pb, g, hr, depth, valid, keep, sources):
        return g.new_full((g.shape[0], 6, 4), 0.25, dtype=torch.float32)

    def forward(self, stage, a, b, pa, pb, hr, g, depth, valid, keep, sources,
                plan, previous_a=None, previous_b=None):
        if len(sources) != 4 or any(x.shape[1] != 32 for x in sources):
            raise ValueError("Expected four RF endpoints after the shared 384->32 projection")
        aa, bb = self.stages[str(stage)]
        da, ra = aa(a, pa, hr, keep, sources)
        db, rb = bb(b, pb, hr, keep, sources)
        return da, db, ra, rb
