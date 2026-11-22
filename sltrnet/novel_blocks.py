"""B1-specific RF layer-bank readouts, MS5--MS8.

All six insertions follow complete original decoder blocks.  Each returned
delta is exactly zero at initialization; the caller applies the common 0.1
residual multiplier.  Layer posteriors are explicit tensors, never module state.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .multilevel_common import masked, norm, resize, stage_index, stage_private_channels, uniform_plan, zero_output


STAGES = (16, 8, 4)
WIDTH = 32


def _private_at(pyramid, stage):
    if isinstance(pyramid, dict):
        return pyramid[stage] if stage in pyramid else pyramid[str(stage)]
    return pyramid[STAGES.index(stage)]


def _support(keep, size, dtype=torch.float32):
    if keep.ndim == 3:
        keep = keep[:, None]
    return resize(keep.float(), size, 'nearest').to(dtype)


def _pool(x, keep):
    mass = _support(keep, x.shape[-2:])
    return (masked(x, keep).float() * mass).sum((-2, -1)) / mass.sum((-2, -1)).clamp_min(1.)


def _bank(sources, size, keep):
    if len(sources) != 4:
        raise ValueError('Expected RF layer endpoints 3/6/9/12')
    if any(x.ndim != 4 or x.shape[1] != WIDTH for x in sources):
        raise ValueError('Four endpoints must share a 32-channel projection basis')
    return torch.stack([masked(resize(masked(x, keep), size), keep) for x in sources], 1)


def _context(a, p, hr, g, depth, valid, keep):
    size = a.shape[-2:]
    def feature(x):
        return norm(masked(resize(masked(x, keep), size), keep))
    geometry = [resize(masked(x.float(), keep), size).to(a.dtype) for x in (depth, valid)]
    return torch.cat([feature(a), feature(p), feature(hr), feature(g), *geometry,
                      _support(keep, size, a.dtype)], 1)


def soft_budget(logits, capacity=1.5, rounds=2, strength=2.):
    """Soft six-request/four-layer congestion penalty; not a hard capacity bound.

    Row sums are one.  A layer's current column mass above six/four is penalized
    before the next normalization.  Requests may choose the same useful layer.
    """
    penalty = torch.zeros_like(logits[:, :1])
    for _ in range(rounds):
        route = (logits.float() - penalty).softmax(-1)
        mass = route.sum(1, keepdim=True)
        penalty = penalty + strength * (mass / capacity).clamp_min(1.).log()
    return (logits.float() - penalty).softmax(-1)


def trajectory_features(bank):
    """Ordered [state, first difference, second difference] in a common basis.

    Endpoint differences have three-block spacing; index zero has no predecessor
    and index one has no second derivative, represented by exact zeros.
    """
    first = torch.cat([torch.zeros_like(bank[:, :1]), bank[:, 1:] - bank[:, :-1]], 1)
    second = torch.cat([torch.zeros_like(bank[:, :2]),
                        bank[:, 2:] - 2 * bank[:, 1:-1] + bank[:, :-2]], 1)
    return torch.cat([bank, first, second], 2)


def explained_residual(bank, basis, keep, ridge=1e-3):
    """Remove the ridge-linear component explained by a small spatial basis.

    bank: B,L,C,H,W; basis: B,K,H,W.  The constant basis removes the mean too.
    Only a (K+1)x(K+1) solve is formed, never an HWxHW affinity matrix.
    FP32 solves are used under AMP.  This is a linear residual, not a claim of
    statistical independence or of a complete decomposition of mutual information.
    """
    batch, layers, channels, height, width = bank.shape
    original_dtype = bank.dtype
    with torch.autocast(device_type=bank.device.type, enabled=False):
        weight = _support(keep, (height, width)).flatten(2)
        basis32 = basis.float().flatten(2)
        basis32 = torch.cat([torch.ones_like(basis32[:, :1]), basis32], 1)
        values = bank.float().reshape(batch, layers * channels, height * width)
        count = weight.sum(-1, keepdim=True).clamp_min(1.)
        weighted_basis = basis32 * weight
        gram = torch.bmm(weighted_basis, basis32.transpose(1, 2)) / count
        rhs = torch.bmm(weighted_basis, values.transpose(1, 2)) / count
        identity = torch.eye(gram.shape[-1], device=bank.device, dtype=torch.float32)[None]
        coeff = torch.linalg.solve(gram + ridge * identity, rhs)
        fitted = torch.bmm(coeff.transpose(1, 2), basis32)
        residual = (values - fitted) * weight
        return residual.reshape(batch, layers, channels, height, width).to(original_dtype)


class _ReadCell(nn.Module):
    def __init__(self, stage, explained=False, posterior=False):
        super().__init__()
        # Decoder128 + private(96/64/32) + HR64 + G128 + D/V/keep3.
        incoming = 323 + stage_private_channels(stage)
        self.query = nn.Sequential(nn.Conv2d(incoming, 16, 1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.GELU(), nn.Conv2d(16, WIDTH, 1))
        self.value = nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 1), nn.GELU())
        self.output = zero_output(WIDTH, 128)
        if explained:
            self.explain_decoder = nn.Conv2d(128, 4, 1, bias=False)
        if posterior and stage != 16:
            self.prior_log_strength = nn.Parameter(torch.zeros(()))


class NovelMultilevel(nn.Module):
    def __init__(self, arm='MS5'):
        super().__init__()
        if arm not in {'MS5', 'MS6', 'MS7', 'MS8'}:
            raise ValueError(arm)
        self.arm = arm
        self.cells = nn.ModuleDict({f'{stage}_{branch}': _ReadCell(stage,
            explained=arm == 'MS7', posterior=arm == 'MS8')
            for stage in STAGES for branch in ('a', 'b')})
        if arm == 'MS5':
            self.plan_private = nn.ModuleDict({f'{stage}_{branch}':
                nn.Linear(stage_private_channels(stage), WIDTH)
                for stage in STAGES for branch in ('a', 'b')})
            self.plan_context = nn.Linear(128 + 64 + 3, WIDTH)
            self.plan_source = nn.Linear(WIDTH, WIDTH, bias=False)
            self.plan_row_layer_bias = nn.Parameter(torch.zeros(6, 4))
        elif arm == 'MS6':
            self.trajectory_projection = nn.Conv2d(3 * WIDTH, WIDTH, 1, bias=False)
        elif arm == 'MS7':
            self.explain_shared = nn.Conv2d(128, 4, 1, bias=False)

    def plan(self, pa, pb, g, hr, depth, valid, keep, sources):
        if self.arm != 'MS5':
            return uniform_plan(g)
        geom = torch.cat([_pool(depth, keep), _pool(valid, keep),
                          keep.float().flatten(1).mean(1, keepdim=True)], 1)
        context = torch.cat([_pool(g, keep), _pool(hr, keep), geom], 1).to(g.dtype)
        global_query = self.plan_context(context)
        queries = []
        for stage in STAGES:
            for branch, pyramid in (('a', pa), ('b', pb)):
                private = _pool(_private_at(pyramid, stage), keep).to(g.dtype)
                queries.append(self.plan_private[f'{stage}_{branch}'](private) + global_query)
        query = torch.stack(queries, 1)
        pooled_sources = torch.stack([_pool(x, keep) for x in sources], 1).to(g.dtype)
        keys = self.plan_source(pooled_sources)
        logits = torch.einsum('brc,blc->brl', query, keys) / math.sqrt(WIDTH)
        return soft_budget(logits + self.plan_row_layer_bias)

    def _one(self, stage, branch, a, private, hr, g, depth, valid, keep, sources, plan, previous):
        cell = self.cells[f'{stage}_{branch}']
        size = a.shape[-2:]
        query = cell.query(_context(a, private, hr, g, depth, valid, keep))
        source_bank = _bank(sources, size, keep)
        values = source_bank
        if self.arm == 'MS6':
            trajectory = trajectory_features(source_bank)
            batch, layers, channels, height, width = trajectory.shape
            values = self.trajectory_projection(trajectory.reshape(batch * layers, channels, height, width))
            values = values.reshape(batch, layers, WIDTH, height, width)
        elif self.arm == 'MS7':
            # Fit at the RF source lattice: G and the current branch's decoder
            # supply only eight low-dimensional spatial basis functions.
            native_size = sources[0].shape[-2:]
            native_bank = _bank(sources, native_size, keep)
            shared = self.explain_shared(norm(masked(resize(g, native_size), keep)))
            decoder = cell.explain_decoder(norm(masked(resize(a, native_size), keep)))
            innovations = explained_residual(native_bank, torch.cat([shared, decoder], 1), keep)
            batch, layers, channels, height, width = innovations.shape
            values = resize(innovations.reshape(batch * layers, channels, height, width), size)
            values = values.reshape(batch, layers, channels, *size)
        # Channel normalization affects the addressing keys only; the values,
        # including signed layer derivatives or innovations, keep their amplitudes.
        keys = F.normalize(values.float(), dim=2, eps=1e-6)
        logits = torch.einsum('bchw,blchw->blhw', query.float(), keys) / math.sqrt(WIDTH)
        row = 2 * stage_index(stage) + (branch == 'b')
        prior = plan[:, row, :, None, None].float()
        if self.arm == 'MS8' and previous is not None:
            prior = resize(previous.float(), size).clamp_min(1e-6)
            prior = prior / prior.sum(1, keepdim=True).clamp_min(1e-6)
            # Five percent uniform mass leaves a path to recover an initially
            # underweighted layer.  Strength learns independently at P8/P4.
            prior = .95 * prior + .05 / 4
        strength = F.softplus(cell.prior_log_strength) if hasattr(cell, 'prior_log_strength') else 1.
        route = (logits + strength * prior.clamp_min(1e-6).log()).softmax(1)
        mixture = (values * route[:, :, None].to(values.dtype)).sum(1)
        delta = masked(cell.output(cell.value(mixture)), keep)
        return delta, route

    def forward(self, stage, a, b, pa, pb, hr, g, depth, valid, keep, sources,
                plan, previous_a=None, previous_b=None):
        if stage not in STAGES:
            raise ValueError(stage)
        da, ra = self._one(stage, 'a', a, pa, hr, g, depth, valid, keep, sources, plan, previous_a)
        db, rb = self._one(stage, 'b', b, pb, hr, g, depth, valid, keep, sources, plan, previous_b)
        return da, db, ra, rb
