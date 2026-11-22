"""Shared-probability layer routing and its single-change attribution controls.

All six query/value/output cells and their parameter storage are retained.
Only bank construction changes in STATE/FIRST/SWAP. Every variant averages
the two independently computed softmax distributions at each decoder stage.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .multilevel_common import masked, stage_index
from .novel_blocks import NovelMultilevel, WIDTH, _bank, _context, trajectory_features


class BaseSemanticInterface(nn.Module):
    """Original Base entry: both decoders receive the native G tensor."""
    def forward(self, g, a, b, valid, keep):
        return g, g


class SharedTrajectory(NovelMultilevel):
    VARIANTS = ('SR_FULL', 'SR_NO_DCSR', 'SR_STATE', 'SR_FIRST', 'SR_SWAP')

    def __init__(self, variant='SR_FULL'):
        if variant not in self.VARIANTS:
            raise ValueError(variant)
        super().__init__('MS6')
        self.variant = variant

    def trajectory_bank(self, source_bank):
        if self.variant == 'SR_STATE':
            return torch.cat((source_bank, torch.zeros_like(source_bank),
                              torch.zeros_like(source_bank)), dim=2)
        if self.variant == 'SR_SWAP':
            source_bank = source_bank[:, [0, 2, 1, 3]]
        trajectory = trajectory_features(source_bank)
        if self.variant == 'SR_FIRST':
            trajectory = torch.cat((trajectory[:, :, :2 * WIDTH],
                                    torch.zeros_like(trajectory[:, :, 2 * WIDTH:])), dim=2)
        return trajectory

    def _values(self, sources, size, keep):
        trajectory = self.trajectory_bank(_bank(sources, size, keep))
        batch, layers, channels, height, width = trajectory.shape
        values = self.trajectory_projection(trajectory.reshape(batch * layers, channels, height, width))
        return values.reshape(batch, layers, WIDTH, height, width)

    def _route(self, stage, branch, a, private, hr, g, depth, valid, keep, values, plan):
        query = self.cells[f'{stage}_{branch}'].query(_context(a, private, hr, g, depth, valid, keep))
        keys = F.normalize(values.float(), dim=2, eps=1e-6)
        logits = torch.einsum('bchw,blchw->blhw', query.float(), keys) / math.sqrt(WIDTH)
        row = 2 * stage_index(stage) + (branch == 'b')
        prior = plan[:, row, :, None, None].float()
        return (logits + prior.clamp_min(1e-6).log()).softmax(1)

    def _output(self, stage, branch, values, route, keep):
        mixture = (values * route[:, :, None].to(values.dtype)).sum(1)
        cell = self.cells[f'{stage}_{branch}']
        return masked(cell.output(cell.value(mixture)), keep)

    def forward(self, stage, a, b, pa, pb, hr, g, depth, valid, keep, sources,
                plan, previous_a=None, previous_b=None):
        values = self._values(sources, a.shape[-2:], keep)
        ra = self._route(stage, 'a', a, pa, hr, g, depth, valid, keep, values, plan)
        rb = self._route(stage, 'b', b, pb, hr, g, depth, valid, keep, values, plan)
        shared_route = .5 * (ra + rb)
        return (self._output(stage, 'a', values, shared_route, keep),
                self._output(stage, 'b', values, shared_route, keep),
                shared_route, shared_route)


# Explicit compatibility name for callers that previously imported this symbol.
ControlledTrajectory = SharedTrajectory
