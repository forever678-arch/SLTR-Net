"""SUN shared layer probabilities; preserve all full MCLTR parameter names.

Only the two branch softmax probabilities are averaged at each decoder stage.
Branch queries, value projections, and output projections remain independent.
"""
import math

import torch
from torch.nn import functional as F

from .multilevel_common import masked, stage_index
from .novel_blocks import NovelMultilevel, WIDTH, _bank, _context, trajectory_features


class SharedRouteReadout(NovelMultilevel):
    def __init__(self, arm='MS6'):
        if arm != 'MS6':
            raise ValueError('Shared-route SUN retains the full MS6 candidate bank')
        super().__init__(arm)

    def _values(self, sources, size, keep):
        trajectory = trajectory_features(_bank(sources, size, keep))
        batch, layers, channels, height, width = trajectory.shape
        values = self.trajectory_projection(
            trajectory.reshape(batch * layers, channels, height, width))
        return values.reshape(batch, layers, WIDTH, height, width)

    def _route(self, stage, branch, state, private, hr, g, depth, valid, keep, values, plan):
        query = self.cells[f'{stage}_{branch}'].query(
            _context(state, private, hr, g, depth, valid, keep))
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
        if stage not in (16, 8, 4):
            raise ValueError(stage)
        values = self._values(sources, a.shape[-2:], keep)
        ra = self._route(stage, 'a', a, pa, hr, g, depth, valid, keep, values, plan)
        rb = self._route(stage, 'b', b, pb, hr, g, depth, valid, keep, values, plan)
        route = .5 * (ra + rb)
        return (self._output(stage, 'a', values, route, keep),
                self._output(stage, 'b', values, route, keep), route, route)
