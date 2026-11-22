"""SUN37 shared-layer-route full model, cold from original RF weights.

SUN has independent 37-class heads. No NYU classifier or fine-tuned weight is
accepted. Added modules use isolated CPU RNG streams, leaving Base intact.
"""
from pathlib import Path
import hashlib
import torch
from torch import nn
from seadpnet.model import EnhancedH, architecture as base_architecture
from seadpnet.blocks import resize, physical_keep, mask_feature, make_branch_input
from .interface_modules import CoupledInterface
from .multilevel_common import masked, norm
from .sun_shared_readout import SharedRouteReadout

CHECKPOINT_FORMAT = 'sltr-sharedroute-sun37-20261024-v1'
ARMS = {'SUN_SHARED': 'SLTR-Net shared layer route'}
EXTRA_PREFIXES = ('semantic_interface.', 'head_rf_source.', 'head_multilevel.')


def retained_digest(model, include_dcsr=False):
    h = hashlib.sha256()
    excluded = EXTRA_PREFIXES[1:] if include_dcsr else EXTRA_PREFIXES
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(excluded):
            continue
        h.update(f'{name}|{value.dtype}|{tuple(value.shape)}'.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class SharedSourceProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(384, 32, 1, bias=False)

    def forward(self, *args):
        *features, keep = args
        if len(features) != 4:
            raise ValueError('Expected RF endpoints 3/6/9/12')
        return tuple(masked(self.projection(norm(masked(x, keep))), keep) for x in features)


class SunReadoutH(EnhancedH):
    def __init__(self, cfg):
        arm = cfg['research_arm']
        if arm not in ARMS or cfg['num_classes'] != 37:
            raise ValueError('SUN controls require known arm and 37 classes')
        if any(cfg.get(k) for k in ('warm_start_checkpoint', 'c11_checkpoint', 'nyu_checkpoint')):
            raise ValueError('Original RF cold initialization only')
        if tuple(cfg['pyramid_widths']) != (32, 64, 96) or cfg['decoder_width'] != 128:
            raise ValueError('DCSR/MC-LTR audited for decoder128, private32/64/96')
        super().__init__(cfg['pretrained_checkpoint'], 'rgb', 'depth',
                         cfg['pyramid_widths'], cfg['decoder_width'],
                         cfg['branch_a'], cfg['branch_b'], 'standard', 37, 1.0)
        self.research_arm = arm
        self.base_initial_digest = retained_digest(self)
        if arm == 'SUN_SHARED':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(4042)
                self.semantic_interface = CoupledInterface()
        self.dcsr_initial_digest = retained_digest(self, include_dcsr=True)
        if arm == 'SUN_SHARED':
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(13042)
                self.head_rf_source = SharedSourceProjection()
                torch.random.default_generator.manual_seed(14042)
                self.head_multilevel = SharedRouteReadout('MS6')
        if retained_digest(self) != self.base_initial_digest:
            raise RuntimeError('Added modules changed original SUN initialization')

    def _locals(self, images, depth, valid, g, hr, keep, *raw):
        pmask = ~keep[:, 0]
        pa = self.head_a_pyramid(make_branch_input(images, depth, valid, pmask, source='rgb'), keep)
        pb = self.head_b_pyramid(make_branch_input(images, depth, valid, pmask, source='depth'), keep)
        ga, gb = self.semantic_interface(g, pa[16], pb[16], valid, keep)
        sources = self.head_rf_source(*raw, keep)
        plan = self.head_multilevel.plan(pa, pb, g, hr, depth, valid, keep, sources)
        previous_a = previous_b = None
        a = b = None
        for stage in (16, 8, 4):
            aparts = [resize(ga if stage == 16 else a, pa[stage].shape[-2:]), pa[stage]]
            bparts = [resize(gb if stage == 16 else b, pb[stage].shape[-2:]), pb[stage]]
            if stage == 4:
                aparts.append(resize(hr, pa[stage].shape[-2:]))
                bparts.append(resize(hr, pb[stage].shape[-2:]))
            a = getattr(self.head_a_decoder, f'fuse{stage}')(aparts, keep)
            b = getattr(self.head_b_decoder, f'fuse{stage}')(bparts, keep)
            da, db, previous_a, previous_b = self.head_multilevel(
                stage, a, b, pa[stage], pb[stage], hr, g, depth, valid, keep,
                sources, plan, previous_a, previous_b)
            a = mask_feature(a + .1 * da, keep)
            b = mask_feature(b + .1 * db, keep)
        return self.head_a_decoder.classifier(a), self.head_b_decoder.classifier(b)

    def forward(self, images, padding_mask=None, depth=None, valid=None):
        if depth is None or valid is None:
            raise ValueError('Depth and raw validity required')
        if images.shape[-2] % 24 or images.shape[-1] % 24:
            raise ValueError('Pad input to multiple24')
        keep = physical_keep(images, padding_mask)
        d = torch.nan_to_num(depth, nan=0., posinf=0., neginf=0.)
        v = torch.nan_to_num(valid, nan=0., posinf=0., neginf=0.)
        rgbd = torch.cat((images, d.to(images.dtype), v.to(images.dtype)), 1)
        raw = []
        def capture(_module, _args, output):
            if len(output) != 4 or any(x.ndim != 4 or x.shape[1] != 384 for x in output):
                raise ValueError('Expected four384-channel RF endpoint maps')
            raw.extend(output)
        hook = self.backbone[0].encoder.register_forward_hook(capture)
        try:
            features, _, _ = self._shared_features(rgbd, ~keep[:, 0])
        finally:
            if hook is not None:
                hook.remove()
        g = self.semantic(features[0].tensors)
        hr = mask_feature(self.hr_stem(rgbd), keep)
        args = (images, d, v, g, hr, keep, *raw)
        if self.checkpoint_heads and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            a, b = checkpoint(self._locals, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            a, b = self._locals(*args)
        return {'semantic_logits': resize(a, images.shape[-2:]),
                'semantic_logits_b': resize(b, images.shape[-2:]), 'stride4_a': a, 'stride4_b': b}


def build_model(cfg):
    return SunReadoutH(cfg)


def architecture(model):
    result = base_architecture(model)
    result.update(format=CHECKPOINT_FORMAT, dataset='SUNRGBD37',
                  model_name='SLTR-Net', experiment_name=ARMS[model.research_arm],
                  research_arm=model.research_arm, dcsr=True,
                  mc_ltr=True, layer_routing='mean_of_branch_softmax_probabilities',
                  initialization='original_rf_pretraining_cold_sun37_heads')
    return result


def load_experiment(path, device='cuda'):
    state = torch.load(Path(path), map_location='cpu', weights_only=True)
    if state.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Expected readout SUN37 checkpoint')
    model = build_model(state['config'])
    if state['architecture'] != architecture(model):
        raise ValueError('Saved SUN architecture differs')
    model.load_state_dict(state['model'], strict=True)
    return model.to(device), state
