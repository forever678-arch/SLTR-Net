"""Original shared Y5 cold graph, with exactly one joint P16 semantic interface."""
import torch
import hashlib
from seadpnet.model import EnhancedH
from seadpnet.blocks import resize,physical_keep,mask_feature,make_branch_input
from .interface_modules import ARMS,CONSTRUCTORS

READOUT_ARMS=set(ARMS)


def retained_shared_state_digest(model):
    # Legacy shared digest excludes head_* only; the new top-level interface
    # must be excluded explicitly to compare the unchanged RF/semantic/HR.
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith(('head_a_','head_b_','semantic_interface.')):continue
        h.update(f'{name}|{value.dtype}|{tuple(value.shape)}'.encode())
        h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class InterfaceH(EnhancedH):
    def __init__(self,cfg,arm):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(cfg['pretrained_checkpoint'],cfg.get('head_a_source','rgb'),
                         cfg.get('head_b_source','depth'),cfg.get('pyramid_widths',(32,64,96)),
                         cfg.get('decoder_width',128),cfg.get('branch_a'),cfg.get('branch_b'),
                         cfg.get('shared_injection','standard'),cfg.get('num_classes',40),
                         cfg.get('encoder_input_scale',1.0))
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4042)
            self.semantic_interface=CONSTRUCTORS[arm]()

    def _locals(self,images,depth,valid,semantic,hr,keep):
        pmask=~keep[:,0]
        pa=self.head_a_pyramid(make_branch_input(images,depth,valid,pmask,source='rgb'),keep)
        pb=self.head_b_pyramid(make_branch_input(images,depth,valid,pmask,source='depth'),keep)
        ga,gb=self.semantic_interface(semantic,pa[16],pb[16],valid,keep)
        return (self.head_a_decoder(ga,hr,pa,keep),self.head_b_decoder(gb,hr,pb,keep))

    def forward(self,images,padding_mask=None,depth=None,valid=None):
        if depth is None or valid is None:
            raise ValueError('Depth and validity are required')
        if images.shape[-2]%24 or images.shape[-1]%24:
            raise ValueError('Inputs must be padded to multiples of24')
        keep=physical_keep(images,padding_mask);pmask=~keep[:,0]
        d=torch.nan_to_num(depth,nan=0.,posinf=0.,neginf=0.)
        v=torch.nan_to_num(valid,nan=0.,posinf=0.,neginf=0.)
        rgbd=torch.cat([images,d.to(images.dtype),v.to(images.dtype)],1)
        features,_,_=self._shared_features(rgbd,pmask)
        semantic=self.semantic(features[0].tensors)
        hr=mask_feature(self.hr_stem(rgbd),keep)
        if self.checkpoint_heads and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            a,b=checkpoint(self._locals,images,d,v,semantic,hr,keep,use_reentrant=False,preserve_rng_state=False)
        else:
            a,b=self._locals(images,d,v,semantic,hr,keep)
        return {'semantic_logits':resize(a,images.shape[-2:]),'semantic_logits_b':resize(b,images.shape[-2:]),
                'stride4_a':a,'stride4_b':b}
