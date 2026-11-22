"""B1 plus RF-layer residuals after all three complete original decoder stages."""
import hashlib
import torch
from torch import nn
from .original_n1_model import InterfaceH
from seadpnet.blocks import resize,physical_keep,mask_feature,make_branch_input
from .multilevel_common import norm,masked
from .borrowed_blocks import BorrowedMultilevel
from .novel_blocks import NovelMultilevel

ARMS={
 'MS1':'DPT-inspired stagewise layer reconstruction',
 'MS2':'SegMAN-inspired multiresolution context from RF layers',
 'MS3':'Frequency-separated RF-layer evidence at progressive scales',
 'MS4':'Deformable sparse multilevel RF readout',
 'MS5':'Coupled modality/decoder-scale budget over RF layer sources',
 'MS6':'RF layer-depth evolution trajectory readout',
 'MS7':'Shared-semantic explained-component removal and innovation readout',
 'MS8':'Coarse-to-fine layer posterior updated by private evidence',
}
READOUT_ARMS=set(ARMS)
B1_DIGEST='3e64e09ef63fe5ce74800492ef8012541e7c1cfdb95dd63c17fdbaa0851882ca'
BASELINE_DIGEST='d5d226da2b16fd79265dddec74175e5d69cf1ca7d88d2248698e9248ab6b43a1'
EXTRA_PREFIXES=('head_rf_source.','head_multilevel.')


def digest(model,exclude=(),shared=False,b0=False):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith(tuple(exclude)):continue
        if shared and name.startswith(('head_','semantic_interface.')):continue
        if b0 and name.startswith('semantic_interface.'):continue
        h.update(f'{name}|{value.dtype}|{tuple(value.shape)}'.encode())
        h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def retained_shared_state_digest(model):return digest(model,shared=True)


class SharedSourceProjection(nn.Module):
    """One shared linear basis for four already exported 384-channel endpoints."""
    def __init__(self):
        super().__init__();self.projection=nn.Conv2d(384,32,1,bias=False)

    def forward(self,*args):
        *features,keep=args
        assert len(features)==4
        return tuple(masked(self.projection(norm(masked(x,keep))),keep) for x in features)


class MultilevelH(InterfaceH):
    def __init__(self,cfg,arm):
        assert arm in ARMS
        super().__init__(cfg,'N1')
        self.arm=arm
        self.baseline_initial_digest=digest(self,b0=True);self.b1_initial_digest=digest(self)
        assert self.baseline_initial_digest==BASELINE_DIGEST and self.b1_initial_digest==B1_DIGEST
        self.expected_retained_initial_digest=B1_DIGEST
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(13042)
            self.head_rf_source=SharedSourceProjection()
            torch.random.default_generator.manual_seed(14042)
            self.head_multilevel=BorrowedMultilevel(arm) if arm in {'MS1','MS2','MS3','MS4'} else NovelMultilevel(arm)
        self.retained_initial_digest=digest(self,exclude=EXTRA_PREFIXES)
        assert self.retained_initial_digest==B1_DIGEST
        self.capture_multilevel=False;self.multilevel_diagnostics={}

    def named_new_parameters(self):
        for name,p in self.named_parameters():
            if name.startswith(EXTRA_PREFIXES):yield name,p

    def _multilevel_locals(self,images,depth,valid,g,hr,keep,*raw):
        pmask=~keep[:,0]
        pa=self.head_a_pyramid(make_branch_input(images,depth,valid,pmask,source='rgb'),keep)
        pb=self.head_b_pyramid(make_branch_input(images,depth,valid,pmask,source='depth'),keep)
        ga,gb=self.semantic_interface(g,pa[16],pb[16],valid,keep)
        sources=self.head_rf_source(*raw,keep)
        plan=self.head_multilevel.plan(pa,pb,g,hr,depth,valid,keep,sources)
        previous_a=previous_b=None;a=b=None
        diagnostics={}
        for stage in (16,8,4):
            if stage==16:
                aparts=[resize(ga,pa[stage].shape[-2:]),pa[stage]]
                bparts=[resize(gb,pb[stage].shape[-2:]),pb[stage]]
            else:
                aparts=[resize(a,pa[stage].shape[-2:]),pa[stage]]
                bparts=[resize(b,pb[stage].shape[-2:]),pb[stage]]
                if stage==4:
                    aparts.append(resize(hr,pa[stage].shape[-2:]));bparts.append(resize(hr,pb[stage].shape[-2:]))
            a=getattr(self.head_a_decoder,f'fuse{stage}')(aparts,keep)
            b=getattr(self.head_b_decoder,f'fuse{stage}')(bparts,keep)
            da,db,previous_a,previous_b=self.head_multilevel(stage,a,b,pa[stage],pb[stage],hr,g,
               depth,valid,keep,sources,plan,previous_a,previous_b)
            assert da.shape==a.shape and db.shape==b.shape
            if self.capture_multilevel and not self.training:
                support=resize(keep.float(),a.shape[-2:],'nearest')
                denominator=support.sum((0,2,3)).clamp_min(1.)
                diagnostics[str(stage)]={
                  'a_residual_norm_ratio':float((.1*da.float()).norm()/a.float().norm().clamp_min(1e-8)),
                  'b_residual_norm_ratio':float((.1*db.float()).norm()/b.float().norm().clamp_min(1e-8)),
                  'a_layer_route_mean':((previous_a.float()*support).sum((0,2,3))/denominator).detach().cpu().tolist(),
                  'b_layer_route_mean':((previous_b.float()*support).sum((0,2,3))/denominator).detach().cpu().tolist(),
                }
            a=mask_feature(a+.1*da,keep);b=mask_feature(b+.1*db,keep)
        if self.capture_multilevel and not self.training:
            self.multilevel_diagnostics={'stages':diagnostics,'plan_mean':plan.float().mean(0).detach().cpu().tolist()}
        return self.head_a_decoder.classifier(a),self.head_b_decoder.classifier(b)

    def forward(self,images,padding_mask=None,depth=None,valid=None):
        if depth is None or valid is None:raise ValueError('Depth and validity required')
        if images.shape[-2]%24 or images.shape[-1]%24:raise ValueError('Pad to multiples of24')
        keep=physical_keep(images,padding_mask);pmask=~keep[:,0]
        d=torch.nan_to_num(depth,nan=0.,posinf=0.,neginf=0.)
        v=torch.nan_to_num(valid,nan=0.,posinf=0.,neginf=0.)
        rgbd=torch.cat((images,d.to(images.dtype),v.to(images.dtype)),1)
        raw=[]
        def capture(_module,_args,output):
            assert isinstance(output,(tuple,list)) and len(output)==4
            assert all(x.ndim==4 and x.shape[1]==384 for x in output)
            raw.extend(output)
        hook=self.backbone[0].encoder.register_forward_hook(capture)
        try:features,_,_=self._shared_features(rgbd,pmask)
        finally:hook.remove()
        assert len(raw)==4
        g=self.semantic(features[0].tensors);hr=mask_feature(self.hr_stem(rgbd),keep)
        args=(images,d,v,g,hr,keep,*raw)
        if self.checkpoint_heads and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            a,b=checkpoint(self._multilevel_locals,*args,use_reentrant=False,preserve_rng_state=False)
        else:a,b=self._multilevel_locals(*args)
        return {'semantic_logits':resize(a,images.shape[-2:]),'semantic_logits_b':resize(b,images.shape[-2:]),
                'stride4_a':a,'stride4_b':b}
