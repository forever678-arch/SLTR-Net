"""S/M/L SLTR family: unchanged legacy scale budgets plus full shared routing."""
import torch
from torch import nn

from seadpnet.model import EnhancedH
from .multilevel_model import MultilevelH, SharedSourceProjection, digest, retained_shared_state_digest
from .interface_modules import CoupledInterface
from .control_blocks import SharedTrajectory
from .novel_blocks import WIDTH, STAGES
from .multilevel_common import zero_output

VARIANTS={
    'SLTR_S':{'config':'seadp-net-s.yaml','scale':.5,'private':(24,48,64),'decoder':80,'base_parameters':26253312},
    'SLTR_M':{'config':'seadp-net-m.yaml','scale':.75,'private':(32,64,96),'decoder':96,'base_parameters':28952528},
    'SLTR_L':{'config':'seadp-net-l.yaml','scale':1.,'private':(32,64,96),'decoder':128,'base_parameters':29822544},
}
ARMS={'SLTR_S':'SLTR-Net-S with full DCSR and shared-routing MCLTR',
      'SLTR_M':'SLTR-Net-M with full DCSR and shared-routing MCLTR'}
READOUT_ARMS=set(ARMS)
EXTRA_PREFIXES=('semantic_interface.','head_rf_source.','head_multilevel.')


class FamilyDCSR(CoupledInterface):
    def __init__(self,private_channels):
        nn.Module.__init__(self)
        self.query_a=nn.Conv2d(private_channels,32,1)
        self.query_b=nn.Conv2d(private_channels,32,1)
        self.key=nn.Conv2d(128,32,1)
        self.bias=nn.Parameter(torch.zeros(2,9))
        self.dim=32


class FamilyReadCell(nn.Module):
    def __init__(self,decoder_channels,private_channels):
        super().__init__()
        incoming=decoder_channels+private_channels+64+128+3
        self.query=nn.Sequential(nn.Conv2d(incoming,16,1),nn.GELU(),
            nn.Conv2d(16,16,3,padding=1,groups=16),nn.GELU(),nn.Conv2d(16,WIDTH,1))
        self.value=nn.Sequential(nn.Conv2d(WIDTH,WIDTH,1),nn.GELU())
        self.output=zero_output(WIDTH,decoder_channels)


class FamilyTrajectory(SharedTrajectory):
    def __init__(self,decoder_channels,private_widths):
        nn.Module.__init__(self)
        self.arm='MS6';self.variant='SR_FULL'
        widths=dict(zip((4,8,16),private_widths))
        self.cells=nn.ModuleDict({f'{stage}_{branch}':FamilyReadCell(decoder_channels,widths[stage])
                                 for stage in STAGES for branch in ('a','b')})
        self.trajectory_projection=nn.Conv2d(3*WIDTH,WIDTH,1,bias=False)


class FamilyH(EnhancedH):
    def __init__(self,cfg,arm):
        spec=VARIANTS[arm]
        assert cfg['encoder_input_scale']==spec['scale']
        assert tuple(cfg['pyramid_widths'])==spec['private'] and cfg['decoder_width']==spec['decoder']
        assert cfg['num_classes']==40 and cfg['shared_injection']=='standard'
        assert cfg.get('encoder_keep_layers',list(range(12)))==list(range(12))
        super().__init__(cfg['pretrained_checkpoint'],'rgb','depth',cfg['pyramid_widths'],
                         cfg['decoder_width'],cfg['branch_a'],cfg['branch_b'],'standard',40,
                         cfg['encoder_input_scale'])
        assert sum(p.numel() for p in self.parameters())==spec['base_parameters']
        self.arm=arm;self.family_variant=arm[-1];self.has_dcsr=True;self.has_multilevel=True
        self.shared_layer_route=True;self.inactive_parameter_elements=0
        self.capture_multilevel=False;self.multilevel_diagnostics={}
        self.baseline_initial_digest=digest(self)
        self.initialization_reference='same-scale original Base cold RF initialization plus DCSR and full shared-routing MCLTR'
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4042)
            self.semantic_interface=FamilyDCSR(spec['private'][-1])
            self.b1_initial_digest=digest(self)
            torch.random.default_generator.manual_seed(13042)
            self.head_rf_source=SharedSourceProjection()
            torch.random.default_generator.manual_seed(14042)
            self.head_multilevel=FamilyTrajectory(spec['decoder'],spec['private'])
        self.retained_initial_digest=digest(self,exclude=EXTRA_PREFIXES)
        self.expected_retained_initial_digest=self.baseline_initial_digest
        assert self.retained_initial_digest==self.baseline_initial_digest
        self.control_initial_digest=digest(self)
        self.original_full_initial_digest=self.control_initial_digest

    # Execute the exact original sequence, including complete Fuse then residual.
    _multilevel_locals=MultilevelH._multilevel_locals
    forward=MultilevelH.forward

    def named_new_parameters(self):
        for name,p in self.named_parameters():
            if name.startswith(EXTRA_PREFIXES):yield name,p

    def required_gradient_groups(self):
        return ['backbone','semantic','hr_stem','head_a_pyramid','head_b_pyramid',
                'head_a_decoder','head_b_decoder','semantic_interface','head_multilevel']
