"""SUN37 full S/M family, cold from RF; original complete SUN-L forward path."""
from pathlib import Path
import torch
from seadpnet.model import EnhancedH,architecture as base_architecture
from .family_model import FamilyDCSR,FamilyTrajectory
from .multilevel_model import SharedSourceProjection
from .sun_reference_model import SunReadoutH,retained_digest

CHECKPOINT_FORMAT='sltr-sun-family-20261025-v1'
ARMS={'SUN_S':'SLTR-Net-S','SUN_M':'SLTR-Net-M'}
SPECS={
 'SUN_S':{'scale':.5,'private':[24,48,64],'decoder':80,'base':26252826,'total':26333740,'widths':[48,96,192],'a':[1,1,1],'b':[1,2,2]},
 'SUN_M':{'scale':.75,'private':[32,64,96],'decoder':96,'base':28951946,'total':29041404,'widths':[64,128,256],'a':[1,2,2],'b':[2,3,4]},
 'SUN_L':{'scale':1.,'private':[32,64,96],'decoder':128,'base':29821770,'total':29920636,'widths':[64,128,256],'a':[1,2,2],'b':[2,3,4]},
}


class SunFamilyH(EnhancedH):
    def __init__(self,cfg):
        spec=SPECS[cfg['research_arm']]
        assert cfg['num_classes']==37 and cfg['encoder_input_scale']==spec['scale']
        assert cfg['pyramid_widths']==spec['private'] and cfg['decoder_width']==spec['decoder']
        assert not any(cfg.get(k) for k in ('nyu_checkpoint','warm_start_checkpoint','c11_checkpoint'))
        super().__init__(cfg['pretrained_checkpoint'],'rgb','depth',spec['private'],spec['decoder'],
                         cfg['branch_a'],cfg['branch_b'],'standard',37,spec['scale'])
        self.research_arm=cfg['research_arm']
        assert sum(p.numel() for p in self.parameters())==spec['base']
        self.base_initial_digest=retained_digest(self)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4042)
            self.semantic_interface=FamilyDCSR(spec['private'][-1])
            self.dcsr_initial_digest=retained_digest(self,include_dcsr=True)
            torch.random.default_generator.manual_seed(13042)
            self.head_rf_source=SharedSourceProjection()
            torch.random.default_generator.manual_seed(14042)
            self.head_multilevel=FamilyTrajectory(spec['decoder'],spec['private'])
        assert retained_digest(self)==self.base_initial_digest
        assert sum(p.numel() for p in self.parameters())==spec['total']

    _locals=SunReadoutH._locals
    forward=SunReadoutH.forward


def build_model(cfg):return SunFamilyH(cfg)
def architecture(model):
    result=base_architecture(model)
    result.update(format=CHECKPOINT_FORMAT,dataset='SUNRGBD37',model_name='SLTR-Net-'+model.research_arm[-1],
                  experiment_name=model.research_arm,research_arm=model.research_arm,dcsr=True,mc_ltr=True,
                  layer_routing='mean_of_branch_softmax_probabilities',
                  initialization='original_rf_pretraining_cold_sun37_heads')
    return result
def load_experiment(path,device='cuda'):
    state=torch.load(Path(path),map_location='cpu',weights_only=True)
    assert state['format']==CHECKPOINT_FORMAT
    model=build_model(state['config']);assert state['architecture']==architecture(model)
    model.load_state_dict(state['model'],strict=True)
    return model.to(device),state
