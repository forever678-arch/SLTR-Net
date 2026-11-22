"""Seven independent G -> dual-P16 interfaces, as specified on 2026-10-18."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = {
    'L1': 'DRA-inspired common/appearance/geometry experts with receiver-specific routing',
    'L2': 'TransNeXt-inspired joint local and pooled-global semantic readout',
    'L3': 'AnyUp-inspired filter-basis structure keys and unprojected G values',
    'N1': 'Coupled dual requests with soft one-sided source congestion suppression',
    'N2': 'Local-condition orthogonal channel-plane rotations; Q independent of G',
    'N3': 'Transpose writeback and cross-branch readback through shared G addresses',
    'N4': 'Modality-conditioned limited linear finite-volume semantic remapping',
}


def resize(x, size, mode='bilinear'):
    return F.interpolate(x, size=size, mode=mode, align_corners=False if mode=='bilinear' else None)


def support(keep, size):
    return F.adaptive_avg_pool2d(keep.float(), size)


def mask(x, keep):
    return x*resize(keep.float(),x.shape[-2:],'nearest').to(x.dtype)


def flat(x):
    return x.flatten(2).transpose(1,2)


def unflat(x, size):
    return x.transpose(1,2).reshape(x.shape[0],x.shape[-1],*size)


def neighborhood(source_size, target_size, device, radius=1):
    sh,sw=source_size;th,tw=target_size
    y=(torch.arange(th,device=device,dtype=torch.float32)+.5)*sh/th-.5
    x=(torch.arange(tw,device=device,dtype=torch.float32)+.5)*sw/tw-.5
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    cy,cx=(yy+.5).floor().long(),(xx+.5).floor().long()
    offsets=torch.tensor([(dy,dx) for dy in range(-radius,radius+1)
                          for dx in range(-radius,radius+1)],device=device)
    sy=cy.flatten()[:,None]+offsets[:,0]
    sx=cx.flatten()[:,None]+offsets[:,1]
    valid=(sy>=0)&(sy<sh)&(sx>=0)&(sx<sw)
    index=sy.clamp(0,sh-1)*sw+sx.clamp(0,sw-1)
    return index,valid


def gather(x,index):
    # [B,C,H,W] -> [B,N,K,C], integer addresses (no trainable offsets).
    return x.flatten(2)[:,:,index.flatten()].reshape(x.shape[0],x.shape[1],*index.shape).permute(0,2,3,1)


def normalized_weights(scores, available):
    with torch.autocast(device_type=scores.device.type,enabled=False):
        a=available.float()
        logits=scores.float()+a.clamp_min(1e-8).log()
        logits=logits.masked_fill(a<=0,-1e4)
        w=logits.softmax(-1)*a.gt(0)
        return w/w.sum(-1,keepdim=True).clamp_min(1e-8)


def read_values(w,values):
    # FP32 attention accumulation, then restore the original feature dtype.
    with torch.autocast(device_type=values.device.type,enabled=False):
        out=(w.float().unsqueeze(-1)*values.float()).sum(-2)
    return out.to(values.dtype)


class LocalAddress(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        self.query_a=nn.Conv2d(96,dim,1)
        self.query_b=nn.Conv2d(96,dim,1)
        self.key=nn.Conv2d(128,dim,1)
        self.bias=nn.Parameter(torch.zeros(2,9))
        self.dim=dim

    def address(self,g,a,b,keep):
        size=a.shape[-2:]
        index,in_bounds=neighborhood(g.shape[-2:],size,g.device)
        available=gather(support(keep,g.shape[-2:]),index)[...,0]*in_bounds
        keys=gather(self.key(g),index).float()
        weights=[]
        for i,q in enumerate((self.query_a(a),self.query_b(b))):
            scores=(flat(q).float().unsqueeze(-2)*keys).sum(-1)/math.sqrt(self.dim)
            weights.append(normalized_weights(scores+self.bias[i],available))
        return weights,index,available


def expert(ci):
    seq=nn.Sequential(nn.Conv2d(ci,32,1),nn.GELU(),
                      nn.Conv2d(32,32,3,padding=1,groups=32),nn.GELU(),nn.Conv2d(32,128,1))
    nn.init.zeros_(seq[-1].weight);nn.init.zeros_(seq[-1].bias)
    return seq


class ExpertInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.common=expert(128);self.appearance=expert(224);self.geometry=expert(225)
        self.routers=nn.ModuleList(nn.Sequential(nn.Conv2d(225,32,1),nn.GELU(),nn.Conv2d(32,3,1)) for _ in range(2))

    def forward(self,g,a,b,valid,keep):
        size=a.shape[-2:];z=resize(g,size);rho=support(valid*keep,size).to(z.dtype)
        deltas=torch.stack([self.common(z),self.appearance(torch.cat([z,a],1)),
                            self.geometry(torch.cat([z,b,rho],1))],1)
        outputs=[]
        for router,p in zip(self.routers,(a,b)):
            pi=router(torch.cat([z,p,rho],1)).float().softmax(1).to(z.dtype)
            outputs.append(mask(z+(deltas*pi.unsqueeze(2)).sum(1),keep))
        return tuple(outputs)


class LocalGlobalInterface(LocalAddress):
    def __init__(self):
        super().__init__()
        self.value=nn.Conv2d(128,128,1)
        self.pool_pre=nn.Conv2d(128,128,1)
        self.pool_norm=nn.LayerNorm(128)
        self.global_bias=nn.Parameter(torch.zeros(2,1))

    def forward(self,g,a,b,valid,keep):
        size=a.shape[-2:];index,inside=neighborhood(g.shape[-2:],size,g.device)
        sup=support(keep,g.shape[-2:])
        avail=gather(sup,index)[...,0]*inside
        k_local=gather(self.key(g),index).float()
        v_local=gather(self.value(g),index)
        # Fixed 4x6 pool, clipped on tiny synthetic test grids.
        pooled_size=(min(4,g.shape[-2]),min(6,g.shape[-1]))
        mass=F.adaptive_avg_pool2d(sup,pooled_size)
        pg=F.adaptive_avg_pool2d(F.gelu(self.pool_pre(g))*sup.to(g.dtype),pooled_size)/mass.clamp_min(1e-8)
        pg=self.pool_norm(pg.to(g.dtype).movedim(1,-1)).movedim(-1,1)
        k_pool=flat(self.key(pg)).float();v_pool=flat(self.value(pg))
        outputs=[]
        for i,q in enumerate((self.query_a(a),self.query_b(b))):
            q=flat(q).float()
            local=(q.unsqueeze(-2)*k_local).sum(-1)/math.sqrt(self.dim)+self.bias[i]
            with torch.autocast(device_type=g.device.type,enabled=False):
                global_scores=q@k_pool.transpose(-1,-2)/math.sqrt(self.dim)+self.global_bias[i]
            scores=torch.cat([local,global_scores],-1)
            available=torch.cat([avail,flat(mass)[...,0][:,None,:].expand(-1,q.shape[1],-1)],-1)
            w=normalized_weights(scores,available)
            vl=read_values(w[...,:9],v_local)
            with torch.autocast(device_type=g.device.type,enabled=False):
                vg=w[...,9:].float()@v_pool.float()
            outputs.append(mask(unflat(vl+vg.to(vl.dtype),size),keep))
        return tuple(outputs)


class FilterBasis(nn.Module):
    """Shared channelwise filter bank -> basis softmax -> channel mean."""
    def __init__(self,bases=8):
        super().__init__();self.bank=nn.Conv2d(1,bases,3,padding=1)

    def forward(self,g):
        bs,c,h,w=g.shape
        # Bound memory by processing 16 channels at a time; no detached paths.
        pieces=[]
        for chunk in g.split(16,dim=1):
            logits=self.bank(chunk.reshape(-1,1,h,w)).float()
            part=logits.softmax(1).reshape(bs,chunk.shape[1],-1,h,w).sum(1)
            pieces.append(part)
        return (sum(pieces)/c).to(g.dtype)


class ValuePreservingInterface(LocalAddress):
    def __init__(self):
        super().__init__()
        self.key=nn.Conv2d(8,32,1)
        self.structure=FilterBasis()
        self.qrho=nn.Conv2d(1,32,1,bias=False)

    def forward(self,g,a,b,valid,keep):
        size=a.shape[-2:];idx,inside=neighborhood(g.shape[-2:],size,g.device)
        keys=gather(self.key(self.structure(g)),idx).float()
        av=gather(support(keep,g.shape[-2:]),idx)[...,0]*inside
        rho=support(valid*keep,size).to(g.dtype)
        out=[]
        for i,q in enumerate((self.query_a(a),self.query_b(b)+self.qrho(rho))):
            score=(flat(q).float().unsqueeze(-2)*keys).sum(-1)/math.sqrt(32)+self.bias[i]
            w=normalized_weights(score,av)
            out.append(mask(unflat(read_values(w,gather(g,idx)),size),keep))
        return tuple(out)


class CoupledInterface(LocalAddress):
    def forward(self,g,a,b,valid,keep):
        size=a.shape[-2:];z=mask(resize(g,size),keep)
        weights,idx,av=self.address(z,a,b,keep)
        w=torch.stack(weights,1)  # B,2,N,K
        query_mask=resize(keep.float(),size,'nearest').flatten(2).transpose(1,2)[...,0]
        w=w*query_mask[:,None,:,None]
        ns=size[0]*size[1]
        indices=idx.flatten()[None].expand(g.shape[0],-1)
        # Three one-sided source-congestion updates; always end row-normalized.
        for _ in range(3):
            usage=torch.zeros(g.shape[0],ns,device=g.device,dtype=torch.float32)
            for m in range(2):
                usage=usage.scatter_add(1,indices,w[:,m].reshape(g.shape[0],-1))
            factor=(2.0/usage.clamp_min(1e-8)).clamp(max=1.)
            factor=factor[:,idx.flatten()].reshape(g.shape[0],*idx.shape)
            w=w*factor[:,None]
            w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)*query_mask[:,None,:,None]
        vals=gather(z,idx)
        outputs=[]
        for m in range(2):
            transported=unflat(read_values(w[:,m],vals),size)
            outputs.append(mask(.75*z+.25*transported,keep))
        return tuple(outputs)


class PlaneRotationInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditions=nn.Sequential(nn.Conv2d(193,32,1),nn.GELU())
        self.bases=nn.Conv2d(32,512,1)
        self.angles=nn.Conv2d(32,4,1)
        nn.init.zeros_(self.angles.weight);nn.init.zeros_(self.angles.bias)

    @staticmethod
    def rotate(z,u,v,angle):
        with torch.autocast(device_type=z.device.type,enabled=False):
            z,u,v,angle=z.float(),u.float(),v.float(),angle.float()
            u=F.normalize(u,dim=1,eps=1e-8)
            v=v-u*(u*v).sum(1,keepdim=True)
            # Deterministic orthogonal fallback for degenerate predicted bases.
            axis=u.abs().argmin(1,keepdim=True)
            e=torch.zeros_like(u).scatter_(1,axis,1.)
            fallback=e-u*(u*e).sum(1,keepdim=True)
            v=torch.where(v.square().sum(1,keepdim=True)>1e-12,v,fallback)
            v=F.normalize(v,dim=1,eps=1e-8)
            uz=(u*z).sum(1,keepdim=True);vz=(v*z).sum(1,keepdim=True)
            return z+(angle.cos()-1)*(u*uz+v*vz)+angle.sin()*(v*uz-u*vz)

    def forward(self,g,a,b,valid,keep):
        size=a.shape[-2:];z=resize(g,size);rho=support(valid*keep,size).to(a.dtype)
        c=self.conditions(torch.cat([a,b,rho],1))  # Q is independent of z.
        bases=self.bases(c).float().split(128,1)
        angles=(math.pi/6)*self.angles(c).float().tanh()
        out=[]
        for m in range(2):
            t=self.rotate(z,bases[0],bases[1],angles[:,2*m:2*m+1])
            t=self.rotate(t,bases[2],bases[3],angles[:,2*m+1:2*m+2])
            out.append(mask(t.to(g.dtype),keep))
        return tuple(out)


class CrossWriteInterface(LocalAddress):
    def __init__(self):
        super().__init__()
        self.message_a=nn.Conv2d(96,32,1,bias=False)
        self.message_b=nn.Conv2d(96,32,1,bias=False)
        self.output_a=nn.Conv2d(32,128,1,bias=False)
        self.output_b=nn.Conv2d(32,128,1,bias=False)
        nn.init.zeros_(self.output_a.weight);nn.init.zeros_(self.output_b.weight)

    @staticmethod
    def writeback(w,message,idx,source_size):
        bs,c,h,w0=message.shape;ns=source_size[0]*source_size[1]
        with torch.autocast(device_type=message.device.type,enabled=False):
            msg=flat(message).float()
            contribution=(w.float()[...,None]*msg[:,:,None,:]).permute(0,3,1,2).reshape(bs,c,-1)
            indices=idx.flatten()[None,None,:].expand(bs,c,-1)
            values=torch.zeros(bs,c,ns,device=message.device).scatter_add(2,indices,contribution)
            mass=torch.zeros(bs,1,ns,device=message.device).scatter_add(2,indices[:,:1],w.reshape(bs,1,-1).float())
            return (values/mass.clamp_min(1e-8)).reshape(bs,c,*source_size).to(message.dtype)

    def forward(self,g,a,b,valid,keep):
        size=a.shape[-2:];ws,idx,_=self.address(g,a,b,keep)
        qm=resize(keep.float(),size,'nearest').flatten(2).transpose(1,2)
        ws=[w*qm for w in ws]
        rho=support(valid*keep,size).to(b.dtype)
        ua=self.writeback(ws[0],self.message_a(a),idx,g.shape[-2:])
        ub=self.writeback(ws[1],self.message_b(b)*(.25+.75*rho),idx,g.shape[-2:])
        messages=[self.output_b(ub),self.output_a(ua)]
        return tuple(mask(unflat(read_values(w,gather(g+.25*msg,idx)),size),keep)
                     for w,msg in zip(ws,messages))


def overlap_axis(source_n,target_n,padded_extent,physical_extent,device):
    """Reference cells: RF patch centers vs CSP stride16 centers, clipped support."""
    source_stride=padded_extent/source_n
    sb=torch.arange(source_n+1,dtype=torch.float64,device=device)*source_stride
    # k3/s2/p1 local convolutions keep first center at 0.5 in pixel coordinates.
    centers=.5+16*torch.arange(target_n,dtype=torch.float64,device=device)
    tb=torch.cat([centers.new_zeros(1),(centers[:-1]+centers[1:])/2,centers.new_tensor([padded_extent])])
    sb=sb.clamp(0,physical_extent);tb=tb.clamp(0,physical_extent)
    sl,sr=sb[:-1],sb[1:];tl,tr=tb[:-1],tb[1:]
    left=torch.maximum(tl[:,None],sl[None]);right=torch.minimum(tr[:,None],sr[None])
    length=(right-left).clamp_min(0)
    midpoint=(sl+sr)/2
    moment=.5*((right-midpoint[None]).square()-(left-midpoint[None]).square())
    moment=torch.where(length>0,moment,torch.zeros_like(moment))
    area=(tr-tl).clamp_min(0)
    return (length/area[:,None].clamp_min(1e-12)).float(),(moment/area[:,None].clamp_min(1e-12)).float(),area.float(),(sr-sl).float()


class SubcellInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Sequential(nn.Conv2d(193,32,1),nn.GELU(),nn.Conv2d(32,4,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        self._geometry={}

    @staticmethod
    def slopes(g,keep):
        s=support(keep,g.shape[-2:])>0
        dx=(g[...,1:]-g[...,:-1])*(s[...,1:]&s[...,:-1])
        dy=(g[...,1:,:]-g[...,:-1,:])*(s[...,1:,:]&s[...,:-1,:])
        def minmod(left,right):
            return torch.where(left*right>0,left.sign()*torch.minimum(left.abs(),right.abs()),torch.zeros_like(left))
        return minmod(F.pad(dx,(1,0)),F.pad(dx,(0,1))),minmod(F.pad(dy,(0,0,1,0)),F.pad(dy,(0,0,0,1)))

    def geometry(self,g,size,keep):
        ih,iw=keep.shape[-2:]
        ph=int(keep[0,0].any(-1).sum());pw=int(keep[0,0].any(-2).sum())
        key=(*g.shape[-2:],*size,ih,iw,ph,pw,str(g.device))
        if key not in self._geometry:
            assert bool((keep==keep[:1]).all()), 'Subcell reference expects common rectangular padding'
            assert bool(keep[0,0,:ph,:pw].all())
            ay,my,ty,sy=overlap_axis(g.shape[-2],size[0],ih,ph,g.device)
            ax,mx,tx,sx=overlap_axis(g.shape[-1],size[1],iw,pw,g.device)
            self._geometry[key]=(ay,my,ax,mx,ty,tx,sy,sx)
        return self._geometry[key]

    def forward(self,g,a,b,valid,keep):
        size=a.shape[-2:]
        rho=support(valid*keep,g.shape[-2:]).to(g.dtype)
        gates=self.gate(torch.cat([resize(a,g.shape[-2:]),resize(b,g.shape[-2:]),rho],1)).float().sigmoid()
        ay,my,ax,mx,*_=self.geometry(g,size,keep)
        with torch.autocast(device_type=g.device.type,enabled=False):
            gf=g.float();sx,sy=self.slopes(gf,keep)
            sx=sx/(keep.shape[-1]/g.shape[-1]);sy=sy/(keep.shape[-2]/g.shape[-2])
            def integrate(field,left,right):
                return torch.einsum('yh,bchw,xw->bcyx',left,field,right)
            base=integrate(gf,ay,ax)
            outputs=[]
            for m in range(2):
                outputs.append((base+integrate(sx*gates[:,2*m:2*m+1],ay,mx)
                                +integrate(sy*gates[:,2*m+1:2*m+2],my,ax)).to(g.dtype))
        # The conservation claim applies here, BEFORE the original decoder mask.
        return tuple(outputs)


CONSTRUCTORS={'L1':ExpertInterface,'L2':LocalGlobalInterface,'L3':ValuePreservingInterface,
              'N1':CoupledInterface,'N2':PlaneRotationInterface,'N3':CrossWriteInterface,'N4':SubcellInterface}
