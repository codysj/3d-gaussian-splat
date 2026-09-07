"""Supplied wiring and scene helpers. Read these before claiming implementation ownership."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def quaternion_matrix(q):
    """Quaternion order (w,x,y,z); normalize before conversion."""
    q = F.normalize(q, dim=-1, eps=1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1,3,3)


def camera(angle, device='cpu', dtype=torch.float32, elevation=0.35):
    eye = torch.tensor([3*math.sin(angle), elevation, -3*math.cos(angle)],
                       device=device, dtype=dtype)
    target = torch.tensor([0.,0.05,0.], device=device, dtype=dtype)
    forward = F.normalize(target-eye, dim=0)
    up = torch.tensor([0.,1.,0.], device=device, dtype=dtype)
    right = F.normalize(torch.linalg.cross(up, forward), dim=0)
    up = torch.linalg.cross(forward, right)
    return torch.stack((right,up,forward)), eye


def pixels_grid(h, w, device, dtype):
    y,x = torch.meshgrid(torch.arange(h,device=device,dtype=dtype),
                         torch.arange(w,device=device,dtype=dtype), indexing='ij')
    return torch.stack((x,y),-1).reshape(-1,2)


def render(params, W, eye, size, core, order='depth', mode='dense'):
    """Dense differentiable path; bounded forward-only path for footprint exercise.
    Empty/culled scenes return background. Center-depth sorting is approximate.
    """
    means, scales, quats, opacity, colors = params
    bg = means.new_tensor([0.035,0.045,0.065])
    depth = ((means-eye) @ W.T)[:,2]
    keep = depth > 0.1
    means, scales, quats = means[keep], scales[keep], quats[keep]
    opacity, colors = opacity[keep], colors[keep]
    if means.shape[0] == 0:
        return bg.expand(size,size,3).clone()
    cov = core.covariance(scales, quaternion_matrix(quats))
    uv, cov2, z = core.project(means,cov,W,eye,.9*size,.9*size,
                               (size-1)/2,(size-1)/2,.3*size/256)
    idx = torch.argsort(z)
    if order == 'reverse': idx = idx.flip(0)
    if order == 'input': idx = torch.arange(len(z),device=z.device)
    uv,cov2,opacity,colors = uv[idx],cov2[idx],opacity[idx],colors[idx]
    if mode == 'bounded':
        if torch.is_grad_enabled():
            raise RuntimeError('Bounded path is forward-only; use torch.no_grad().')
        return bounded_forward(uv,cov2,opacity,colors,bg,size,core)
    pixels = pixels_grid(size,size,means.device,means.dtype)
    g = core.pixel_weights(uv,cov2,pixels)
    if mode == 'cutoff': g = g * (g >= math.exp(-4.5))
    alpha = (opacity[:,None]*g).clamp(max=.99)
    return core.composite(alpha,colors,bg).reshape(size,size,3)


def bounded_forward(uv,cov2,opacity,colors,bg,size,core):
    """Supplied reference for a 3-sigma bounded renderer; no backward support.
    Rewrite after the dense path works. Compare against mode='cutoff'.
    """
    C = bg.new_zeros(size,size,3)
    T = bg.new_ones(size,size)
    for i in range(len(uv)):
        extent = 3*torch.sqrt(torch.diagonal(cov2[i]))
        low = torch.floor(uv[i]-extent).long()
        high = torch.ceil(uv[i]+extent).long()+1
        x0,y0 = max(0,int(low[0])),max(0,int(low[1]))
        x1,y1 = min(size,int(high[0])),min(size,int(high[1]))
        if x0 >= x1 or y0 >= y1: continue
        pix = pixels_grid(y1-y0,x1-x0,uv.device,uv.dtype)
        pix = pix + pix.new_tensor([x0,y0])
        g = core.pixel_weights(uv[i:i+1],cov2[i:i+1],pix)[0]
        g = g * (g >= math.exp(-4.5))
        a = (opacity[i]*g).clamp(max=.99).reshape(y1-y0,x1-x0)
        patchT = T[y0:y1,x0:x1]
        C[y0:y1,x0:x1] += (patchT*a)[...,None]*colors[i]
        T[y0:y1,x0:x1] = patchT*(1-a)
    return C+T[...,None]*bg


def teacher_scene():
    """A stylized robot made of 31 ellipsoids. Synthetic, not a photograph."""
    rows=[]
    def add(p,s,c,rz=0):
        rows.append((p,s,[math.cos(rz/2),0,0,math.sin(rz/2)],.94,c))
    blue=[.08,.52,.85]; orange=[.95,.4,.08]; light=[.85,.9,.98]
    add([0,.12,0],[.26,.32,.14],blue)
    add([0,.65,0],[.23,.19,.15],orange)
    add([0,.41,0],[.075,.09,.08],light)
    for x in [-.09,.09]:
        add([x,.68,-.145],[.037,.047,.025],[.05,.08,.13])
    add([0,.56,-.145],[.09,.018,.018],light)
    add([0,.91,0],[.015,.09,.015],light)
    add([0,1.01,0],[.05,.05,.05],orange)
    for sign in [-1,1]:
        for j in range(3):
            add([sign*(.33+.07*j),.28-.17*j,0],[.075,.13,.075],blue,sign*.35)
        add([sign*.53,-.15,0],[.095,.09,.08],orange)
        for j in range(3):
            add([sign*.15,-.3-.17*j,0],[.095,.13,.09],blue)
        add([sign*.15,-.8,-.06],[.13,.07,.17],orange)
    for y in [-.02,.1,.22]:
        add([0,y,-.14],[.038,.038,.022],light)
    # Side/back details make a turntable more informative.
    for x in [-.17,0,.17]:
        add([x,.12,.14],[.055,.17,.04],orange)
    add([0,-.87,0],[.53,.028,.32],[.23,.27,.34])
    return tuple(np.asarray([r[i] for r in rows],dtype=np.float32) for i in range(5))


class Scene(nn.Module):
    def __init__(self, arrays, device='cpu', dtype=torch.float32):
        super().__init__()
        m,s,q,o,c = [torch.as_tensor(x,device=device,dtype=dtype) for x in arrays]
        self.means=nn.Parameter(m.clone())
        self.log_scales=nn.Parameter(s.log().clone())
        self.quats=nn.Parameter(q.clone())
        self.opacity_logits=nn.Parameter(torch.logit(o.clamp(.001,.999)))
        self.color_logits=nn.Parameter(torch.logit(c.clamp(.001,.999)))

    def physical(self):
        return (self.means,self.log_scales.exp(),self.quats,
                self.opacity_logits.sigmoid(),self.color_logits.sigmoid())

    def groups(self, all_params=True, factor=1.):
        groups=[{'params':[self.means],'lr':.003*factor},
                {'params':[self.color_logits],'lr':.03*factor}]
        for p,lr in [(self.log_scales,.006),(self.quats,.003),
                     (self.opacity_logits,.01)]:
            p.requires_grad_(all_params)
            if all_params: groups.append({'params':[p],'lr':lr*factor})
        return groups
