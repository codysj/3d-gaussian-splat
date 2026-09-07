"""Supplied CLI. Run from this directory; default imports YOUR core.py."""
import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from support import Scene, camera, render, teacher_scene


def save_image(path, x):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    Image.fromarray((np.clip(x,0,1)*255).round().astype('uint8')).save(path)


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def load_data(path, size, device):
    with np.load(path) as d:
        images=torch.tensor(d['images'],device=device)
        angles=d['angles'].copy()
        train=d['train'].tolist(); test=d['test'].tolist()
        arrays=tuple(d['init_'+str(i)].copy() for i in range(5))
    # The target at 512 is fixed. Smaller resolutions use area downsampling.
    images=F.interpolate(images.permute(0,3,1,2),size=(size,size),
                         mode='area').permute(0,2,3,1)
    return images, angles, train, test, arrays


def make(args,core):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    # --scene: five arrays from scene_from_mesh.py instead of the built-in robot.
    arrays=teacher_scene() if not args.scene else tuple(
        np.load(args.scene)[k] for k in ('means','scales','quats','opacity','colors'))
    truth=Scene(arrays,args.device)
    angles=np.linspace(0,2*math.pi,12,endpoint=False).astype('float32')
    frames=[]
    with torch.no_grad():
        for i,a in enumerate(angles):
            W,eye=camera(float(a),args.device)
            im=render(truth.physical(),W,eye,args.size,core)
            frames.append(im.cpu().numpy())
            save_image(out/f'target_{i:02d}.png',im)
    rng=np.random.default_rng(17)
    init=[a.copy() for a in arrays]
    init[0]+=rng.normal(0,.045,init[0].shape).astype('float32')
    init[1]*=np.exp(rng.normal(0,.12,init[1].shape)).astype('float32')
    init[2]+=rng.normal(0,.06,init[2].shape).astype('float32')
    init[3]=np.full_like(init[3],.8)
    init[4]=np.clip(init[4]+rng.normal(0,.12,init[4].shape),.03,.97).astype('float32')
    test=[1,4,7,10]; train=[i for i in range(12) if i not in test]
    payload={f'init_{i}':a for i,a in enumerate(init)}
    np.savez_compressed(out/'dataset.npz',images=np.stack(frames),angles=angles,
                        train=train,test=test,**payload)
    print(f'Wrote {out}/dataset.npz: {len(arrays[0])} Gaussians, 8 train + 4 held-out views.')
    print('Known-count, perturbed-ground-truth initialization. Targets are synthetic.')


def evaluate(scene,images,angles,indices,core,size,device):
    with torch.no_grad():
        losses=[]
        for i in indices:
            W,eye=camera(float(angles[i]),device)
            pred=render(scene.physical(),W,eye,size,core)
            losses.append(float((pred-images[i]).square().mean()))
    return float(np.mean(losses))


def save_checkpoint(path,scene,opt,step,metadata):
    torch.save({'model':scene.state_dict(),'optimizer':opt.state_dict(),
                'step':step,'metadata':metadata},path)


def train(args,core):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    images,angles,train_ids,test_ids,arrays=load_data(args.data,args.size,args.device)
    if args.views=='one': train_ids=[train_ids[0]]
    scene=Scene(arrays,args.device)
    if args.resume:
        ckpt=torch.load(args.resume,map_location=args.device,weights_only=True)
        scene.load_state_dict(ckpt['model'])
    opt=torch.optim.Adam(scene.groups(args.learn=='all',args.lr_factor))
    # Resume loads model only: deliberate new Adam state for resolution refinement.
    meta=vars(args).copy()
    meta.update(torch_version=str(torch.__version__),python=platform.python_version(),
                gpu=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU',
                gaussian_count=len(arrays[0]),train_indices=train_ids,test_indices=test_ids,
                source='reference' if args.reference else 'learner core.py',
                initialization='known count + perturbed target parameters')
    (out/'config.json').write_text(json.dumps(meta,indent=2))
    before=evaluate(scene,images,angles,test_ids,core,args.size,args.device)
    show=test_ids[0]; W,eye=camera(float(angles[show]),args.device)
    with torch.no_grad(): save_image(out/'before.png',render(scene.physical(),W,eye,args.size,core))
    save_image(out/'target.png',images[show])
    rows=[]; sync(args.device); start=time.perf_counter()
    for step in range(args.steps):
        i=train_ids[step % len(train_ids)]
        W,eye=camera(float(angles[i]),args.device)
        pred=render(scene.physical(),W,eye,args.size,core)
        loss=core.update(pred,images[i],opt)
        for name,p in scene.named_parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise RuntimeError(f'Missing/non-finite gradient: {name}')
        value=float(loss.detach())
        rows.append([step+1,i,value])
        if (step+1)%100==0 or step==0:
            print(f'update {step+1:5d} view {i:2d} MSE {value:.6f}',flush=True)
            save_checkpoint(out/'checkpoint.pt',scene,opt,step+1,meta)
    sync(args.device); elapsed=time.perf_counter()-start
    save_checkpoint(out/'checkpoint.pt',scene,opt,args.steps,meta)
    after=evaluate(scene,images,angles,test_ids,core,args.size,args.device)
    training=evaluate(scene,images,angles,train_ids,core,args.size,args.device)
    summary={'training_MSE':training,'held_out_MSE_before':before,
             'held_out_MSE_after':after,'loop_seconds_including_logs':elapsed,
             'updates':args.steps,'held_out_PSNR':-10*math.log10(max(after,1e-12))}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    with (out/'loss.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['update','view','MSE']);writer.writerows(rows)
    fig,ax=plt.subplots(figsize=(6,3))
    ax.plot([r[0] for r in rows],[r[2] for r in rows],lw=.8)
    ax.set(xlabel='Update (camera changes)',ylabel='Training-view MSE',yscale='log')
    fig.tight_layout();fig.savefig(out/'loss.png',dpi=180);plt.close(fig)
    with torch.no_grad():
        for i in test_ids:
            W,eye=camera(float(angles[i]),args.device)
            save_image(out/f'heldout_{i:02d}.png',render(scene.physical(),W,eye,args.size,core))
    print(json.dumps(summary,indent=2))


def load_scene(args):
    # Only open trusted checkpoints you created yourself.
    ckpt=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    sd=ckpt['model']
    arrays=(sd['means'],sd['log_scales'].exp(),sd['quats'],
            sd['opacity_logits'].sigmoid(),sd['color_logits'].sigmoid())
    return Scene(arrays,args.device)


def export(args,core):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    scene=load_scene(args);frames=[]
    with torch.no_grad():
        for j,a in enumerate(np.linspace(0,2*math.pi,args.frames,endpoint=False)):
            W,eye=camera(float(a),args.device)
            im=render(scene.physical(),W,eye,args.size,core)
            save_image(out/f'frame_{j:03d}.png',im)
            frames.append(Image.open(out/f'frame_{j:03d}.png').convert('RGB'))
    frames[0].save(out/'orbit.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    print(f'Exported {len(frames)} frames and orbit.gif to {out}')


def eval_checkpoint(args,core):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    images,angles,train_ids,test_ids,_=load_data(args.data,args.size,args.device)
    scene=load_scene(args)
    error=evaluate(scene,images,angles,test_ids,core,args.size,args.device)
    summary={'evaluation_size':args.size,'held_out_MSE':error,
             'checkpoint':args.checkpoint,'test_indices':test_ids}
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
    with torch.no_grad():
        for i in test_ids:
            W,eye=camera(float(angles[i]),args.device)
            pred=render(scene.physical(),W,eye,args.size,core)
            save_image(out/f'heldout_{i:02d}.png',pred)
            save_image(out/f'target_{i:02d}.png',images[i])
    print(json.dumps(summary,indent=2))


def benchmark(args,core):
    scene=Scene(teacher_scene(),args.device)
    opt=torch.optim.Adam(scene.parameters(),lr=0.)
    W,eye=camera(0.,args.device)
    def update():
        opt.zero_grad(set_to_none=True)
        pred=render(scene.physical(),W,eye,args.size,core)
        pred.square().mean().backward();opt.step()
    for _ in range(5): update()
    sync(args.device)
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter()
    for _ in range(args.steps): update()
    sync(args.device);dt=(time.perf_counter()-t)/args.steps
    print(json.dumps({'size':args.size,'gaussians':len(scene.means),
       'seconds_per_update':dt,'estimated_2000_update_minutes':2000*dt/60,
       'peak_allocated_MB':torch.cuda.max_memory_allocated()/1e6 if args.device=='cuda' else None},indent=2))


def check(args,core):
    dtype=torch.float64; device=args.device
    def t(x):return torch.tensor(x,device=device,dtype=dtype)
    stage=args.stage
    if stage in ['all','covariance']:
        R=t([[[0,-1,0],[1,0,0],[0,0,1]]])
        cov=core.covariance(t([[2,1,1]]),R)
        torch.testing.assert_close(cov,t([[[1,0,0],[0,4,0],[0,0,1]]]))
        assert bool((torch.linalg.eigvalsh(cov)>0).all())
        print('PASS covariance: rotated axes, positive eigenvalues')
    if stage in ['all','projection']:
        cov=torch.eye(3,device=device,dtype=dtype)[None]*.01
        uv,c2,z=core.project(t([[0,0,2],[0,0,4]]),cov.expand(2,3,3),
                         torch.eye(3,device=device,dtype=dtype),t([0,0,0]),100,100,32,32,0.)
        torch.testing.assert_close(uv,t([[32,32],[32,32]]))
        torch.testing.assert_close(c2[0],c2[1]*4)
        # Off-axis covariance includes depth uncertainty through J.
        _,off,_=core.project(t([[1,0,2]]),cov,torch.eye(3,device=device,dtype=dtype),
                            t([0,0,0]),100,100,32,32,0.)
        torch.testing.assert_close(off[0,0,0],t(31.25))
        print('PASS projection: center, distance scaling, off-axis depth term')
    if stage in ['all','weights']:
        g=core.pixel_weights(t([[0,0]]),t([[[4,0],[0,1]]]),t([[0,0],[2,0],[0,1]]))
        torch.testing.assert_close(g,t([[1,math.exp(-.5),math.exp(-.5)]]))
        print('PASS weights: peak and equal standardized distance')
    if stage in ['all','blend']:
        rgb=core.composite(t([[.5],[.5]]),t([[1,0,0],[0,0,1]]),t([0,0,0]))
        torch.testing.assert_close(rgb,t([[.5,0,.25]]))
        print('PASS blend: red .5 + blue .25; .25 background remains')
    if stage in ['all','gradient']:
        arrays=(np.array([[.08,.05,0.]]),np.array([[.18,.12,.1]]),
                np.array([[1.,0,0,0]]),np.array([.7]),np.array([[.8,.3,.1]]))
        s=Scene(arrays,device,dtype)
        W,eye=camera(0.,device,dtype)
        def loss():return render(s.physical(),W,eye,20,core).square().mean()
        loss().backward();analytic=float(s.means.grad[0,0]);eps=1e-5
        original=float(s.means[0,0].detach())
        with torch.no_grad():
            s.means[0,0]=original+eps;plus=float(loss())
            s.means[0,0]=original-eps;minus=float(loss())
            s.means[0,0]=original
        numeric=(plus-minus)/(2*eps)
        assert abs(analytic-numeric)<1e-7+1e-3*abs(numeric),(analytic,numeric)
        print(f'PASS gradient: autograd={analytic:.8g}, finite difference={numeric:.8g}')
    if stage=='all':
        s=Scene(teacher_scene(),device);W,eye=camera(.3,device)
        with torch.no_grad():
            a=render(s.physical(),W,eye,64,core,mode='cutoff')
            b=render(s.physical(),W,eye,64,core,mode='bounded')
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        print('PASS bounded footprint: agrees with dense path using same cutoff')


def ablate(args,core):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    scene=load_scene(args);W,eye=camera(0.,args.device)
    p=scene.physical()
    with torch.no_grad():
        save_image(out/'sorted.png',render(p,W,eye,args.size,core))
        save_image(out/'reversed.png',render(p,W,eye,args.size,core,order='reverse'))
        s=p[1].log().mean(-1,keepdim=True).exp().expand_as(p[1])
        isotropic=(p[0],s,p[2],p[3],p[4])
        save_image(out/'isotropic_without_retraining.png',render(isotropic,W,eye,args.size,core))
        save_image(out/'bounded_3sigma.png',render(p,W,eye,args.size,core,mode='bounded'))
    print('Wrote ablations. Isotropic change preserves each ellipsoid volume; no retraining.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['make','check','train','bench','export','ablate','eval'])
    parser.add_argument('--reference',action='store_true',help='Use supplied answers, not learner core')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--out',default='outputs/run')
    parser.add_argument('--data',default='outputs/data/dataset.npz')
    parser.add_argument('--learn',choices=['basic','all'],default='all')
    parser.add_argument('--views',choices=['one','multi'],default='multi')
    parser.add_argument('--lr-factor',type=float,default=1.)
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint',default='outputs/main/checkpoint.pt')
    parser.add_argument('--frames',type=int,default=36)
    parser.add_argument('--scene',help='npz from scene_from_mesh.py; make only')
    parser.add_argument('--stage',choices=['all','covariance','projection','weights','blend','gradient'],default='all')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.size<4 or args.steps<1 or args.frames<1: parser.error('Invalid size/steps/frames')
    torch.set_num_threads(args.threads);torch.manual_seed(17)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check RunPod template; CPU wheel cannot use NVIDIA GPU.')
    core=importlib.import_module('reference.core' if args.reference else 'core')
    print('CORE:', 'supplied reference' if args.reference else 'your core.py',flush=True)
    {'make':make,'check':check,'train':train,'bench':benchmark,
     'export':export,'ablate':ablate,'eval':eval_checkpoint}[args.command](args,core)


if __name__=='__main__':main()
