"""Turn a textured mesh into an initial Gaussian scene: the same five arrays
teacher_scene() returns, saved as .npz for `run.py make --scene`.

    python scene_from_mesh.py assets/garden_gnome/garden_gnome_1k.gltf --count 6000 --out assets/gnome_6000.npz

Each sampled surface point becomes one small isotropic Gaussian carrying the
texture colour at that point. Training then adjusts everything, including shape.
"""
import argparse

import numpy as np
import trimesh

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mesh')
p.add_argument('--count', type=int, default=6000)
p.add_argument('--height', type=float, default=1.8, help='object height in scene units')
p.add_argument('--out', required=True)
args = p.parse_args()

mesh = trimesh.load(args.mesh, force='mesh')
# Fit the fixed camera orbit in support.camera(): it looks at (0, .05, 0) from
# radius 3, and the robot it was built for is about 1.8 units tall.
mesh.apply_translation(-mesh.bounds.mean(0))
mesh.apply_scale(args.height / mesh.extents[1])
mesh.apply_translation([0, .05, 0])

mesh.visual.material = mesh.visual.material.to_simple()  # glTF PBR -> plain image texture
points, face, colour = trimesh.sample.sample_surface(mesh, args.count, sample_color=True)
spacing = np.sqrt(mesh.area / args.count)  # mean distance between neighbouring samples
# ponytail: isotropic splats, radius ~ one sample spacing so neighbours overlap.
# Flatten along the face normal if training leaves visible gaps.
scales = np.full((args.count, 3), .8 * spacing)
quats = np.tile([1., 0, 0, 0], (args.count, 1))
opacity = np.full(args.count, .9)
colours = colour[:, :3] / 255.

np.savez(args.out, **{k: np.asarray(v, dtype=np.float32) for k, v in
         dict(means=points, scales=scales, quats=quats, opacity=opacity, colors=colours).items()})
print(f'{args.count} Gaussians, splat radius {.8 * spacing:.4f}, '
      f'bounds y {points[:, 1].min():.2f}..{points[:, 1].max():.2f}, '
      f'mean colour {colours.mean(0).round(2)} -> {args.out}')
