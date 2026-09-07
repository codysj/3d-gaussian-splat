"""Reference answers supplied with the workbook, not the learner's original work."""
import torch


def covariance(scales, rotations):
    S = torch.diag_embed(scales.square())
    return rotations @ S @ rotations.transpose(-1, -2)


def project(means, cov_world, W, eye, fx, fy, cx, cy, blur):
    cam = (means - eye) @ W.T
    x, y, z = cam.unbind(-1)
    uv = torch.stack((fx*x/z + cx, cy - fy*y/z), dim=-1)
    zero = torch.zeros_like(z)
    J = torch.stack((fx/z, zero, -fx*x/z.square(),
                     zero, -fy/z, fy*y/z.square()), dim=-1)
    J = J.reshape(-1, 2, 3)
    cov_cam = W @ cov_world @ W.T
    cov2 = J @ cov_cam @ J.transpose(-1, -2)
    I = torch.eye(2, device=means.device, dtype=means.dtype)
    return uv, cov2 + blur**2 * I, z


def pixel_weights(uv, cov2, pixels):
    d = pixels[None, :, :] - uv[:, None, :]
    inv = torch.linalg.inv(cov2)
    q = torch.einsum('npi,nij,npj->np', d, inv, d)
    return torch.exp(-0.5*q)


def composite(alpha, colors, background):
    one = torch.ones_like(alpha[:1])
    T = torch.cumprod(torch.cat((one, 1-alpha), dim=0), dim=0)
    return (T[:-1]*alpha).T @ colors + T[-1, :, None]*background


def update(prediction, target, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss=(prediction-target).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss; see bug checklist.')
    loss.backward()
    optimizer.step()
    return loss
