"""Your five exercises. Read the PDF before opening reference/core.py.

Conventions: row-stored points, camera +z forward / +y up, image +v down.
All tensors must preserve their device and dtype. No numpy/detach in these functions.
"""
import torch


def covariance(scales, rotations):
    """scales [N,3] positive; rotations [N,3,3]; return [N,3,3].
    TODO 1: Sigma = R diag(scales**2) R.T (transpose the LAST two axes).
    """
    # Start from an axis-aligned ellipsoid, then rotate it into world space.
    # Eigenvalues are squared scales, so the result is positive-definite.
    D = torch.diag_embed(scales.square())
    return rotations @ D @ rotations.transpose(-1, -2)


def project(means, cov_world, W, eye, fx, fy, cx, cy, blur):
    """Return uv [N,2], cov2 [N,2,2], z [N].
    W [3,3] maps world directions to camera directions; eye [3].
    Caller culls means behind the near plane first. blur is a pixel stddev.
    TODO 2: camera means/covariances, perspective means, Jacobian, cov2.
    """
    # Translate first, then rotate axes. Translation shifts the center but not
    # the spread around it, so covariance only sees the rotation W.
    cam = (means - eye) @ W.T                      # [N,3]
    cov_cam = W @ cov_world @ W.T                  # [N,3,3], W broadcasts
    x, y, z = cam.unbind(-1)

    # Perspective divide. The minus on v flips camera-up to image-down.
    uv = torch.stack((fx * x / z + cx, cy - fy * y / z), -1)

    # J[out, in]: how (u,v) respond to a small change in (x,y,z).
    # The z column is the depth term; dropping it leaves off-axis splats wrong.
    zero = torch.zeros_like(z)
    J = torch.stack((fx / z, zero, -fx * x / z.square(),
                     zero, -fy / z, fy * y / z.square()), -1).reshape(-1, 2, 3)

    # Local linear approximation: a linear map A sends Sigma to A Sigma A.T.
    # The blur term is a minimum screen footprint, not real antialiasing.
    I2 = torch.eye(2, device=cam.device, dtype=cam.dtype)
    cov2 = J @ cov_cam @ J.transpose(-1, -2) + blur**2 * I2
    return uv, cov2, z


def pixel_weights(uv, cov2, pixels):
    """uv [N,2], cov2 [N,2,2], pixels [P,2]; return weights [N,P].
    TODO 3: exp(-0.5 * d.T @ inverse(cov2) @ d). Peak is 1.
    """
    d = pixels[None, :, :] - uv[:, None, :]        # [N,P,2] offset per pair
    # q is squared distance measured in units of the ellipse's own spread.
    q = torch.einsum('npi,nij,npj->np', d, torch.linalg.inv(cov2), d)
    return torch.exp(-.5 * q)                      # peak 1, not a density


def composite(alpha, colors, background):
    """alpha [N,P] already sorted near-to-far, colors [N,3], bg [3].
    Return [P,3]. Use EXCLUSIVE transmittance before each splat.
    TODO 4: T, weights T*alpha, RGB sum, remaining background.
    """
    # cumprod of (1-alpha) with a leading 1: row i is what reached splat i,
    # counting only the splats in front of it. The extra last row is the
    # fraction of background still visible.
    T = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1 - alpha)), 0)
    return (T[:-1] * alpha).T @ colors + T[-1, :, None] * background


def update(prediction, target, optimizer):
    """TODO 5: clear gradients, mean squared image error, backward, step.
    Return the scalar loss. Do not detach before backward.
    """
    optimizer.zero_grad(set_to_none=True)          # Adam accumulates otherwise
    loss = (prediction - target).square().mean()
    loss.backward()                                # fills .grad for every leaf
    optimizer.step()                               # then Adam moves parameters
    return loss
