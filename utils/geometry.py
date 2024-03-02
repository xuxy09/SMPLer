import torch
from torch.nn import functional as F
import numpy as np


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Note: This is actually weak perspective projection: s*(X+t)
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """ 
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.reshape(shape[0], -1)).view(shape)
    return X_2d


def compute_weak_perspective_cam(joints_3d, joints_2d, conf):
    """ joints_3d: [B, K, 3]
        joints_2d: [B, K, 2]
        conf: [B, K]
    """
    B, K = joints_3d.shape[0:2]
    b = torch.cat([conf*joints_2d[..., 0], conf*joints_2d[..., 1]], dim=1)   # [B, 2K]
    ones = torch.ones(B, K, device=joints_3d.device)
    zeros = torch.zeros(B, K, device=joints_3d.device)
    A1 = conf.unsqueeze(-1) * torch.stack([joints_3d[..., 0], ones, zeros], dim=-1)
    A2 = conf.unsqueeze(-1) * torch.stack([joints_3d[..., 1], zeros, ones], dim=-1)
    A = torch.cat([A1, A2], dim=1)   # [B, 2K, 3]
    A_ = torch.bmm(A.transpose(1, 2), A)  # [B, 3, 3]
    b_ = torch.bmm(A.transpose(1, 2), b.unsqueeze(-1))   # [B, 3, 1]
    cam = torch.bmm(torch.inverse(A_), b_).squeeze(-1)   # [B, 3]
    cam[:, 1:] = cam[:, 1:] / cam[:, 0:1]
    return cam


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat   


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


def batch_rodrigues_v2(theta):
    """ theta: [B, K*3]
        output: [B, K, 3, 3]
    """
    B = theta.shape[0]
    theta = theta.reshape(-1, 3)  # [B*K, 3]
    theta_mat = batch_rodrigues(theta)  # [B*K, 3, 3]
    return theta_mat.reshape(B, -1, 3, 3)   # [B, K, 3, 3]