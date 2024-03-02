import torch
import numpy as np

def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """ 
    Compute mPJPE
    """
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :-1]
    pred = pred[has_3d_joints == 1]

    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error


def mean_per_vertex_error(pred, gt, has_smpl):
    """
    Compute mPVE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error


# --- MPJPE-PA ---
def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re
# --- end of MPJPE-PA ---

def pytorch_compute_similarity_transform(S1, S2):
    """ S1: [B, 3, N]
    """
    mu1 = torch.mean(S1, dim=2, keepdim=True)  # [B, 3, 1]
    mu2 = torch.mean(S2, dim=2, keepdim=True)  # [B, 3, 1]
    X1 = S1 - mu1    # [B, 3, N]
    X2 = S2 - mu2    # [B, 3, N]

    var1 = torch.sum(X1 ** 2, dim=[1,2])   # [B]
    K = X1 @ X2.transpose(1, 2)   # [B, 3, 3]
    U, S, Vh = torch.linalg.svd(K)   # [B, 3, 3]
    V = Vh.transpose(1, 2)   
    Z = torch.eye(S1.shape[1], device=S1.device).expand(S1.shape[0], -1, -1).contiguous()   # [B, 3, 3]
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ Vh))
    R = V @ Z @ U.transpose(1, 2)   # [B, 3, 3]

    scale = torch.diagonal(R @ K, dim1=1, dim2=2).sum(dim=1) / var1  # [B]

    t = mu2 - scale.view(-1, 1, 1)*(R @ mu1)   # [B, 3, 1]

    S1_hat = scale.view(-1, 1, 1)*(R @ S1) + t   # [B, 3, N]
    return S1_hat

def pytorch_reconstruction_error(S1, S2, reduction='mean'):
    """ S1: [B, N, 3]
    """
    S1 = S1.transpose(1, 2)  # [B, 3, N]
    S2 = S2.transpose(1, 2)  # [B, 3, N]
    S1_hat = pytorch_compute_similarity_transform(S1, S2)   # [B, 3, N]
    re = torch.sqrt(((S1_hat - S2) ** 2).sum(dim=1))  # [B, N]
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.mean(-1).sum()
    
    return re


if __name__ == '__main__':
    test_s1 = torch.randn(128, 14, 3)
    test_s2 = torch.randn(128, 14, 3)

    pytorch_result = pytorch_reconstruction_error(test_s1, test_s2, 'mean')
    numpy_result = reconstruction_error(test_s1.numpy(), test_s2.numpy(), 'mean')
    print(pytorch_result.item(), numpy_result, pytorch_result.item()-numpy_result)



