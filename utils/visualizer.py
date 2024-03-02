import pickle
import torch
import numpy as np
from utils.pytorch3d_renderer import Pt3dRenderer
from utils.draw_2d_joints import draw_skeleton

class Visualizer:
    def __init__(self, path_to_smpl, img_size, device, mesh_color=(0.93, 0.68, 0.62)):
        smpl_meta_dict = pickle.load(open(path_to_smpl,'rb'),encoding='latin1')
        faces = torch.tensor(smpl_meta_dict['f'].astype(np.int64)).unsqueeze(0).long()
        self.img_size = img_size
        self.mesh_renderer = Pt3dRenderer(faces=faces, device=device, img_size=img_size)
        self.mesh_color = mesh_color

    @torch.no_grad()
    def draw_2d_joints(self, gt_keypoints_2d, pred_keypoints_2d, images, num_draws):
        gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
        pred_keypoints_2d = pred_keypoints_2d.cpu().numpy()
        to_lsp = list(range(14))
        skel_list_gt = []
        skel_list_pred = []

        # Do visualization for the first 6 images of the batch
        for i in range(num_draws):
            img = images[i].cpu().numpy().transpose(1,2,0)
            # Get LSP keypoints from the full list of keypoints
            gt_kp = gt_keypoints_2d[i, to_lsp]
            pred_kp = pred_keypoints_2d[i, to_lsp]

            gt_vis = gt_kp[:, 2].astype(bool)
            gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * (self.img_size-1)
            pred_joint = ((pred_kp + 1) * 0.5) * (self.img_size-1)

            skel_img_gt = draw_skeleton(img, gt_joint, draw_edges=True, vis=gt_vis)
            skel_img_pred = draw_skeleton(img, pred_joint)
            # img_with_gt = draw_skeleton(img, gt_joint, draw_edges=False, vis=gt_vis)
            # skel_img = draw_skeleton(img_with_gt, pred_joint)
            skel_list_gt.append(skel_img_gt)
            skel_list_pred.append(skel_img_pred)
            
        return skel_list_gt, skel_list_pred

    @torch.no_grad()
    def draw_mesh(self, gt_vertices, pred_vertices, gt_cam, pred_cam, images, num_draws):
        gt_vertices = gt_vertices[:num_draws]
        pred_vertices = pred_vertices[:num_draws]
        gt_cam = gt_cam[:num_draws]
        pred_cam = pred_cam[:num_draws]
        gt_rends = self.mesh_renderer.render(gt_vertices, gt_cam, color=(0.63, 0.93, 0.88))
        pred_rends = self.mesh_renderer.render(pred_vertices, pred_cam, color=self.mesh_color)  # (0.93, 0.68, 0.62)
        
        images = images[:num_draws].permute(0, 2, 3, 1)
        gt_rends = gt_rends[..., 3:] * gt_rends[..., :3] + (1-gt_rends[..., 3:]) * images
        pred_rends = pred_rends[..., 3:] * pred_rends[..., :3] + (1-pred_rends[..., 3:]) * images
        
        return gt_rends.cpu().numpy(), pred_rends.cpu().numpy()

    @torch.no_grad()
    def draw_skeleton_and_mesh(self, images, gt_keypoints_2d, pred_keypoints_2d, gt_vertices, pred_vertices, gt_cam, pred_cam, num_draws=1):
        batch_size = gt_keypoints_2d.shape[0]
        num_draws = min(batch_size, num_draws)
        skel_list_gt, skel_list_pred = self.draw_2d_joints(gt_keypoints_2d, pred_keypoints_2d, images, num_draws)
        gt_rends, pred_rends = self.draw_mesh(gt_vertices, pred_vertices, gt_cam, pred_cam, images, num_draws)
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()

        rend_imgs = []
        for i in range(num_draws):
            # rend_img = np.hstack([images_np[i], skel_list_pred[i], skel_list_gt[i], pred_rends[i], gt_rends[i]])
            rend_img = np.hstack([images_np[i], skel_list_pred[i], pred_rends[i]])
            rend_imgs.append(rend_img)
        # import ipdb; ipdb.set_trace()
        combine = np.vstack(rend_imgs)
        return combine

    @torch.no_grad()
    def draw_pred_mesh(self, pred_vertices, pred_cam, images, num_draws, mesh_color=None):
        if mesh_color is None:
            mesh_color = self.mesh_color
        
        pred_vertices = pred_vertices[:num_draws]
        pred_cam = pred_cam[:num_draws]
        pred_rends = self.mesh_renderer.render(pred_vertices, pred_cam, color=mesh_color)  # (0.93, 0.68, 0.62)
        
        images = images[:num_draws].permute(0, 2, 3, 1)
        pred_rends = pred_rends[..., 3:] * pred_rends[..., :3] + (1-pred_rends[..., 3:]) * images
        
        images_np = images.cpu().numpy()
        pred_rends = pred_rends.cpu().numpy()

        rend_imgs = []
        for i in range(num_draws):
            rend_img = np.hstack([images_np[i], pred_rends[i]])
            rend_imgs.append(rend_img)
        
        return rend_imgs