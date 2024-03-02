import os
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
import tqdm
import glob
import json
from datetime import datetime, timedelta
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from utils.geometry import batch_rodrigues_v2, compute_weak_perspective_cam
from utils.eval_metrics import mean_per_joint_position_error, mean_per_vertex_error, reconstruction_error, pytorch_reconstruction_error

import config
import constants
from utils.visualizer import Visualizer
from models.transformer_basics import TranformerConfig
from models.SMPL_handler import SMPLHandler
from data_functions.human_mesh_tsv import MeshTSVYamlDataset

from utils import ddp_utils, misc
from easydict import EasyDict

class Trainer:
    def __init__(self, args):
        # meta
        self.args = args
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_only = args.eval_only
        if not self.eval_only:
            self.log_dir = self.args.log_dir
            self.checkpoint_dir = self.args.checkpoint_dir

        # ddp: init
        self.args.ddp = EasyDict(dist_url='env://')
        ddp_utils.init_distributed_mode(self.args.ddp)
        misc.set_seed(self.args.seed + ddp_utils.get_rank(), False, True)

        # data
        if self.args.data_mode == 'h36m':
            if not self.eval_only:
                self.train_dataset = MeshTSVYamlDataset(config.H36m_coco40k_Muco_UP_Mpii_yaml, True, False, 1)   
            self.val_dataset = MeshTSVYamlDataset(config.H36m_val_p2_yaml, False, False, 1)
        elif self.args.data_mode == '3dpw':
            if not self.eval_only:
                self.train_dataset = MeshTSVYamlDataset(config.PW3D_train_yaml, True, False, 1)   
            self.val_dataset = MeshTSVYamlDataset(config.PW3D_val_yaml, False, False, 1)
        else:
            raise Exception('Unknown data mode {}'.format(self.args.data_mode))
        
        if not self.eval_only:
            print('Dataset finished: train-{}, test-{}'.format(len(self.train_dataset), len(self.val_dataset)))
        else:
            print('Dataset finished: test-{}'.format(len(self.val_dataset)))

        # model
        trans_cfg = TranformerConfig()
        trans_cfg.raw_feat_dim = config.hrnet_dict[args.hrnet_type][2]
        
        
        if args.model_type == 'backbone':
            from models.baseline import BaselineModel
            self.model = BaselineModel(args)
            find_unused_parameters = True
        elif args.model_type == 'smpler':
            from  models.smpler import SMPLer
            self.model = SMPLer(args, trans_cfg)
            find_unused_parameters = False 
        else:
            raise NotImplementedError('model type {}'.format(args.model_type))

        if ddp_utils.is_dist_avail_and_initialized():
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params: {:.1f}M'.format(n_parameters/(1024**2)))

        self.model.to(self.device)   # put it after ddp initialization, but before auto_load

        # load weights
        if self.args.load_checkpoint is not None:
            self.load_checkpoint(self.model, self.args.load_checkpoint, edit_state_dict=1)

        num_tasks = ddp_utils.get_world_size()
        global_rank = ddp_utils.get_rank()
        if not self.eval_only:
            self.global_iter = 0
            self.start_epoch = 0
            self.optimizer = self.prepare_optimizer()
            self.auto_load()
        
            # ddp: sampler 
            sampler_train = DistributedSampler(self.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=False)
            self.train_dataloader = DataLoader(self.train_dataset, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        if len(self.val_dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve equal num of samples per-process.')
        sampler_val = DistributedSampler(self.val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False)
        self.val_dataloader = DataLoader(self.val_dataset, sampler=sampler_val, batch_size=args.val_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
        # ddp: model
        self.model_ddp = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.ddp.gpu], find_unused_parameters=find_unused_parameters)

        # joint loss criterion
        if self.args.joint_criterion == 'l1':
            self.joint_criterion_func = F.l1_loss
        else:
            self.joint_criterion_func = F.mse_loss

        if self.args.data_mode == 'h36m':
            self.gt_smpl_handler = SMPLHandler(path_to_regressor=config.JOINT_REGRESSOR_H36M_correct).to(self.device)   # no parameters
        else:
            self.gt_smpl_handler = SMPLHandler(path_to_regressor=config.JOINT_REGRESSOR_3DPW).to(self.device)   # no parameters
        self.vis_loss_list = ['loss_vertices', 'loss_2d_joints', 'loss_3d_joints', 'loss_theta', 'loss_combine']

        if ddp_utils.get_rank() == 0:
            self.visualizer = Visualizer(config.smpl_neutral, 224, self.device)
            if not self.eval_only:
                self.writer = SummaryWriter(self.args.log_dir)
        
                # save updated args
                with open(os.path.join(self.log_dir, "config_updated.json"), "a") as f:
                    self.args.start_epoch = self.start_epoch
                    self.args.global_iter = self.global_iter
                    self.args.datetime = (datetime.utcnow()+timedelta(hours=8)).strftime("%Y/%m/%d, %H:%M:%S")
                    json.dump(vars(self.args), f, indent=4)
    
    def auto_load(self):
        checkpoint_paths = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pt')))
        if self.args.auto_load > 0 and len(checkpoint_paths) > 0:
            checkpoint_path = checkpoint_paths[-1]
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.start_epoch = state_dict['epoch'] + 1
            self.global_iter = state_dict['global_iter']
            
            print(f'{checkpoint_path} is loaded!')

    def prepare_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=0)
        return optimizer
    
    @staticmethod
    def load_checkpoint(model, checkpoint_path, edit_state_dict=0):
        old_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'optimizer' in old_dict:
            old_dict = old_dict['model']
        if edit_state_dict > 0:
            new_dict = {}
            for k, v in old_dict.items():
                if "J_regressor_h36m_correct" not in k:
                    new_dict[k] = v
            model.load_state_dict(new_dict, strict=False)
        else:
            model.load_state_dict(old_dict, strict=False)
        
        print(f'{checkpoint_path} is loaded!')

    def compute_loss_2djoint(self, pred, gt, has_gt=None):
        if len(gt) > 0:
            conf = gt[:, :, -1].unsqueeze(-1).clone()
            return (conf * self.joint_criterion_func(pred, gt[:, :, :-1], reduction='none')).mean()
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_loss_3djoint(self, pred, gt, has_gt, midpoint_as_origin=True):
        gt = gt[has_gt == 1]
        pred = pred[has_gt == 1]
        if len(gt) > 0:
            conf = gt[:, :, -1].unsqueeze(-1).clone()
            gt = gt[:, :, :-1].clone()
            if midpoint_as_origin:
                gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
                gt = gt - gt_pelvis[:, None, :]
                pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
                pred = pred - pred_pelvis[:, None, :]
            return (conf * self.joint_criterion_func(pred, gt, reduction='none')).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_loss_3djoint_PA(self, pred, gt, has_gt):
        gt = gt[has_gt == 1]
        pred = pred[has_gt == 1]
        if len(gt) > 0:
            conf = gt[:, :, -1].clone()
            gt = gt[:, :, :-1].clone()
            return (conf * pytorch_reconstruction_error(pred, gt, reduction='none')).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_loss_J_Tpose(self, pred, gt, has_gt):
        gt = gt[has_gt == 1]
        pred = pred[has_gt == 1]
        if len(gt) > 0:
            return F.l1_loss(pred, gt, reduction='mean')
        else:
            return torch.tensor(0.0, device=self.device)

    def compute_loss_vertices(self, pred, gt, has_gt):
        pred = pred[has_gt == 1]
        gt = gt[has_gt == 1]
        if len(gt) > 0:
            return F.l1_loss(pred, gt) 
        else:
            return torch.tensor(0.0, device=self.device)
    
    def compute_loss_theta_beta(self, pred, gt, has_gt):
        pred = pred[has_gt == 1]
        gt = gt[has_gt == 1]
        if len(gt) > 0:
            return F.l1_loss(pred, gt) 
        else:
            return torch.tensor(0.0, device=self.device)
    

    def load_batch(self, batch):
        img_paths, images, annotations = batch
        images = images.to(self.device)
        ori_img = annotations['ori_img'].to(self.device)

        # GT 2d keypoint
        gt_2d_joints = annotations['joints_2d'].to(self.device)
        gt_2d_joints = gt_2d_joints[:, constants.J24_TO_J14, :]
        has_2d_joints = annotations['has_2d_joints'].to(self.device)

        # GT 3d keypoint
        gt_3d_joints = annotations['joints_3d'].to(self.device)
        gt_3d_pelvis = gt_3d_joints[:,constants.J24_NAME.index('Pelvis'),:3]
        gt_3d_joints = gt_3d_joints[:,constants.J24_TO_J14,:] 
        gt_3d_joints_minus_pelvis = gt_3d_joints.clone()
        gt_3d_joints_minus_pelvis[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
        has_3d_joints = annotations['has_3d_joints'].to(self.device)

        # GT smpl
        gt_pose = annotations['pose'].to(self.device)
        gt_betas = annotations['betas'].to(self.device)
        has_smpl = annotations['has_smpl'].to(self.device)

        gt_smpl_dict = self.gt_smpl_handler(gt_pose, gt_betas, 'axis-angle')
        gt_vertices = gt_smpl_dict['vertices']
        gt_vertices_minus_pelvis = gt_smpl_dict['vertices_minus_pelvis'] 

        # GT cam
        gt_3d_joints_from_smpl = gt_smpl_dict['joints']
        has_cam = torch.logical_and(has_smpl==1, has_2d_joints==1)
        gt_cam = compute_weak_perspective_cam(gt_3d_joints_from_smpl[has_cam], gt_2d_joints[has_cam, :, 0:-1], gt_2d_joints[has_cam, :, -1])
        
        return {'images': images, 'img_paths': img_paths, 'ori_img': ori_img, 
                'gt_2d_joints': gt_2d_joints, 'has_2d_joints': has_2d_joints, 'gt_cam': gt_cam, 'has_gt_cam': has_cam,
                'gt_3d_joints': gt_3d_joints, 'gt_3d_joints_minus_pelvis': gt_3d_joints_minus_pelvis, 'has_3d_joints': has_3d_joints, 
                'gt_pose': gt_pose, 'gt_betas': gt_betas, 'has_smpl': has_smpl, 'gt_vertices': gt_vertices, 'gt_vertices_minus_pelvis': gt_vertices_minus_pelvis}

    # Simplified
    def forward_step(self, batch, phase='train'):
        # --- Load batch data ---
        batch_dict = self.load_batch(batch)

        # --- Run model ---
        model = self.model_ddp
        
        pred_smpl_dicts = model(batch_dict['images'])
        
        pred_smpl_dict = pred_smpl_dicts[-1]
        pred_rotmat = pred_smpl_dict['theta']  
        pred_vertices_minus_pelvis = pred_smpl_dict['vertices_minus_pelvis']
        pred_3d_joints_from_smpl_minus_pelvis = pred_smpl_dict['joints_minus_pelvis']
        pred_2d_joints_from_smpl = pred_smpl_dict['joints2d']

        # losses
        if phase == 'train':
            self.loss_vertices = self.compute_loss_vertices(pred_vertices_minus_pelvis, batch_dict['gt_vertices_minus_pelvis'], batch_dict['has_smpl'])
            self.loss_2d_joints = self.compute_loss_2djoint(pred_2d_joints_from_smpl, batch_dict['gt_2d_joints'], batch_dict['has_2d_joints'])
            self.loss_3d_joints = self.compute_loss_3djoint(pred_3d_joints_from_smpl_minus_pelvis, batch_dict['gt_3d_joints_minus_pelvis'], batch_dict['has_3d_joints'], midpoint_as_origin=True)
            self.loss_theta = self.compute_loss_theta_beta(pred_rotmat, batch_rodrigues_v2(batch_dict['gt_pose']), batch_dict['has_smpl'])
            
            self.loss_combine = self.args.w_vert * self.loss_vertices + self.args.w_2dj * self.loss_2d_joints + \
                                self.args.w_3dj * self.loss_3d_joints + self.args.w_theta * self.loss_theta  
        else:
            error_vertices = mean_per_vertex_error(pred_vertices_minus_pelvis.detach(), batch_dict['gt_vertices_minus_pelvis'], batch_dict['has_smpl'])
            error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl_minus_pelvis.detach(), batch_dict['gt_3d_joints_minus_pelvis'], batch_dict['has_3d_joints'])
            error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl_minus_pelvis.detach().cpu().numpy(), batch_dict['gt_3d_joints_minus_pelvis'][:,:,:3].cpu().numpy(), reduction=None)
            self.mpve_sum += np.sum(error_vertices)    # mean per-vertex error
            self.mpve_count += torch.sum(batch_dict['has_smpl']).item()
            self.mpjpe_sum += np.sum(error_joints)   # mean per-joint position error
            self.mpjpepa_sum += np.sum(error_joints_pa)
            self.mpjpe_count += torch.sum(batch_dict['has_3d_joints']).item()

        # for visualization
        self.images = batch_dict['ori_img']
        self.pred_vertices = pred_smpl_dict['vertices']
        self.pred_2d_joints_from_smpl = pred_smpl_dict['joints2d']
        self.pred_cam = pred_smpl_dict['cam']
        self.gt_vertices = batch_dict['gt_vertices']
        self.gt_2d_joints = batch_dict['gt_2d_joints']
        self.gt_cam = self.pred_cam.clone()   
        
    
    def run_training(self):
        print('Start training...')

        model = self.model_ddp
        save_model = self.model
        
        model.train()
        
        for epoch in tqdm.tqdm(range(self.start_epoch, self.num_epochs), total=self.num_epochs, initial=self.start_epoch, disable=(ddp_utils.get_rank()!=0)):          
            
            
            self.train_dataloader.sampler.set_epoch(epoch)
            for iter, batch in tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f'Train-{epoch:03}', disable=(ddp_utils.get_rank()!=0)):
                self.global_iter += 1
                self.forward_step(batch, phase='train')
            
                # optimize
                self.optimizer.zero_grad()
                self.loss_combine.backward()
                if self.args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
                self.optimizer.step()
                
                # tensorboard summary
                if ddp_utils.get_rank() == 0:
                    if self.global_iter % self.args.summary_steps == 0:
                        self.summary_tensorboard(do_render=True)

                    
            # evaluation
            if epoch % self.args.eval_epochs == 0:
                self.run_evaluation(epoch, do_render=True)
                # model.train()
            
            # save checkpoint
            if ddp_utils.get_rank() == 0:
                if epoch % self.args.save_epochs == 0:
                    save_dict = {'model': save_model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch, 'global_iter': self.global_iter}
                    torch.save(save_dict, os.path.join(self.checkpoint_dir, 'epoch_{:03}.pt'.format(epoch)))
    
    @torch.no_grad()
    def summary_tensorboard(self, do_render=True):
        for name in self.vis_loss_list:
            value = getattr(self, name).item()
            self.writer.add_scalar(name, value, self.global_iter)   # last stage
        
        if do_render:
            
            rendered = self.visualizer.draw_skeleton_and_mesh(self.images, self.gt_2d_joints, self.pred_2d_joints_from_smpl, 
                                                              self.gt_vertices, self.pred_vertices, self.gt_cam, self.pred_cam, num_draws=3)
            self.writer.add_image('train-vis', rendered, self.global_iter, dataformats='HWC')

    @torch.no_grad()
    def run_evaluation(self, epoch=0, do_tb=True, do_render=True):
        self.model_ddp.eval()

        self.mpve_sum = 0
        self.mpve_count = 0
        self.mpjpe_sum = 0
        self.mpjpepa_sum = 0
        self.mpjpe_count = 0
        
        for batch in tqdm.tqdm(self.val_dataloader, total=len(self.val_dataloader), desc=f'Eval-{epoch:03}', disable=(ddp_utils.get_rank()!=0)):
            self.forward_step(batch, phase='eval')

        self.mpve_sum, self.mpve_count, self.mpjpe_sum, self.mpjpepa_sum, self.mpjpe_count = \
            ddp_utils.synchronize_between_processes([self.mpve_sum, self.mpve_count, self.mpjpe_sum, self.mpjpepa_sum, self.mpjpe_count])
        
        mpve = self.mpve_sum / (self.mpve_count + 1e-8)
        mpjpe = self.mpjpe_sum / self.mpjpe_count
        mpjpepa = self.mpjpepa_sum / self.mpjpe_count

        if ddp_utils.get_rank() == 0:
            if do_render:
                rendered = self.visualizer.draw_skeleton_and_mesh(self.images, self.gt_2d_joints, self.pred_2d_joints_from_smpl, 
                                                                self.gt_vertices, self.pred_vertices, self.gt_cam, self.pred_cam, num_draws=6)
            
            if do_tb:
                self.writer.add_scalar('mpve', mpve, epoch)
                self.writer.add_scalar('mpjpe', mpjpe, epoch)
                self.writer.add_scalar('mpjpe-pa', mpjpepa, epoch)
                if do_render:
                    self.writer.add_image('eval-vis', rendered, epoch, dataformats='HWC')
            else:
                print(f'mpve:         {mpve*1000:.3f}')
                print(f'mpjpe:        {mpjpe*1000:.3f}')
                print(f'mpjpe-pa:     {mpjpepa*1000:.3f}')
                if do_render:
                    save_path = self.args.load_checkpoint[:-3] + '.png'
                    imageio.imwrite(save_path, (rendered*255).astype(np.uint8))

            
        
        self.model_ddp.train()

    def run(self):
        if self.eval_only:
            self.run_evaluation(do_tb=False, do_render=False)
        else:
            self.run_training()



if __name__ == '__main__':
    from utils.argument_manager import ArgManager
    arg_manager = ArgManager()
    
    trainer = Trainer(arg_manager.args)
    trainer.run()
    


