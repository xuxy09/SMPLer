import torch.nn as nn
import numpy as np
import torch
from utils.geometry import rot6d_to_rotmat
import config


class SMPLPredictor(nn.Module):
    def __init__(self, manual_in_dim=None, args=None):
        super().__init__()
        npose = 24 * 6

        if args is None:
            in_dim = 256
            
        else:
            in_dim = config.hrnet_dict[args.hrnet_type][2][-1]
            
        
        if manual_in_dim is not None:
            in_dim = manual_in_dim
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_dim + npose + 10 + 3, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        # make initial prediction close to zero
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(config.smpl_mean_params_path)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.zeros(1, 3)
        init_cam[0, 0] = 1

        
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def forward(self, x, n_iter=3, global_pooling=True):
        """ x: [b, c]
        """
        batch_size = x.shape[0]
        init_pose = self.init_pose.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        if global_pooling:
            x = x.flatten(2).mean(2)   # global pooling
        
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


