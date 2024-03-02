from smplx import SMPL
import torch.nn as nn
import torch
import numpy as np

import config 
import constants
from utils.geometry import orthographic_projection


class SMPLHandler(nn.Module):
    def __init__(self, path_to_pkl=config.smpl_neutral, path_to_regressor=config.JOINT_REGRESSOR_H36M_correct):
        super().__init__()
        self.smpl = SMPL(path_to_pkl, create_transl=False, create_betas=False, create_body_pose=False, create_global_orient=False)
        J_regressor_h36m_correct = torch.from_numpy(np.load(path_to_regressor)).float()
        self.register_buffer('J_regressor_h36m_correct', J_regressor_h36m_correct)


    def forward(self, theta, beta, theta_form='axis-angle', cam=None):
        """ theta, beta --> vertices --> joints --> joints_2d (if cam is not None)
        """
        if theta_form == 'axis-angle':
            output = self.smpl(betas=beta, body_pose=theta[:,3:], global_orient=theta[:,:3], pose2rot=True) 
        elif theta_form == 'rot-matrix':
            output = self.smpl(betas=beta, body_pose=theta[:,1:], global_orient=theta[:,0].unsqueeze(1), pose2rot=False)
        else:
            raise NotImplementedError('Unknown theta form: {}'.format(theta_form))
        
        vertices = output.vertices  
        smpl_joints = output.joints[:, :24]    # original-smpl 24 keypoints
        
        regressor = self.J_regressor_h36m_correct
        
        joints = torch.einsum('bik,ji->bjk', [vertices, regressor])   # 17 joints following h36m
        pelvis = joints[:,constants.H36M_J17_NAME.index('Pelvis'),:]

        joints = joints[:,constants.H36M_J17_TO_J14,:]    # 14 joints

        vertices_minus_pelvis = vertices - pelvis[:, None, :]
        joints_minus_pelvis = joints - pelvis[:, None, :]

        output_dict = {'smpl_joints': smpl_joints, 'joints': joints, 'joints_minus_pelvis': joints_minus_pelvis, #'others': output.others,
                       'vertices': vertices, 'vertices_minus_pelvis': vertices_minus_pelvis, 'theta': theta, 'beta': beta}

        if cam is not None:
            joints_2d = orthographic_projection(joints, cam)
            smpl_joints_2d = orthographic_projection(smpl_joints, cam)
            
            output_dict.update({'joints2d': joints_2d, 'smpl_joints2d': smpl_joints_2d, 'cam': cam})

        return output_dict


   
