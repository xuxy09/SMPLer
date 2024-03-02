import torch 
import config as project_cfg
from models.hrnet.pose_hrnet import get_pose_net
from models.hrnet.utils.default import _C as cfg
from models.SMPL_handler import SMPLHandler
from  models.SMPL_predictor import SMPLPredictor


def build_hrnet(hrnet_type='w32'):
    hrnet_dict = project_cfg.hrnet_dict
    # hrnet config
    cfg.defrost()
    cfg.merge_from_file(hrnet_dict[hrnet_type][0])
    cfg.multi_scale_feature = True
    cfg.final_layer = False
    cfg.freeze()
    # build hrnet
    net = get_pose_net(cfg, True)
    net.load_state_dict(torch.load(hrnet_dict[hrnet_type][1], map_location='cpu'), strict=False)
    return net


class BaselineModel(torch.nn.Module):
    """ Predict SMPL
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = build_hrnet(args.hrnet_type)
        self.SMPL_predictor = SMPLPredictor(args=args)
        self.smpl_handler = SMPLHandler()

    def forward(self, img):
        """
        # img --> (backbone) --> feat_list --> global_feat
        # globa_feat --> (SMPL_predictor) --> theta, beta, cam
        # theta, beta, cam --> (SMPL_handler) --> joints_2d
        """
        feat_list = self.backbone(img) 
        global_feat = feat_list[-1].flatten(2).mean(2)
        pred_rotmat, pred_shape, pred_cam = self.SMPL_predictor(global_feat, global_pooling=False) 
        pred_smpl_dict = self.smpl_handler(pred_rotmat, pred_shape, theta_form='rot-matrix', cam=pred_cam)
        return [pred_smpl_dict] 