import numpy as np
import torch 
import imageio
from argparse import ArgumentParser
from torchvision import transforms

import config
from models.transformer_basics import TranformerConfig
from models.smpler import SMPLer
from utils.visualizer import Visualizer


# Basic arguments
parser = ArgumentParser()
parser.add_argument('--img_path', required=True, type=str, help='Input image path')
parser.add_argument('--device', default='cuda', type=str, help='Device to run the model')
args = parser.parse_args()

args.data_mode = 'h36m'
args.model_type = 'smpler'
args.hrnet_type = 'w32'
args.num_transformers = 3
args.load_checkpoint = 'pretrained/SMPLer_h36m.pt'

trans_cfg = TranformerConfig()
trans_cfg.raw_feat_dim = config.hrnet_dict[args.hrnet_type][2]


# Define SMPLer model
model = SMPLer(args, trans_cfg)

state_dict = torch.load(args.load_checkpoint, map_location='cpu')
if 'model' in state_dict:
    state_dict = state_dict['model']
model.load_state_dict(state_dict, strict=True)

model.eval()
model.to(args.device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224, 224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

visualizer = Visualizer(config.smpl_neutral, 224, args.device)


# Run model
img = imageio.imread(args.img_path)
img = transform(img)[None].to(args.device)
img_vis = img * torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(args.device) + torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(args.device)

pred_smpl_dict = model(img)[-1]
pred_vertices = pred_smpl_dict['vertices']
pred_cam = pred_smpl_dict['cam']
rendered = visualizer.draw_pred_mesh(pred_vertices, pred_cam, img_vis, num_draws=1, mesh_color=(0.93,0.68,0.62))[-1]

imageio.imwrite(args.img_path[:-4]+'_out.png', (rendered*255).astype(np.uint8))
print('Result is saved at:', args.img_path[:-4]+'_out.png')

