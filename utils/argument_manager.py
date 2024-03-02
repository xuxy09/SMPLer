import os
import json
from argparse import ArgumentParser


class ArgManager:
    def __init__(self, manual_args=None):
        parser = ArgumentParser()
        parser.add_argument('--exp_name', type=str, default='smpler', help='Name of the experiment')
        parser.add_argument('--log_root', type=str, default='logs', help='Root directory to store logs')
        parser.add_argument('--seed', type=int, default=1234, help='Random seed')
        parser.add_argument("--device", default='cuda', help="Training device, has to be cuda")

        # training
        parser.add_argument('--num_epochs', type=int, default=100, help='Total number of training epochs')
        parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
        parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
        parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for evaluation')
        parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
        parser.add_argument('--summary_steps', type=int, default=500, help='Summary saving frequency')
        parser.add_argument('--save_epochs', type=int, default=1, help='Checkpoint saving frequency')
        parser.add_argument('--eval_epochs', type=int, default=1, help='Evaluation frequency')
        parser.add_argument('--joint_criterion', type=str, default='mse', help='L1 or mse')

        # model
        parser.add_argument('--model_type', type=str, default='backbone', help='Model: backbone or smpler')
        parser.add_argument('--hrnet_type', type=str, default='w32', help='HRNet: w32 or w48')
        parser.add_argument('--num_transformers', default=3, type=int, help='Number of Transformer Blocks')
        
        # loss
        parser.add_argument("--w_vert", type=float, default=100, help="Weight of vertex loss")
        parser.add_argument("--w_2dj", type=float, default=100, help="Weight of 2D joint loss")
        parser.add_argument("--w_3dj", type=float, default=1000, help="Weight of 3D joint loss")
        parser.add_argument("--w_theta", type=float, default=50, help="Weight of SMPL theta loss")

        # others
        parser.add_argument('--data_mode', type=str, default='h36m', help='Dataset: h36m or 3dpw')
        parser.add_argument('--load_checkpoint', default=None, help='Pre-load checkpoint path')
        parser.add_argument('--eval_only', action='store_true', default=False, help='Only for evaluation purpose')
        parser.add_argument('--mesh_color', type=str, default='0.93,0.68,0.62', help='Mesh color for visualizer')
        parser.add_argument('--auto_load', type=int, default=0, help='Auto-load recent checkpoint')
        parser.add_argument('--clip_grad', type=float, default=None, help='Clip gradient')

        if manual_args is None:
            self.args = parser.parse_args()
        else:
            self.args = parser.parse_args(args=manual_args)
        
        if self.args.load_checkpoint is not None:
            if 'perscam' in self.args.exp_name:
                assert 'perscam' in self.args.load_checkpoint

        self.args.mesh_color = tuple(float(x) for x in self.args.mesh_color.split(','))
        if not self.args.eval_only:
            self.set_train()
    
    def set_train(self):
        self.make_log_dirs()
        self.save_dump()
    
    def make_log_dirs(self):
        """ logs
             |
             +-- exp_name
                 |
                 +-- config.json
                 |
                 +-- tensorboard_event
                 |
                 +-- checkpoints
        """
        # mkdir for exp_name/, exp_name/checkpoints/
        self.args.log_dir = os.path.join(os.path.abspath(self.args.log_root), self.args.exp_name)
        os.makedirs(self.args.log_dir, exist_ok=True)
        self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/exp_name/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        
        with open(os.path.join(self.args.log_dir, "config.json"), "a") as f:
            json.dump(vars(self.args), f, indent=4)
