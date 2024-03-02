import os

# dataset path 
dataset_root = './datasets'
H36m_coco40k_Muco_UP_Mpii_yaml = os.path.join(dataset_root, "Tax-H36m-coco40k-Muco-UP-Mpii/train.yaml")
H36m_val_p2_yaml = os.path.join(dataset_root, "human3.6m/valid.protocol2.yaml")
PW3D_train_yaml = os.path.join(dataset_root, "3dpw/train.yaml")
PW3D_val_yaml = os.path.join(dataset_root, "3dpw/test_has_gender.yaml")

# meta data path
smpl_mean_params_path = "meta_data/smpl_mean_params.npz"
smpl_neutral = "meta_data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
JOINT_REGRESSOR_H36M_correct = "meta_data/J_regressor_h36m_correct.npy"
JOINT_REGRESSOR_3DPW = "meta_data/J_regressor_3dpw.npy"

hrnet_dict = {'w32': ('pretrained/hrnet/w32_256x256_adam_lr1e-3.yaml', 'pretrained/hrnet/pose_hrnet_w32_256x256.pth', [32, 64, 128, 256]),
              'w48': ('pretrained/hrnet/w48_256x192_adam_lr1e-3.yaml', 'pretrained/hrnet/pose_hrnet_w48_256x192.pth', [48, 96, 192, 384]),
              }
