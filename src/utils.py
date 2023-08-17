from env import MultiJointEnv
import argparse
import torch 
import numpy as np 
from pytorch3d.transforms import euler_angles_to_matrix

def get_env(**kwargs):
    return MultiJointEnv(**kwargs)

def readParser():
    parser = argparse.ArgumentParser(description='WorldModel')

    parser.add_argument('--seed', type=int, default=18, metavar='N', help='random seed (default: 18)')
    # world model network structure config
    parser.add_argument('--pred_hidden_size', type=int, default=256, metavar='E', help='hidden size for predictive model')
    parser.add_argument('--pred_n_heads', type=int, default=1, metavar='E', help='number of heads for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E', help='environment reward size')
    # env setting
    parser.add_argument('--env_name', default="Excavator",help='environment name')
    parser.add_argument('--joint_name', type=str, default='boom_arm_swing', metavar='A', help='joint_name')
    # env setting for state seq data
    parser.add_argument('--state_seq_length', type=int, default=20)
    parser.add_argument('--state_round_num', type=int, default=5, help='number of rounds for state seq data')
    parser.add_argument('--world_model_type', type=str, default="dt", help='world model type: dt')
    parser.add_argument('--world_model_data_file_max_num', type=int, default=-1, help='max file num of the data')
    parser.add_argument('--world_model_train_data_max_size', type=int, default=-1, help='world_model_train_data_max_size')
    parser.add_argument('--world_model_datalaoder_batch_size', type=int, default=100000, metavar='A', help='the number of the data fetched by the dataloader in one batch')
    # delta raw
    parser.add_argument('--world_model_output', type=str, default="delta", help='delta or raw')
    parser.add_argument('--world_model_train_batch_size', type=int, default=1024, metavar='A', help='batch size for training world model')
    parser.add_argument('--world_model_learning_rate', type=float, default=5.8e-5, metavar='A', help='learning rate for training world model')
    parser.add_argument('--world_model_local_data_shuffle_num', type=int, default=1e5, metavar='A', help='local data shuffle num')
    parser.add_argument('--world_model_clip_grad_norm', type=float, default=1.0, metavar='A', help='clip grad norm')
    parser.add_argument('--epochs', type=int, default=None, metavar='A', help='epochs for training world model')
    parser.add_argument('--world_model_lr_schedule_T_max', type=int, default=90, metavar='A', help='lr schedule T max')
    # save and load model
    parser.add_argument("--save_model_interval", default=1e6, type=int, help='save model interval')
    parser.add_argument('--load_world_model', default="")
    parser.add_argument('--world_model_loss_scale', type=str, default="all", help='all or last')
    return parser

def excavator_arm_fk(angles: torch.Tensor)-> torch.Tensor:
    
    '''
    输入关节角度，输出末端坐标位置
    输入：[batch_size, joint_pos] shape:[N, 3]
    输出：[batch_size, end_pos] shape: [N, 3]
    在数据集中总共有3维的关节角度,分别是boom, arm, swing
    示例:
    boom_pos = 0.1
    arm_pos = 0.1
    swing_pos = 0.1
    end_pos = excavator_arm_fk(torch.tensor([[boom_pos, arm_pos, swing_pos]]))
    '''

    def torch_matrix_from_pose(pose):
        assert isinstance(pose, torch.Tensor), f"pose must be a torch.Tensor, got {type(pose)}"
        m = torch_sxyz_euler_angles_to_matrix(pose)  # [B, 3, 3]
        t = pose[:, :3]  # [B, 3]
        mt = torch.cat([m, t.unsqueeze(-1)], dim=-1)  # [B, 3, 4]
        constant_row_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=pose.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        # add a constant row to the bottom of the matrix
        return torch.cat([mt, constant_row_tensor.expand(mt.shape[0], 1, 4)], dim=1)  # [B, 4, 4]
    def torch_sxyz_euler_angles_to_matrix(pose):
        assert isinstance(pose, torch.Tensor), f"pose must be a torch.Tensor, got {type(pose)}"
        angles = pose[:, 3:]
        # 'sxyz' means static frame, rotation order x-y-z.
        # So we reverse the angles and call the standard function with reversed order.
        angles_reversed = torch.flip(angles, dims=(-1,))
        return euler_angles_to_matrix(angles_reversed, "ZYX")
    
    if not isinstance(angles, torch.Tensor):
        try:
            angles = torch.tensor(np.array(angles), dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Input must be a torch.Tensor or a numpy array, got {type(angles)}") from e
    else:
        assert angles.dtype == torch.float32, "Input tensor must be of dtype float32"
    angles = torch.cat([angles[:,:2], torch.ones_like(angles[:,-1]).unsqueeze(1), angles[:,-1].unsqueeze(1)], dim=1) 
    # angles: [boom, arm, bucket, swing]
    assert angles.shape[1] == 4, "angles must be a list of 4 angles"
    batch_size = angles.shape[0]
    boom_len = 5.74602
    arm_len = 2.93823
    operating_arm_height = 1.9
    boom_init_angle = 1.0
    arm_init_angle = 1.5
    arm_offset_x = 0.
    arm_offset_y = 0.
    xcmg_simple_fk = [[0, 0, 0, 0, boom_init_angle, 0],
                      [0, 0, boom_len, 0, arm_init_angle, 0],
                      [0, 0, arm_len, 0, 0, 0],
                      [0, 0, operating_arm_height, 0, 0, 0],
                      [arm_offset_x, arm_offset_y, 0, 0, 0, 0]]
    # expand xcmg_simple_fk to [B, 5, 6]
    xcmg_simple_fk = np.expand_dims(xcmg_simple_fk, axis=0).repeat(batch_size, axis=0)
    xcmg_simple_fk = torch.tensor(xcmg_simple_fk, dtype=torch.float32, device=angles.device)
    # tf link: gripper(bucket), arm2, arm1, base(base of the arm1), upper(swing), ground
    # compute the transformation matrices for each joint
    tf_arm1_base = torch_matrix_from_pose(xcmg_simple_fk[:, 0])
    tf_arm2_arm1 = torch_matrix_from_pose(xcmg_simple_fk[:, 1])
    tf_gripper_arm2 = torch_matrix_from_pose(xcmg_simple_fk[:, 2])
    tf_upper_ground = torch_matrix_from_pose(xcmg_simple_fk[:, 3])
    tf_base_upper = torch_matrix_from_pose(xcmg_simple_fk[:, 4])

    # compute the transformation matrices for real joints angles
    # get a new angles tensor with the shape of [B, 4, 1]
    expand_angles = torch.unsqueeze(angles, dim=-1)
    tf_arm1new_arm1 = torch_matrix_from_pose(torch.cat([torch.zeros([batch_size, 4], dtype=torch.float32, device=angles.device), expand_angles[:, 0, :],
                                                        torch.zeros([batch_size, 1], dtype=torch.float32, device=angles.device)], dim=-1))
    tf_arm2new_arm2 = torch_matrix_from_pose(torch.cat([torch.zeros([batch_size, 4], dtype=torch.float32, device=angles.device), expand_angles[:, 1, :],
                                                        torch.zeros([batch_size, 1], dtype=torch.float32, device=angles.device)], dim=-1))
    tf_grippernew_gripper = torch_matrix_from_pose(torch.cat([torch.zeros([batch_size, 4], dtype=torch.float32, device=angles.device), expand_angles[:, 2, :],
                                                              torch.zeros([batch_size, 1], dtype=torch.float32, device=angles.device)], dim=-1))
    tf_uppernew_upper = torch_matrix_from_pose(torch.cat([torch.zeros([batch_size, 5], dtype=torch.float32, device=angles.device), expand_angles[:, 3, :]], dim=-1))


    # compute the multiplication of the corresponding transformation matrices
    tf_grippernew_arm2 = torch.matmul(tf_gripper_arm2, tf_grippernew_gripper)
    tf_arm2new_arm1 = torch.matmul(tf_arm2_arm1, tf_arm2new_arm2)
    tf_arm1new_base = torch.matmul(tf_arm1_base, tf_arm1new_arm1)
    tf_grippernew_arm1new = torch.matmul(tf_arm2new_arm1, tf_grippernew_arm2)
    tf_grippernew_base = torch.matmul(tf_arm1new_base, tf_grippernew_arm1new)
    tf_uppernew_base = torch.matmul(tf_upper_ground, tf_uppernew_upper)
    tf_swingbase_arm = torch.matmul(torch.inverse(tf_base_upper), torch.inverse(tf_uppernew_base))
    tf_grippernew_swingbase = torch.matmul(torch.inverse(tf_swingbase_arm), tf_grippernew_base)
    return tf_grippernew_swingbase[:,:3,3]