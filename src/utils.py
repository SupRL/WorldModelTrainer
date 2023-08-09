from env import MultiJointEnv
import argparse

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