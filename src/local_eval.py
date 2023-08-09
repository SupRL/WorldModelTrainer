from predict import state_predict
import inspect
import numpy as np
import random
from utils import get_env, readParser
import inspect
import os
import random
import math
import datetime
from copy import deepcopy
import numpy as np
import torch
from itertools import accumulate
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

state_columns = ['pos_boom', 'pos_arm', 'pos_swing', 'vel_boom', 'vel_arm', 'vel_swing', 'state_time', 'action_time']
action_columns = ['pwm_boom', 'pwm_arm', 'pwm_swing']

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, env, mode, csv_file_names, dfs_map, **kwargs):
        super(IterDataset).__init__()
        self.env = env
        self.args_dict = kwargs
        self.joint_name = self.args_dict["joint_name"]
        self.train_data_max_size = self.args_dict.get("world_model_train_data_max_size", None)
        self.train_data_max_size = (self.train_data_max_size
                                   if (self.train_data_max_size is not None and self.train_data_max_size > 0) else None)
        self.dfs_map_curser = {"train": 0, "test": 0}
        self.mode = mode
        self.data_dir = self._get_data_dir()
        self.total_step = 0
        self.total_data_size = 0
        if dfs_map is None or csv_file_names is None:
            self.pre_load_csvs()
        else:
            self.dfs_map = dfs_map
            self.csv_file_names = csv_file_names
            self.total_data_size = sum([len(df) for df in self.dfs_map[self.mode]])
            random.shuffle(self.dfs_map[mode])
        self.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
        self.global_world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]"
              f"[IterableDataset] Init dataset for global_rank/global world size: "
              f"{self.global_rank}/{self.global_world_size}")
        self.data_iter = self.get_data()
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]"
              f"[IterableDataset] Get iterator for global_rank/global world size: "
              f"{self.global_rank}/{self.global_world_size}")

    def pre_load_csvs(self):
        self.dfs_map = {}
        self.csv_file_names = {self.mode: self.get_all_csvfiles(self.mode)}
        print(f"get {self.mode} dfs...")
        self.dfs_map[self.mode] = [k for k in self.collect_df(self.csv_file_names[self.mode]) if k is not None]
        random.shuffle(self.dfs_map[self.mode])
        total_data_size = sum([len(df) for df in self.dfs_map[self.mode]])
        if self.train_data_max_size is not None and self.mode == "train":
            file_index = next((i for i, total in enumerate(accumulate([len(df) for df in self.dfs_map[self.mode]]))
                               if total >= self.train_data_max_size), None)
            if file_index is not None:
                self.dfs_map[self.mode] = self.dfs_map[self.mode][:file_index+1]
        self.total_data_size = sum([len(df) for df in self.dfs_map[self.mode]])
        print(f"Total {self.mode} data: {self.total_data_size} / {total_data_size}")

    def _get_data_dir(self):
        from pathlib import Path
        # 此处按照需要将data_path修改为本地测评所需的数据集路径
        data_path = os.path.join(Path(__file__).absolute().parent.parent, 'dataset')
        return data_path

    def get_folders(self, path):
        folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        folders = [folder for folder in folders]
        return folders

    def get_all_csvfiles(self, mode):
        data_path = os.path.join(self.data_dir, mode)
        data_path_list = self.get_folders(data_path)
        csv_files = []
        for path in data_path_list:
            print(f"Reading {mode} {path} {os.path.join(data_path, path)}...")
            csv_files.extend(glob.glob(os.path.join(data_path, path, "*.csv")))
        print(f"Total {len(csv_files)} files.")
        random.shuffle(csv_files)
        if self.args_dict["world_model_data_file_max_num"] is None or self.args_dict["world_model_data_file_max_num"] == -1:
            return csv_files
        else:
            return csv_files[:self.args_dict["world_model_data_file_max_num"]]

    def collect_df(self, csv_files, num_workers=10):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            outputs = list(tqdm(executor.map(self.get_df, csv_files), total=len(csv_files)))
        return outputs

    def get_df(self, csv_file):
        df = pd.read_csv(csv_file, header=0, low_memory=False)
        nan_include = df.isna().any().any()
        if nan_include:
            nan_rows = df[df.isna().any(axis=1)]
            last_row = nan_rows.iloc[-1]
            if df.iloc[-1].equals(last_row) and len(nan_rows) <= 1:
                df = df.iloc[:-1]
            else:
                raise ValueError(f"nan in {csv_file}")
        return df

    def get_data(self):
        mode = self.mode
        env = self.env
        if env is None:
            raise ValueError("env is None")
        global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
        global_world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        worker_info = torch.utils.data.get_worker_info()
        process_iter_start = 0
        process_iter_end = len(self.dfs_map[mode])
        if worker_info is None:
            worker_id = 0
            iter_start = process_iter_start
            iter_end = process_iter_end
        else:
            per_worker = int(math.ceil((process_iter_end - process_iter_start) / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker + process_iter_start
            iter_end = min(iter_start + per_worker, process_iter_end)
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]"
                  f"[IterableDataset] global rank/global world size: "
                  f"{global_rank}/{global_world_size}, worker_id: {worker_id}/{worker_info.num_workers}, "
                  f"iter_start/process_iter_start: {iter_start}/{process_iter_start}, "
                  f"iter_end/process_iter_end: {iter_end}/{process_iter_end}, "
                  f"worker sample number: {sum([len(self.dfs_map[mode][i]) for i in range(iter_start, iter_end)])}")
        local_shuffle_num = self.args_dict["world_model_local_data_shuffle_num"]
        local_data_buffer = {}
        for dfs_map_curser in range(iter_start, iter_end):
            df = self.dfs_map[mode][dfs_map_curser]
            s, a, r, next_s, d, rtg, timesteps, mask = self.get_one_df_data_vectorized(df, env)
            local_data_buffer["s"] = np.concatenate((local_data_buffer["s"], s), axis=0) if "s" in local_data_buffer else s
            local_data_buffer["a"] = np.concatenate((local_data_buffer["a"], a), axis=0) if "a" in local_data_buffer else a
            local_data_buffer["r"] = np.concatenate((local_data_buffer["r"], r), axis=0) if "r" in local_data_buffer else r
            local_data_buffer["next_s"] = np.concatenate((local_data_buffer["next_s"], next_s), axis=0) if "next_s" in local_data_buffer else next_s
            local_data_buffer["d"] = np.concatenate((local_data_buffer["d"], d), axis=0) if "d" in local_data_buffer else d
            local_data_buffer["rtg"] = np.concatenate((local_data_buffer["rtg"], rtg), axis=0) if "rtg" in local_data_buffer else rtg
            local_data_buffer["timesteps"] = np.concatenate((local_data_buffer["timesteps"], timesteps), axis=0) if "timesteps" in local_data_buffer else timesteps
            local_data_buffer["mask"] = np.concatenate((local_data_buffer["mask"], mask), axis=0) if "mask" in local_data_buffer else mask
            if local_data_buffer["s"].shape[0] < local_shuffle_num and dfs_map_curser < iter_end - 1:
                continue
            else:
                index_list = np.arange(local_data_buffer["s"].shape[0])
                np.random.shuffle(index_list)
                for i in index_list:
                    yield local_data_buffer["s"][i], local_data_buffer["a"][i], local_data_buffer["r"][i], local_data_buffer["next_s"][i], local_data_buffer["d"][i], local_data_buffer["rtg"][i], local_data_buffer["timesteps"][i], local_data_buffer["mask"][i]
                local_data_buffer = {}

    def get_one_df_data_vectorized(self, df, env):
        # 截取df，只取前2**16行,这是timestamp的最大值
        if len(df) > 2**16:
            df = df.iloc[:2**16]
        df["attention_mask"] = 1
        df["reward"] = 0
        df["done"] = 0
        df["rtg"] = 0
        df["timestep"] = np.arange(df.shape[0])
        df.iloc[-1, df.columns.get_loc("done")] = 1
        df.columns = [col.strip() for col in df.columns]
        # padding the first line of padding_line_df with value in df for the columns pos, vel, next_pos, next_vel
        padding_line_df = df.iloc[[0]].copy()
        # pandding the columns start with pwm with the value of 0
        padding_line_df[[f"pwm_{k}" for k in env.policy_joints]] = 0
        # pandding the columns start with action_time with the value of 0.008
        padding_line_df["action_time"] = 0.008
        # paddding the state_time with the value of 0.002
        padding_line_df["state_time"] = 0.002
        # pandding the attention_mask with the value of 0
        padding_line_df["attention_mask"] = 0
        # repeat the padding line df for env.seq_length times
        padding_df = pd.concat([padding_line_df] * env.seq_length, ignore_index=True)
        # concat the padding df and the original df
        df = pd.concat([padding_df, df], ignore_index=True)
        df = df.round(env.state_round_num)
        
        state_columns = [f"pos_{k}" for k in env.policy_joints] + [f"vel_{k}" for k in env.policy_joints]
        next_state_columns = [f"next_pos_{k}" for k in env.policy_joints] + [f"next_vel_{k}" for k in env.policy_joints]

        state_columns = state_columns + ["state_time", "action_time"]
        action_columns = [f"pwm_{k}" for k in env.policy_joints]
        s_df = np.asarray(df[state_columns].values, dtype=np.float32)
        a_df = np.asarray(df[action_columns].values, dtype=np.float32)
        mask_df = np.asarray(df[["attention_mask"]].values, dtype=np.float32)
        r_df = np.asarray(df[["reward"]].values, dtype=np.float32)
        done_df = np.asarray(df[["done"]].values, dtype=np.float32)
        rtg_df = np.asarray(df[["rtg"]].values, dtype=np.float32)
        timestep_df = np.asarray(df[["timestep"]].values, dtype=np.float32)
        next_s_df = np.asarray(df[next_state_columns].values, dtype=np.float32)

        s = [s_df[i:i + env.seq_length].reshape(1, -1, len(state_columns)) for i in range(1, s_df.shape[0] - env.seq_length + 1)]
        a = [a_df[i:i + env.seq_length].reshape(1, -1, len(action_columns)) for i in range(1, a_df.shape[0] - env.seq_length + 1)]
        mask = [mask_df[i:i + env.seq_length].reshape(1, -1) for i in range(1, mask_df.shape[0] - env.seq_length + 1)]
        r = [r_df[i:i + env.seq_length].reshape(1, -1, 1) for i in range(1, r_df.shape[0] - env.seq_length + 1)]
        d = [done_df[i:i + env.seq_length].reshape(1, -1) for i in range(1, done_df.shape[0] - env.seq_length + 1)]
        rtg = [rtg_df[i:i + env.seq_length].reshape(1, -1,  1) for i in range(1, rtg_df.shape[0] - env.seq_length + 1)]
        timesteps = [timestep_df[i:i + env.seq_length].reshape(1, -1) for i in range(1, timestep_df.shape[0] - env.seq_length + 1)]
        next_s = [next_s_df[i:i + env.seq_length].reshape(1, -1, len(next_state_columns)) for i in range(1, next_s_df.shape[0] - env.seq_length + 1)]


        s = np.concatenate(s, axis=0)
        a = np.concatenate(a, axis=0)
        r = np.concatenate(r, axis=0)
        d = np.concatenate(d, axis=0)
        rtg = np.concatenate(rtg, axis=0)
        timesteps = np.concatenate(timesteps, axis=0)
        mask = np.concatenate(mask, axis=0)
        next_s = np.concatenate(next_s, axis=0)

        return s, a, r, next_s, d, rtg, timesteps, mask

    def __iter__(self):
        return self.data_iter

    def __len__(self):
        return self.total_data_size

class DynamicsModel():
    def __init__(self, state_size, action_size, seq_length, **kwargs):
        self.arg_dict = kwargs
        self.seed = self.arg_dict.get("seed", 18)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.state_size = state_size
        self.action_size = action_size
        self.seq_len = seq_length
        self.action_time = kwargs.get("world_model_action_time", True)
        self.device = kwargs.get("device")
        self.action_time_dim = 2 if self.action_time else 0
        self.env_name = kwargs.get("env_name", "excavator")
        self.env = kwargs.get("env", None)

    def rad_norm(self, pos_states):
        # convert the rad to -math.pi ~ math.pi
        return (pos_states + math.pi) % (2 * math.pi) - math.pi

    def updata_log_dict(self, old_log_dict, new_log_dict):
        for key, value in new_log_dict.items():
            old_log_dict.setdefault(key, []).append(value)
        return old_log_dict

    def eval(self, test_csv_file_names, test_dfs_map, eval_epoch, batch_size, **kwargs):
        test_num_workers = 2
        test_cost_samples = 0
        test_dataset = IterDataset(self.env, "test", test_csv_file_names, test_dfs_map, **kwargs)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                                                        drop_last=True, num_workers=test_num_workers)
        for s, a, r, next_s, d, rtg, timesteps, mask in test_dataloader:
            test_s = s.float().to(self.device)
            test_a = a.float().to(self.device)
            test_next_s = next_s.float().to(self.device)
            test_timesteps = timesteps.long().to(self.device)
            test_mask = mask.float().to(self.device)
            with torch.no_grad():
                next_state = state_predict(**{
                    "states": deepcopy(test_s),
                    "actions": deepcopy(test_a),
                    "timesteps": deepcopy(test_timesteps),
                    "masks": deepcopy(test_mask),
                    "device": self.device,
                    'state_columns': deepcopy(state_columns),
                    'action_columns': deepcopy(action_columns),
                })
            test_prediction_s = next_state
            assert isinstance(test_prediction_s, torch.Tensor), (
            f'state_predict函数返回值类型异常,期望类型为torch.Tensor')
            assert test_prediction_s.shape == (batch_size, 1, self.state_size+self.action_time_dim), (
            f'state_predict函数返回值shape异常, 期望shape为[{batch_size}, 1, {self.state_size+self.action_time_dim}]')
            if test_prediction_s.device != self.device:
                test_prediction_s = test_prediction_s.to(self.device)
            test_cost_samples += (batch_size)
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}][EnsembleDynamicsModel] "
                    f"test costs samples num {test_cost_samples} / {len(test_dataset)}")

def model_eval():
    torch.set_printoptions(precision=7)
    # 参数
    args_parser = readParser()
    args = args_parser.parse_args()
    kwargs = vars(args)

    # Set random seed
    torch.manual_seed(kwargs["seed"])
    np.random.seed(kwargs["seed"])
    random.seed(kwargs["seed"])
    kwargs['seq_length'] = kwargs['state_seq_length']
    world_model_env = get_env(env_purpose="world_model", **kwargs)
    print(f"env init finished, env seq_length: {world_model_env.seq_length}")

    test_dataset4_file_load = IterDataset(world_model_env, "test", None, None, **kwargs)

    kwargs["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {kwargs['device']}")
    world_model_state_dim = world_model_env.joint_num * 2
    world_model_action_dim = world_model_env.joint_num
    seq_len = args.state_seq_length
    env_model_kwargs = {k: v for k, v in kwargs.items() if
                        k not in inspect.getfullargspec(DynamicsModel.__init__).args}
    env_model = DynamicsModel(state_size=world_model_state_dim, 
                            action_size=world_model_action_dim,
                            seq_length=seq_len,
                            **env_model_kwargs)

    # init_local_wandb(**kwargs)
    test_csv_file_names, test_dfs_map = {}, {}
    env_model.env = world_model_env
    test_csv_file_names, test_dfs_map = test_dataset4_file_load.csv_file_names, test_dataset4_file_load.dfs_map
    env_model.eval(test_csv_file_names, test_dfs_map, 1, args.world_model_train_batch_size, **kwargs)



if __name__ == '__main__':
    model_eval()
    print('local evaluation success !')
