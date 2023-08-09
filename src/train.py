import inspect
import os
import time
import random
import itertools
import glob
import math
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import accumulate
from typing import Tuple, Union
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput
from concurrent.futures import ThreadPoolExecutor
from transformers import DecisionTransformerConfig, DecisionTransformerModel, DecisionTransformerGPT2Model
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from utils import get_env, readParser

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
device = accelerator.device


class WorldModelDecisionTransformerModel(DecisionTransformerModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.encoder = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(config.hidden_size, 1)

        self.post_init()

    def forward(
        self,
        states=None,
        actions=None,
        rewards=None,
        returns_to_go=None,
        timesteps=None,
        attention_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ) -> Union[Tuple, DecisionTransformerOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        stacked_inputs = (
            torch.stack((state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )
        device = stacked_inputs.device
        encoder_outputs = self.encoder(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = encoder_outputs[0]

        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        state_preds = self.predict_state(x[:, 1]) 
        if self.config.data_type == "delta":
            state_preds = state_preds + states

        action_preds = None
        return_preds = None

        if not return_dict:
            return (state_preds, action_preds, return_preds)

        return DecisionTransformerOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            state_preds=state_preds,
            action_preds=action_preds,
            return_preds=return_preds,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DynamicsModel():
    def __init__(self, state_size, action_size, reward_size=1, seq_length=20,
                 pred_hidden_size=200, **kwargs):
        self.arg_dict = kwargs
        self.seed = self.arg_dict.get("seed", 18)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.seq_len = seq_length
        self.learning_rate = kwargs.get("world_model_learning_rate", 1e-3)
        self.device = kwargs.get("device", device)
        self.data_type = kwargs.get("world_model_output", "delta")
        self.action_time_dim = 2  # state_time, action_time
        self.n_head = kwargs.get("pred_n_heads", 1)
        self.env_name = kwargs.get("env_name", "excavator")
        self.env = kwargs.get("env", None)
        self.world_model_train_batch_size = kwargs.get("world_model_train_batch_size", 1024)

        # 初始化部分变量
        self.model_update_times = 0
        self.total_cost_samples = 0
        self.total_cost_sequences = 0
        self.train_sample_count_for_fps = 0
        self.last_update_time = 0
        self.last_save_model_cost_sequences = 0
        # 初始化模型
        self.model_config = DecisionTransformerConfig(state_dim=self.state_size + self.action_time_dim,
                                                      act_dim=self.action_size,
                                                      max_ep_len=2**16,
                                                      hidden_size=pred_hidden_size,
                                                      n_head=self.n_head,
                                                      data_type=self.data_type)
        self.model = WorldModelDecisionTransformerModel(self.model_config)
        self.model = self.model.to(self.device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, 
                                                                             T_max=self.arg_dict["world_model_lr_schedule_T_max"], 
                                                                             eta_min=1e-6)
        self.model, self.model_optimizer, self.model_lr_scheduler = accelerator.prepare(self.model, 
                                                                                        self.model_optimizer, 
                                                                                        self.model_lr_scheduler)


    def rad_norm(self, pos_states):
        # convert the rad to -math.pi ~ math.pi
        return (pos_states + math.pi) % (2 * math.pi) - math.pi

    def get_joint_pos(self, states):

        new_states = states[:, :len(self.env.policy_joints)]
        new_states = torch.cat([new_states[:, i, None] * (
            self.env.pos_min_max_dic[joint_name][1] - self.env.pos_min_max_dic[joint_name][0]) +
                                        self.env.pos_min_max_dic[joint_name][0] for i, joint_name in
                                        enumerate(self.env.policy_joints)], axis=1)
        new_states = torch.cat(
            [new_states[:, :2], torch.zeros_like(new_states[:, 1, None]),
             new_states[:, 2, None]], axis=1)

        return self.rad_norm(new_states)

    def joint_dis_loss(self, real_current_s, real_next_s, prediction_s, attention_mask):
        real_current_state, predict_current_state, real_next_state, predict_next_state, mask = self.get_current_and_next_states(real_current_s, real_next_s, prediction_s, attention_mask)
        predict_end_location_dis_2norm = torch.norm(real_next_state - predict_next_state, dim=-1)
        loss = 0.5 * torch.mean(torch.pow(predict_end_location_dis_2norm, 2))
        return loss

    def get_current_and_next_states(self, real_current_s, real_next_s, prediction_s, attention_mask):

        if self.arg_dict["world_model_loss_scale"] == "last":
            real_current_state = real_current_s[:, -1, :self.state_size]
            predict_current_state = None
            real_next_state = real_next_s[:, -1, :self.state_size]
            predict_next_state = prediction_s[:, -1, :self.state_size]
            mask = attention_mask[:, -1]
        else: # self.arg_dict["world_model_loss_scale"] == "all":
            real_current_state = real_current_s[:, :, :self.state_size]
            predict_current_state = None
            real_next_state = real_next_s[:, :, :self.state_size]
            predict_next_state = prediction_s[:, :, :self.state_size]
            mask = attention_mask
        mask = mask.reshape(-1)
        real_current_state = real_current_state.reshape(-1, self.state_size)[mask > 0]
        real_next_state = real_next_state.reshape(-1, self.state_size)[mask > 0]
        predict_next_state = predict_next_state.reshape(-1, self.state_size)[mask > 0]
        return real_current_state, predict_current_state, real_next_state, predict_next_state, mask

    def model_loss(self, real_current_s, real_next_s, prediction_s, attention_mask):
        loss = self.joint_dis_loss(real_current_s, real_next_s,  prediction_s, attention_mask)
        return loss

    def model_train(self, real_current_s, real_next_s, prediction_s, attention_mask):

        loss = self.model_loss(real_current_s, real_next_s, prediction_s, attention_mask)
        self.model_optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg_dict["world_model_clip_grad_norm"])
        self.model_optimizer.step()
        return loss

    def accelerate_distributed_train(self, train_csv_file_names, train_dfs_map, test_csv_file_names, test_dfs_map, max_data_num, batch_size, **kwargs):
        self.last_update_time = time.time()
        global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
        global_world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        test_metric_list_dict = None
        for epoch in itertools.count():
            if kwargs.get("epochs", None) is not None and epoch >= kwargs["epochs"]:
                break
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}][EnsembleDynamicsModel] "
                    f"global rank/global world size: {global_rank}/{global_world_size}, world model epoch: {epoch}, "
                    f"accelerator.num_processes: {accelerator.num_processes}")
            num_workers = min(8, max(4, accelerator.num_processes))
            train_dataset = IterDataset(self.env, "train", train_csv_file_names, train_dfs_map, **kwargs)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=math.ceil(max_data_num/batch_size) * batch_size,
                                                           pin_memory=True, drop_last=True, num_workers=num_workers, prefetch_factor=5)
            train_dataloader = accelerator.prepare(train_dataloader)
            get_batch_start_time = time.time()
            batch_cnt = 0
            test_frequency = 50
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}][EnsembleDynamicsModel] "
                  f"Set test_frequency every {test_frequency} batches, {test_frequency * (math.ceil(max_data_num/batch_size) * batch_size)} samples.")
            for s, a, r, next_s, d, rtg, timesteps, mask in train_dataloader:
                print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}][EnsembleDynamicsModel] global rank/global world size: "
                      f"{global_rank}/{global_world_size}, epoch: {epoch}, train input shape: {s.shape}, train label shape: {s.shape}, "
                      f"train_dataset len: {len(train_dataset)}, time for get one batch data: {time.time() - get_batch_start_time} seconds")
                epoch_log_dict = {}
                for mini_epoch in range(1):
                    mini_batch_cost_samples = 0
                    for train_data_idx in range(math.ceil(max_data_num / batch_size)):
                        self.model.train()
                        train_s = s[train_data_idx * batch_size: (train_data_idx + 1) * batch_size].float().to(self.device)
                        train_a = a[train_data_idx * batch_size: (train_data_idx + 1) * batch_size].float().to(self.device)
                        train_r = r[train_data_idx * batch_size: (train_data_idx + 1) * batch_size].float().to(self.device)
                        train_next_s = next_s[train_data_idx * batch_size: (train_data_idx + 1) * batch_size].float().to(self.device)
                        train_d = d[train_data_idx * batch_size: (train_data_idx + 1) * batch_size].float().to(self.device)
                        train_rtg = rtg[train_data_idx * batch_size: (train_data_idx + 1) * batch_size].float().to(self.device)
                        train_timesteps = timesteps[train_data_idx * batch_size: (train_data_idx + 1) * batch_size].long().to(self.device)
                        train_mask = mask[train_data_idx * batch_size: (train_data_idx + 1) * batch_size].float().to(self.device)
                        output = self.model(**{
                            "states": train_s,
                            "actions": train_a,
                            "rewards": train_r,
                            "returns_to_go": train_rtg,
                            "timesteps": train_timesteps,
                            "attention_mask": train_mask,
                        })
                        prediction_s = output["state_preds"]
                        loss = self.model_train(train_s, train_next_s, prediction_s, train_mask)
                        if accelerator.is_main_process:
                            num_processes = accelerator.num_processes
                            self.model_update_times += 1
                            self.train_sample_count_for_fps += (batch_size * num_processes)
                            self.total_cost_sequences += (batch_size * num_processes)
                            self.total_cost_samples += (batch_size * num_processes * self.arg_dict['state_seq_length'])
                            mini_batch_cost_samples += batch_size * num_processes
                            epoch_log_dict = self.updata_log_dict(epoch_log_dict, {"train_mse_losses": loss.detach().cpu().numpy().item()})
                            log_str = self.log2str(epoch_log_dict, mode="train")
                            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}][EnsembleDynamicsModel] epoch {epoch},"
                                  f'batch_cnt {batch_cnt}, mini epoch {mini_epoch}, {log_str},'
                                  f'training costs samples num {mini_batch_cost_samples} / {s.shape[0] * global_world_size}')
                # test the model every 50 batches
                if (batch_cnt + 1) % test_frequency == 0 or (batch_cnt == 0 and epoch != 0):
                    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}][EnsembleDynamicsModel] global rank/global"
                          f"world size: {global_rank}/{global_world_size}, batch_cnt {batch_cnt} test start.")
                    self.model.eval()
                    test_cost_samples = 0
                    test_num_workers = 2
                    test_dataset = IterDataset(self.env, "test", test_csv_file_names, test_dfs_map, **kwargs)
                    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                                                                  drop_last=True, num_workers=test_num_workers)
                    test_dataloader = accelerator.prepare(test_dataloader)
                    for s, a, r, next_s, d, rtg, timesteps, mask in test_dataloader:
                        test_s = s.float().to(self.device)
                        test_a = a.float().to(self.device)
                        test_r = r.float().to(self.device)
                        test_d = d.float()
                        test_next_s = next_s.float().to(self.device)
                        test_rtg = rtg.float().to(self.device)
                        test_timesteps = timesteps.long().to(self.device)
                        test_mask = mask.float().to(self.device)
                        with torch.no_grad():
                            test_output = self.model(**{
                                "states": test_s,
                                "actions": test_a,
                                "rewards": test_r,
                                "returns_to_go": test_rtg,
                                "timesteps": test_timesteps,
                                "attention_mask": test_mask,
                            })
                        test_prediction_s = test_output["state_preds"]
                        # test_prediction_a = test_output["action_preds"]
                        test_loss = self.model_loss(test_s, test_next_s, test_prediction_s, test_mask)
                        # gather data from multi-gpus (used when in ddp mode)
                        test_s = accelerator.gather(test_s)
                        test_a = accelerator.gather(test_a)
                        test_mask = accelerator.gather(test_mask)
                        test_next_s = accelerator.gather(test_next_s)
                        test_loss = accelerator.gather_for_metrics(test_loss)
                        test_prediction_s = accelerator.gather_for_metrics(test_prediction_s)
                        if accelerator.is_main_process:
                            epoch_log_dict = self.updata_log_dict(epoch_log_dict, 
                                                                  {"test_total_loss": np.mean(test_loss.detach().cpu().numpy())})
                            test_cost_samples += (batch_size * accelerator.num_processes)
                            log_str = self.log2str(epoch_log_dict, mode="test")
                            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}][EnsembleDynamicsModel] "
                                  f"epoch {epoch}, batch_cnt {batch_cnt}, {log_str}, test costs samples num {test_cost_samples} / {len(test_dataset)}")

                if accelerator.is_main_process:
                    self.save_accelerate_model()
                batch_cnt += 1
                get_batch_start_time = time.time()
            if accelerator.is_main_process:
                self.model_lr_scheduler.step()

    def save_accelerate_model(self, force_save=False):
        if self.total_cost_sequences - self.last_save_model_cost_sequences < self.arg_dict["save_model_interval"] and not force_save:
            print(f"Break but not saved, total_cost_sequences {self.total_cost_sequences}, "
                  f"last_save_model_cost_sequences {self.last_save_model_cost_sequences}, "
                  f"save_model_interval {self.arg_dict['save_model_interval']}")
            return
        world_model_dir = "./world_models"
        run_name = "default"
        world_model_path = os.path.join(world_model_dir, run_name)
        os.makedirs(world_model_path, exist_ok=True)
        world_model_file_path = os.path.join(world_model_path,
                                             f"world_model_{str(self.total_cost_samples//self.arg_dict['save_model_interval']).zfill(5)}_"
                                             f"{str(self.model_update_times//1000).zfill(5)}k.pth")
        print('Saving world_models to {}'.format(world_model_file_path))
        unwrap_model = accelerator.unwrap_model(self.model)
        unwrap_optim = accelerator.unwrap_model(self.model_optimizer)
        unwarp_lr_scheduler = accelerator.unwrap_model(self.model_lr_scheduler)
        torch.save({'model_state': unwrap_model.state_dict()},
                    # 'optim_state': unwrap_optim.state_dict(),
                    # 'lr_scheduler_state': unwarp_lr_scheduler.state_dict()},
                    world_model_file_path)
        print(f'Save world_models to {world_model_file_path}')
        self.last_save_model_cost_sequences = self.total_cost_sequences

    def load_accelerate_model(self, world_model_path):
        ckpt = torch.load(world_model_path)
        new_cpkt = {}
        for k, v in ckpt.items():
            new_state_dict = {}
            for key, value in v.items():
                # 添加 'module.' 前缀
                new_key = 'module.' + key
                new_state_dict[new_key] = value
            new_cpkt[k] = new_state_dict
        self.model.load_state_dict(new_cpkt['model_state'])
        print(f"Load world model from {world_model_path} successfully.")

    def updata_log_dict(self, old_log_dict, new_log_dict):
        for key, value in new_log_dict.items():
            old_log_dict.setdefault(key, []).append(value)
        return old_log_dict

    def log2str(self, log_dict, mode=None, custom_str=None, calculate_mean=False):
        log_str = ""
        for key, value in log_dict.items():
            if (key.endswith("total_loss") or key.endswith("mse_losses")) and (mode is None or mode in key) and (custom_str is None or custom_str in key):
                if calculate_mean:
                    log_str += f"{key.split('/')[-1]}: {np.mean(value):.4f} "
                else:
                    log_str += f"{key.split('/')[-1]}: {value[-1]:.4f} "
        return log_str

    def get_params_num(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


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
        # normalize the data
        for idx, joint_name in enumerate(env.policy_joints):
            df[f"pos_{joint_name}"] = df[f"pos_{joint_name}"].apply(lambda x: (x - env.pos_min_max_dic[joint_name][0]) / (env.pos_min_max_dic[joint_name][1] - env.pos_min_max_dic[joint_name][0]))
            df[f"vel_{joint_name}"] = df[f"vel_{joint_name}"].apply(lambda x: (x - env.vel_min_max_dic[joint_name][0]) / (env.vel_min_max_dic[joint_name][1] - env.vel_min_max_dic[joint_name][0]))
            df[f"next_pos_{joint_name}"] = df[f"next_pos_{joint_name}"].apply(lambda x: (x - env.pos_min_max_dic[joint_name][0]) / (env.pos_min_max_dic[joint_name][1] - env.pos_min_max_dic[joint_name][0]))
            df[f"next_vel_{joint_name}"] = df[f"next_vel_{joint_name}"].apply(lambda x: (x - env.vel_min_max_dic[joint_name][0]) / (env.vel_min_max_dic[joint_name][1] - env.vel_min_max_dic[joint_name][0]))
            df[f"pwm_{joint_name}"] = df[f"pwm_{joint_name}"].apply(lambda x: 0 if x == 0
                                                                    else ((x-env.pwm_min_max[joint_name]["positive"][0])/(env.pwm_min_max[joint_name]["positive"][1]-env.pwm_min_max[joint_name]["positive"][0])*(x >= 0)
                                                                          + (x - env.pwm_min_max[joint_name]["negative"][1])/(env.pwm_min_max[joint_name]["negative"][1]-env.pwm_min_max[joint_name]["negative"][0])*(x < 0)))

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

def main_accelerate():
    torch.set_printoptions(precision=7)
    args_parser = readParser()
    args = args_parser.parse_args()
    kwargs = vars(args)

    # Set random seed
    torch.manual_seed(kwargs["seed"])
    np.random.seed(kwargs["seed"])
    random.seed(kwargs["seed"])

    world_model_env = get_env(**kwargs)
    print(f"env init finished, env seq_length: {world_model_env.seq_length}")
    # 预载入数据集
    train_dataset4_file_load = IterDataset(world_model_env, "train", None, None, **kwargs)
    test_dataset4_file_load = IterDataset(world_model_env, "test", None, None, **kwargs)

    total_train_data_size = len(train_dataset4_file_load)
    kwargs["total_train_data_size"] = total_train_data_size
    print(f"Total train data size: {total_train_data_size}")
    kwargs["device"] = accelerator.device
    print(f"Using device: {kwargs['device']}")
    # 使用了叠帧以后，初始化world model时，不能用policy的state_dim和action_dim
    world_model_state_dim = world_model_env.joint_num * 2
    world_model_action_dim = world_model_env.joint_num
    seq_len = args.state_seq_length
    env_model_kwargs = {k: v for k, v in kwargs.items() if
                        k not in inspect.getfullargspec(DynamicsModel.__init__).args}
    env_model = DynamicsModel(state_size=world_model_state_dim, action_size=world_model_action_dim,
                              reward_size=kwargs["reward_size"],
                              seq_length=seq_len,
                              pred_hidden_size=kwargs["pred_hidden_size"],
                              **env_model_kwargs)
    print(f"World model init finished, params number: {env_model.get_params_num()}")
    if os.path.exists(args.load_world_model):
        env_model.load_accelerate_model(args.load_world_model)
    world_model_datalaoder_batch_size = args.world_model_datalaoder_batch_size

    train_csv_file_names, train_dfs_map = train_dataset4_file_load.csv_file_names, train_dataset4_file_load.dfs_map
    test_csv_file_names, test_dfs_map = test_dataset4_file_load.csv_file_names, test_dataset4_file_load.dfs_map
    env_model.env = world_model_env

    env_model.accelerate_distributed_train(train_csv_file_names, train_dfs_map,
                                           test_csv_file_names, test_dfs_map, world_model_datalaoder_batch_size,
                                           args.world_model_train_batch_size, **kwargs)

    accelerator.wait_for_everyone()
    accelerator.clear()
    if "env_model" in locals():
        del env_model
    if "train_dataset4_file_load" in locals():
        del train_dataset4_file_load
    if "test_dataset4_file_load" in locals():
        del test_dataset4_file_load


if __name__ == '__main__':
    main_accelerate()
