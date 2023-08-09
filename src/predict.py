import torch.nn as nn
from transformers import DecisionTransformerConfig, DecisionTransformerModel, DecisionTransformerGPT2Model
import torch
from typing import Tuple, Union
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerOutput
import math
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

        state_preds = self.predict_state(x[:, 1])  # predict next state given state and action
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


state_size = 6
action_time_dim = 2
action_size = 3
pred_hidden_size = 256
n_head = 1
data_type = 'delta'
model_config = DecisionTransformerConfig(state_dim=state_size+action_time_dim,
                                         act_dim=action_size,
                                         max_ep_len=2**16,
                                         hidden_size=pred_hidden_size,
                                         n_head=n_head,
                                         data_type=data_type)
model = WorldModelDecisionTransformerModel(model_config)
# load model if needed 
# from pathlib import Path
# model.eval()
# world_model_path = str(Path(__file__).parent.absolute() / 'model.pth') # note: 这里需要引入相对路径
# ckpt = torch.load(world_model_path)
# model.load_state_dict(ckpt['model_state'])

pos_min_max_dic = {'boom': [-math.pi, math.pi], 'arm': [-math.pi, math.pi],
                        'bucket': [-math.pi, math.pi], 'swing': [-math.pi, math.pi]}
vel_min_max_dic = {'boom': [-0.4, 0.5], 'arm': [-0.55, 0.65],
                        'bucket': [-1.0, 1.0], 'swing': [-0.82, 0.82]}
pwm_min_max = {'boom': {"negative": [-800, -250], "positive": [250, 800]},
                    'arm': {"negative": [-800, -250], "positive": [250, 800]},
                    'bucket': {"negative": [-600, -250], "positive": [250, 600]},
                    'swing': {"negative": [-450, -180], "positive": [180, 450]}}

def state_predict(states, actions, timesteps, masks, device, state_columns, action_columns):
    
    # note: 由于输入的是数据集中的原始数据，因此如果在训练过程中对数据做了归一化处理，则需要在该函数中对数据做归一化处理
    for index, key in enumerate(state_columns):
        value = states[:,:,index]
        type, joint = key.split('_')
        if type == 'pos':
            states[:,:,index] = (value - pos_min_max_dic[joint][0]) / (pos_min_max_dic[joint][1] - pos_min_max_dic[joint][0])
        elif type == 'vel':
            states[:,:,index] = (value - vel_min_max_dic[joint][0]) / (vel_min_max_dic[joint][1] - vel_min_max_dic[joint][0])
    for index, key in enumerate(action_columns):
        value = actions[:,:,index]
        _, joint = key.split('_')
        actions[:,:,index] = ((value - pwm_min_max[joint]["positive"][0]) / (pwm_min_max[joint]["positive"][1] - pwm_min_max[joint]["positive"][0]) * (value > 0)
                            + (value - pwm_min_max[joint]["negative"][0]) / (pwm_min_max[joint]["negative"][1] - pwm_min_max[joint]["negative"][0]) * (value < 0))
        
    states = states.to(device)
    actions = actions.to(device)
    timesteps = timesteps.to(device)
    masks = masks.to(device)
    model.to(device)
    model_output = model(states, actions, timesteps, masks)
    next_state = model_output['state_preds'][:,-1,:].unsqueeze(1)
    # 如果输出的是归一化后的数据，则需要做逆归一化的操作
    for index, key in enumerate(state_columns):
        value = next_state[:,:,index]
        type, joint = key.split('_')
        if type == 'pos':
            next_state[:,:,index] = value * (pos_min_max_dic[joint][1] - pos_min_max_dic[joint][0]) + pos_min_max_dic[joint][0]
        elif type == 'vel':
            next_state[:,:,index] = value * (vel_min_max_dic[joint][1] - vel_min_max_dic[joint][0]) + vel_min_max_dic[joint][0]
            
    return next_state
