import torch

# 函数格式
def state_predict(states:torch.Tensor, actions:torch.Tensor, timesteps:torch.Tensor, mask:torch.Tensor, device, state_columns, action_columns) -> torch.Tensor:
    
    """
    
    输入状态时间序列(序列长度为20), 预测下一个时刻(t=21)的state;
    注意:输入输出均为原始状态

    Args:
        states (torch.tensor): 状态
        actions (torch.tensor): 动作
        timesteps (torch.tensor): 时间步
        mask (torch.tensor): 掩码
        state_columns: state中每一个维度的物理含义
        action_columns: action中每一个维度的物理含义
        device
    return:
        next_states(torch.tensor), 请使用float32类型的tensor
        
    example usage:
    
    states = torch.ones(batch_size, seq_length, state_dim)
    actions = torch.ones(batch_size, seq_length, action_dim)
    timesteps = torch.ones(batch_size, seq_length)
    masks = torch.ones(batch_size, seq_length)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_columns = state_columns = ['pos_boom', 'pos_arm', 'pos_swing', 'vel_boom', 'vel_arm', 'vel_swing', 'state_time', 'action_time']
    action_columns = ['pwm_boom', 'pwm_arm', 'pwm_swing']    
    next_state = state_predict(states, actions,timesteps, masks, device, state_columns, action_columns) -> torch.tensor(batch_size, 1, state_dim)

    """
