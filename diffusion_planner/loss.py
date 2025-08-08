from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],

    futures: Tuple[torch.Tensor, torch.Tensor],
    
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    eps: float = 1e-3,
):   
    """
    计算扩散模型的损失函数，包含智能体（ego）和周围车辆（neighbors）的预测损失。

    参数:
        model (nn.Module): 神经网络模型用于前向传播。
        inputs (Dict[str, torch.Tensor]): 输入数据，包括智能体当前状态、邻居车辆的历史轨迹等。
        marginal_prob (Callable[[torch.Tensor], torch.Tensor]): 边际概率函数，输入状态返回对应的概率均值和标准差。
        futures (Tuple[torch.Tensor, torch.Tensor]): 包含未来状态和掩码的信息：(ego_future, neighbors_future, neighbor_future_mask)。
        norm (StateNormalizer): 状态归一化器，用于对轨迹进行标准化。
        loss (Dict[str, Any]): 存储损失值的字典，输出时会更新其中的内容。
        model_type (str): 模型类型，支持 "score" 和 "x_start"。
        eps (float, optional): 随机时间 t 的最小偏移量，默认为 1e-3。

    返回:
        loss (Dict[str, Any]): 更新后的损失字典，包含 ego_planning_loss 和 neighbor_prediction_loss。
        decoder_output (Dict[str, torch.Tensor]): 解码器的输出结果。
    """
    # 解包输入的未来轨迹信息
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask # [B, P, V]

    # 提取张量维度并构造 mask
    B, Pn, T, _ = neighbors_future.shape
    ego_current, neighbors_current = inputs["ego_current_state"][..., :4], inputs["neighbor_agents_past"][..., :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.cat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    # 构建 ground truth future 轨迹以及当前状态
    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1) # [B, P = 1 + 1 + neighbor, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1) # [B, P, 4]

    # 随机采样时间步 t 并生成噪声 z
    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps # [B,]
    z = torch.randn_like(gt_future, device=gt_future.device) # [B, P, T, 4]
    
    # 构造完整 ground truth，并应用 normalization
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    # 计算扩散过程中的均值和标准差，并调整标准差维度以匹配后续计算
    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

    # 根据均值和标准差生成 xT
    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    
    # 构造新的输入字典，包含采样轨迹和扩散时间
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT.requires_grad_(True),
        "diffusion_time": t.requires_grad_(False)
    }

    # 前向传播模型获取解码器输出
    _, decoder_output = model(merged_inputs) # [B, P, 1 + T, 4]
    score = decoder_output["score"][..., 1:, :] # [B, P, T, 4]

    # 字典基本信息
    print(f"字典键数量: {len(merged_inputs)}")
    print(f"所有键名: {list(merged_inputs.keys())}")
    print_inputs_grad_status(merged_inputs, "merged_inputs")
    print_inputs_grad_status(decoder_output, "score")

    # 根据模型类型选择不同的损失计算方式
    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :])**2, dim=-1)
    
    # 应用 mask 并计算邻近车辆的预测损失
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    # 计算 ego 的规划损失
    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    # 断言检查损失中没有 NaN 值
    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    print(f"Model requires grad: {any(p.requires_grad for p in model.parameters())}")
    print(f"Input requires grad: {merged_inputs['sampled_trajectories'].requires_grad}")
    print(f"Loss requires grad: {loss['ego_planning_loss'].requires_grad}")

    return loss, decoder_output

def print_inputs_grad_status(input_dict, name="输入字典"):
    """
    打印输入字典中所有张量的梯度状态（带颜色和格式优化）
    
    Args:
        input_dict: 要检查的输入字典
        name: 字典的标识名称
    """
    # ANSI颜色代码
    COLOR_RED = "\033[91m"
    COLOR_GREEN = "\033[92m"
    COLOR_YELLOW = "\033[93m"
    COLOR_BLUE = "\033[94m"
    COLOR_END = "\033[0m"
    
    print(f"\n{COLOR_BLUE}=== 检查 {name} 的梯度状态 ==={COLOR_END}")
    
    # 统计信息
    tensor_count = 0
    grad_enabled_count = 0
    
    # 遍历字典
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            if value.requires_grad:
                grad_enabled_count += 1
            
            # 高亮关键信息
            key_str = f"{COLOR_YELLOW}{key}{COLOR_END}"
            grad_status = f"{COLOR_GREEN}需要梯度{COLOR_END}" if value.requires_grad else f"{COLOR_RED}无梯度{COLOR_END}"
            shape_str = f"{COLOR_BLUE}{tuple(value.shape)}{COLOR_END}"
            dtype_str = f"{str(value.dtype):<10}"
            
            print(f"  {key_str:<25} | 状态: {grad_status:<15} | 形状: {shape_str:<15} | 类型: {dtype_str}")
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            print_inputs_grad_status(value, f"{name}.{key}")
    
    # 打印统计摘要
    print(f"\n{COLOR_BLUE}=== 统计摘要 ===")
    print(f"总张量数: {tensor_count}")
    print(f"需要梯度的张量: {grad_enabled_count} ({grad_enabled_count/max(1,tensor_count):.0%}){COLOR_END}")