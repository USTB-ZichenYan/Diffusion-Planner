import os
import torch
import argparse
import json
from diffusion_planner.model.diffusion_planner import Diffusion_Planner

def apply_lora(model, lora_config):
    """
    Apply LoRA to the model based on the provided configuration.
    
    Args:
        model: The model to apply LoRA to
        lora_config: Configuration containing LoRA parameters
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("To use LoRA, please install the `peft` library: `pip install peft`")
    
    # Define LoRA configuration
    peft_config = LoraConfig(
        r=lora_config.lora_rank,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.lora_target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias="none",
        modules_to_save=lora_config.modules_to_save if hasattr(lora_config, 'modules_to_save') else None,
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    return model

def freeze_base_model(model):
    """
    优化的基础模型冻结函数，使用PEFT库原生方法
    
    Args:
        model: 已应用LoRA的PEFT模型
    """
    # 检查是否是PEFT模型
    if not hasattr(model, 'print_trainable_parameters'):
        raise ValueError("模型不是PEFT模型，请先应用LoRA")
    
    # 冻结基础模型参数
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # 确保LoRA参数可训练
    for name, param in model.named_parameters():
        if any(key in name for key in ['lora_A', 'lora_B', 'lora_embedding']):
            param.requires_grad = True
    
    return model

def load_pretrained_model_with_lora(args_file, ckpt_file, lora_config):
    """
    加载预训练模型并应用LoRA，使用优化的冻结逻辑
    """
    try:
        # 加载参数文件
        with open(args_file, 'r') as f:
            args_dict = json.load(f)
        args = argparse.Namespace(**args_dict)
        
        # 初始化模型
        model = Diffusion_Planner(args)
        
        # 加载预训练权重
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
        # 应用LoRA
        model = apply_lora(model, lora_config)
        
        # 冻结基础模型（使用优化后的方法）
        if getattr(lora_config, 'freeze_base_model', True):
            model = freeze_base_model(model)
            print("基础模型已冻结，仅LoRA参数可训练")
            
            # 使用PEFT原生方法验证可训练参数
            print("PEFT模型可训练参数:")
            model.print_trainable_parameters()
        else:
            print("基础模型和LoRA参数均可训练")
        
        return model
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise

def get_optimizer(model, lora_config):
    """
    Create an optimizer that only optimizes the LoRA parameters.
    
    Args:
        model: The model with LoRA applied
        lora_config: Configuration containing LoRA parameters
    """
    # Collect only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Use AdamW optimizer which is commonly used for transformer models
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lora_config.learning_rate,
        weight_decay=lora_config.weight_decay if hasattr(lora_config, 'weight_decay') else 0.01
    )
    
    return optimizer

def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: The model to analyze
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
    return trainable_params, all_param

