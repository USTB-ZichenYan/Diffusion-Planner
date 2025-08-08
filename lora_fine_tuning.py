import os
import torch
import argparse
import json
from diffusion_planner.model.diffusion_planner import Diffusion_Planner

class ColorFormatter:
    """ANSI颜色格式工具类"""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    END = "\033[0m"
    BOLD = "\033[1m"

    @classmethod
    def header(cls, text):
        return f"{cls.PURPLE}{cls.BOLD}=== {text} ==={cls.END}"

    @classmethod
    def success(cls, text):
        return f"{cls.GREEN}✓ {text}{cls.END}"

    @classmethod
    def warning(cls, text):
        return f"{cls.YELLOW}⚠ {text}{cls.END}"

    @classmethod
    def error(cls, text):
        return f"{cls.RED}✗ {text}{cls.END}"

    @classmethod
    def info(cls, text):
        return f"{cls.BLUE}→ {text}{cls.END}"

    @classmethod
    def param(cls, text):
        return f"{cls.CYAN}{text}{cls.END}"

def apply_lora(model, lora_config):
    """应用LoRA适配器到模型"""
    try:
        from peft import LoraConfig, get_peft_model
        
        peft_config = LoraConfig(
            r=lora_config.lora_rank,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.lora_target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias="none",
            modules_to_save=getattr(lora_config, 'modules_to_save', None),
        )
        return get_peft_model(model, peft_config)
        
    except ImportError:
        raise ImportError("PEET库未安装，请运行: pip install peft")

def _print_model_parameters(model, filter_key=None, max_lines=300, title=None):
    """增强版参数打印函数"""
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 打印标题和摘要
    if title:
        print(f"\n{ColorFormatter.header(title)}")
    print(f"{ColorFormatter.info(f'参数总数: {total_params:,} | 可训练: {trainable_params:,} ({trainable_params/total_params:.2%})')}")

    # 打印参数详情
    count = 0
    for name, param in model.named_parameters():
        if filter_key and filter_key not in name:
            continue
            
        color = ColorFormatter.GREEN if param.requires_grad else ColorFormatter.RED
        print(f"  {ColorFormatter.param(name):<{len(name)+2}} | "
            f"可训练: {color}{str(param.requires_grad):<5}{ColorFormatter.END} | "
            f"形状: {ColorFormatter.YELLOW}{tuple(param.shape)}{ColorFormatter.END}")
        
        count += 1
        if max_lines and count >= max_lines:
            remaining = sum(1 for _ in model.named_parameters() if (not filter_key or filter_key in _[0])) - count
            if remaining > 0:
                print(f"{ColorFormatter.warning(f'...（已显示{count}项，剩余{remaining}项未显示）')}")
            break

def freeze_base_model(model):
    """冻结基础模型参数（保留LoRA可训练）"""
    if not hasattr(model, 'print_trainable_parameters'):
        raise ValueError(ColorFormatter.error("非PEFT模型，请先应用LoRA"))
    
    _print_model_parameters(model, title="模型冻结前状态")
    
    # 执行冻结
    print(ColorFormatter.info("冻结基础模型参数..."))
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # 确保LoRA参数可训练
    for name, param in model.named_parameters():
        if any(key in name for key in ['lora_A', 'lora_B', 'lora_embedding']):
            param.requires_grad = True

    # 验证结果
    _print_model_parameters(model, 
                          title="模型冻结后状态",
                          filter_key='lora')
    
    if hasattr(model, 'print_trainable_parameters'):
        print(f"\n{ColorFormatter.header('PEFT官方统计')}")
        model.print_trainable_parameters()
    
    return model

def load_pretrained_model_with_lora(args_file, ckpt_file, lora_config):
    """完整模型加载流程（带增强打印）"""
    try:
        # 初始化阶段
        print(f"\n{ColorFormatter.header(f'开始加载模型 (配置: {args_file}, 权重: {ckpt_file}')}")
        
        # 加载配置
        with open(args_file, 'r') as f:
            args = argparse.Namespace(**json.load(f))
        
        # 初始化模型
        model = Diffusion_Planner(args)
        
        # 加载权重
        print(ColorFormatter.info("加载预训练权重..."))
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        _print_model_parameters(model, title="初始模型状态")
        
        # 应用LoRA
        print(ColorFormatter.info("应用LoRA适配器..."))
        model = apply_lora(model, lora_config)
        _print_model_parameters(model, 
                              title="LoRA应用后状态",
                              filter_key='lora',
                              max_lines=20)

        # 冻结处理
        if getattr(lora_config, 'freeze_base_model', True):
            model = freeze_base_model(model)
            print(ColorFormatter.success("基础模型冻结完成"))
        else:
            print(ColorFormatter.warning("基础模型保持未冻结状态"))
        
        return model
        
    except Exception as e:
        print(ColorFormatter.error(f"加载失败: {str(e)}"))
        raise

# 使用示例
if __name__ == "__main__":
    class LoRAConfig:
        lora_rank = 8
        lora_alpha = 16
        lora_dropout = 0.1
        lora_target_modules = ["attn", "mlp"]
        freeze_base_model = True
    
    try:
        model = load_pretrained_model_with_lora(
            "config.json",
            "model.ckpt",
            LoRAConfig()
        )
        print(ColorFormatter.success("模型加载成功！"))
    except Exception as e:
        print(ColorFormatter.error(f"错误: {e}"))