# from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiplicativeLR

# def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1):
#     assert epoch >= warm_up_epoch
#     T_warmup = warm_up_epoch
    
#     warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
#     fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
#     scheduler = SequentialLR(optimizer, 
#                              schedulers=[warmup_scheduler, fixed_scheduler], 
#                              milestones=[T_warmup])
    
#     return scheduler

# 导入兼容低版本的模块
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, CosineAnnealingLR
import numpy as np
def CosineAnnealingWarmUpRestarts(optimizer, total_epochs, warm_up_epochs, start_factor=0.1):
    """
    实现带有预热阶段的余弦退火学习率调度器
    
    参数:
    optimizer: 优化器
    total_epochs: 总训练轮数
    warm_up_epochs: 预热轮数
    start_factor: 预热起始学习率因子
    """
    # 定义预热阶段的学习率调整函数
    def warmup_lr_lambda(epoch):
        if epoch < warm_up_epochs:
            # 线性预热: 从 start_factor * base_lr 到 base_lr
            return start_factor + (1.0 - start_factor) * epoch / max(1, warm_up_epochs - 1)
        else:
            # 余弦退火: 在预热后应用
            progress = (epoch - warm_up_epochs) / max(1, total_epochs - warm_up_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    # 创建LambdaLR调度器
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    
    return scheduler