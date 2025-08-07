import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型和输入张量
model = SimpleModel()
input_tensor = torch.randn(1, 10)  # 随机输入

# 使用TensorBoard记录计算图
writer = SummaryWriter(log_dir='runs/calculation_graph')
writer.add_graph(model, input_tensor)
writer.close()

# 启动TensorBoard查看（在命令行执行）
# tensorboard --logdir=runs/calculation_graph






