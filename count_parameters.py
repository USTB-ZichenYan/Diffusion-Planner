import sys
import os
import json
import tempfile

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config

def create_temp_config():
    # 创建一个临时配置文件
    config_dict = {
        # Trajectory prediction parameters
        "future_len": 80,                    # Length of future trajectory to predict (timesteps)
        "time_len": 21,                      # Length of historical observation timesteps
        
        # Agent (vehicle) related parameters
        "agent_state_dim": 11,               # Dimension of each agent's state vector (position, velocity, etc.)
        "agent_num": 32,                     # Maximum number of agents (vehicles) to consider in the scene
        
        # Static objects parameters
        "static_objects_state_dim": 10,      # Dimension of each static object's state representation
        "static_objects_num": 5,             # Maximum number of static objects to consider
        
        # Lane features parameters
        "lane_len": 20,                      # Number of points per lane segment
        "lane_state_dim": 12,                # Dimension of lane feature representation
        "lane_num": 70,                      # Maximum number of lane segments to consider
        
        # Route information parameters
        "route_len": 20,                     # Number of waypoints in the route
        "route_state_dim": 12,               # Dimension of route feature representation
        "route_num": 25,                     # Maximum number of route elements to consider
        
        # Model architecture parameters
        "encoder_drop_path_rate": 0.1,       # Drop path rate for encoder's stochastic depth
        "decoder_drop_path_rate": 0.1,       # Drop path rate for decoder's stochastic depth
        "encoder_depth": 3,                  # Number of transformer layers in the encoder
        "decoder_depth": 3,                  # Number of transformer layers in the decoder
        "num_heads": 6,                      # Number of attention heads in transformer layers
        "hidden_dim": 192,                   # Hidden dimension size for transformer layers
        
        # Diffusion model parameters
        "diffusion_model_type": "x_start",   # Type of diffusion model parameterization
        
        # Prediction parameters
        "predicted_neighbor_num": 10,        # Number of neighboring agents to predict trajectories for
        
        # Normalization parameters
        "state_normalizer": {
            "mean": [[[10, 0, 0, 0]] * 11],   # Mean values for state normalization
            "std": [[[1, 1, 1, 1]] * 11]     # Standard deviation values for state normalization
        },
        "observation_normalizer": {
            "test": {
                "mean": [[0, 0, 0]],         # Mean values for observation normalization during testing
                "std": [[1, 1, 1]]           # Standard deviation values for observation normalization during testing
            }
        },
        
        # Device configuration
        "device": "cpu"                      # Computation device (CPU or GPU)
    }
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config_dict, temp_file)
    temp_file.close()
    
    return temp_file.name

def count_parameters():
    # 创建临时配置文件
    temp_config_path = create_temp_config()
    
    try:
        # 创建配置对象
        config = Config(
            args_file=temp_config_path,
            guidance_fn=None
        )
        
        # 创建模型
        model = Diffusion_Planner(config)
        
        # 计算参数总数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # 分别计算编码器和解码器的参数
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Decoder parameters: {decoder_params:,}")
        
    finally:
        # 删除临时文件
        os.unlink(temp_config_path)
    
    return total_params

if __name__ == "__main__":
    count_parameters()