# # 配置环境变量
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 指定所有可用GPU，与nproc-per-node匹配

# ###################################
# # User Configuration Section
# ###################################
# RUN_PYTHON_PATH="/home/SENSETIME/yanzichen/anaconda3/envs/nuplan/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# # Set training data path
# TRAIN_SET_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/preprocess_training_data" # preprocess data using data_process.sh
# TRAIN_SET_LIST_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/diffusion_planner_training.json"
# ###################################

# # sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
# # --train_set  $TRAIN_SET_PATH \
# # --train_set_list  $TRAIN_SET_LIST_PATH \

# python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 --standalone train_predictor.py \
# --train_set  $TRAIN_SET_PATH \
# --train_set_list  $TRAIN_SET_LIST_PATH \


###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/home/SENSETIME/yanzichen/anaconda3/envs/nuplan/bin/python"
TRAIN_SET_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/preprocess_training_data"
TRAIN_SET_LIST_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/diffusion_planner_training.json"
###################################

# 单GPU训练命令（非分布式）
$RUN_PYTHON_PATH train_predictor.py \
--train_set "$TRAIN_SET_PATH" \
--train_set_list "$TRAIN_SET_LIST_PATH" \
--use_data_augment False \
--ddp False \
--device cuda \
--batch_size 1 \
--learning_rate 5e-4  # 可选：指定学习率等其他参数