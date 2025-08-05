# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HYDRA_FULL_ERROR=1

# ###################################
# # User Configuration Section
# ###################################
# # Set environment variables
# export NUPLAN_DEVKIT_ROOT="REPLACE_WITH_NUPLAN_DEVIKIT_DIR"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
# export NUPLAN_DATA_ROOT="REPLACE_WITH_DATA_DIR"  # nuplan dataset absolute path (e.g. "/data")
# export NUPLAN_MAPS_ROOT="REPLACE_WITH_MAPS_DIR" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
# export NUPLAN_EXP_ROOT="REPLACE_WITH_EXP_DIR" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

#!/bin/bash
# 将当前目录添加到Python路径
export PYTHONPATH="/home/SENSETIME/yanzichen/Diffusion-Planner:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

###################################
# 用户配置部分
###################################
# 设置环境变量 - 请检查这些路径是否存在
export NUPLAN_DEVKIT_ROOT="/home/SENSETIME/yanzichen/nuplan-devkit"  # nuplan-devkit绝对路径
export NUPLAN_DATA_ROOT="/home/SENSETIME/yanzichen/nuplan/dataset"   # nuplan数据集绝对路径
export NUPLAN_MAPS_ROOT="/home/SENSETIME/yanzichen/nuplan/dataset/maps" # nuplan地图绝对路径
export NUPLAN_EXP_ROOT="/home/SENSETIME/yanzichen/nuplan/exp"  # nuplan实验绝对路径


# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="val14-collision"  # e.g., "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_nonreactive_agents"  # e.g., "closed_loop_nonreactive_agents"
###################################

BRANCH_NAME=collision
# ARGS_FILE=./checkpoints/args.json
# CKPT_FILE=./checkpoints/model.pth
ARGS_FILE=/home/SENSETIME/yanzichen/Diffusion-Planner/checkpoints/args.json
CKPT_FILE=/home/SENSETIME/yanzichen/Diffusion-Planner/checkpoints/model.pth

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi

echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner_guidance

# Use python from current environment instead of hardcoded path
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=128 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"