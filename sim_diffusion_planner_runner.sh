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

# 验证路径是否存在
if [ ! -d "$NUPLAN_DEVKIT_ROOT" ]; then
    echo "错误: NUPLAN_DEVKIT_ROOT 不存在: $NUPLAN_DEVKIT_ROOT"
    exit 1
fi

if [ ! -d "$NUPLAN_DATA_ROOT" ]; then
    echo "错误: NUPLAN_DATA_ROOT 不存在: $NUPLAN_DATA_ROOT"
    exit 1
fi

if [ ! -d "$NUPLAN_MAPS_ROOT" ]; then
    echo "错误: NUPLAN_MAPS_ROOT 不存在: $NUPLAN_MAPS_ROOT"
    exit 1
fi

if [ ! -d "$NUPLAN_EXP_ROOT" ]; then
    echo "创建 NUPLAN_EXP_ROOT 目录: $NUPLAN_EXP_ROOT"
    mkdir -p "$NUPLAN_EXP_ROOT"
fi

# 数据集分割选项
# 可选项: "test14-random", "test14-hard", "val14"
SPLIT="val14"

# 挑战类型
# 可选项: "closed_loop_nonreactive_agents", "closed_loop_reactive_agents"
CHALLENGE="closed_loop_nonreactive_agents"
###################################

BRANCH_NAME=diffusion_planner_release
ARGS_FILE=/home/SENSETIME/yanzichen/Diffusion-Planner/checkpoints/args.json
CKPT_FILE=/home/SENSETIME/yanzichen/Diffusion-Planner/checkpoints/model.pth

# 验证检查点文件是否存在
if [ ! -f "$ARGS_FILE" ]; then
    echo "错误: ARGS_FILE 不存在: $ARGS_FILE"
    exit 1
fi

if [ ! -f "$CKPT_FILE" ]; then
    echo "错误: CKPT_FILE 不存在: $CKPT_FILE"
    exit 1
fi

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi

echo "正在处理 $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner

# 检查python命令是否存在
if ! command -v python &> /dev/null; then
    echo "错误: Python未安装或不在PATH中"
    exit 1
fi

# 检查run_simulation.py文件是否存在
if [ ! -f "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" ]; then
    echo "错误: run_simulation.py 不存在: $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py"
    exit 1
fi

# 运行模拟并进行错误处理
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
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"