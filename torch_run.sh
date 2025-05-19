export CUDA_VISIBLE_DEVICES=1

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/home/SENSETIME/yanzichen/anaconda3/envs/nuplan/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/preprocess_training_data" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/diffusion_planner_training.json"
###################################

sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \

python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \