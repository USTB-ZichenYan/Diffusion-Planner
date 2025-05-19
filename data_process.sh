###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/home/SENSETIME/yanzichen/nuplan/dataset/nuplan-v1.1/splits/mini" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/home/SENSETIME/yanzichen/nuplan/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/preprocess_training_data" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 100 \
