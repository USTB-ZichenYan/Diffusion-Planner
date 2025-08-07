
###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/home/SENSETIME/yanzichen/anaconda3/envs/nuplan/bin/python"
TRAIN_SET_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/preprocess_training_data"
TRAIN_SET_LIST_PATH="/home/SENSETIME/yanzichen/Diffusion-Planner/diffusion_planner_training.json"
###################################

# 单GPU训练命令（非分布式）使用LoRA
$RUN_PYTHON_PATH train_predictor.py \
--train_set "$TRAIN_SET_PATH" \
--train_set_list "$TRAIN_SET_LIST_PATH" \
--use_data_augment False \
--ddp False \
--device cuda \
--batch_size 1 \
--use_lora True \
--lora_rank 8 \
--lora_alpha 16.0 \
--lora_dropout 0.05 \
--lora_target_modules out_proj \
--freeze_base_model True \
--lora_lr 3e-4 \
--learning_rate 5e-4 \
--save_dir ./lora_finetuning_results