export MODEL_NAME='stabilityai/stable-diffusion-2-1-base'
export PROJ_NAME='add_v21_base'
export EXP_NAME='init_setting_bs384'
export DATA_ROOT='../generated_data'


accelerate launch --multi_gpu --mixed_precision fp16 --num_machines 1 --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7  train_add_sdv21.py \
    --base_model=$MODEL_NAME \
    --mixed_precision=fp16 \
    --G_lr=1e-6 \
    --D_lr=1e-6 \
    --max_train_steps=100000 \
    --dataloader_num_workers=0 \
    --dataset_root=$DATA_ROOT \
    --data_pkl_name='summary.pkl' \
    --validation_steps=500 \
    --checkpointing_steps=2500 \
    --train_batch_size=8 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=6 \
    --report_to=wandb \
    --seed=1 \
    --project_name=${PROJ_NAME} \
    --exp_name=${EXP_NAME} \
    --zero_snr \
    --num_ts 1 \
    --multiscale_D \
    --misaligned_pairs_D
