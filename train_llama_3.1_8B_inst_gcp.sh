export HF_DATASETS_CACHE="/workspace/cache"
export TOKENIZERS_PARALLELISM="False"

export BASE_DIR="/your/workspace"


export TRAIN_DATA="/your/train/data/after/parsing.train"
export EVAL_DATA="/your/train/data/after/parsing.eval"


export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_PATH="$BASE_DIR/models/$MODEL_NAME"

mkdir -p $MODEL_PATH

export TRAIN_OPTIM="adafactor"
#TRAIN_OPTIM="paged_adamw_32bit"

export NUM_EPOCH=5
export WANDB_PROJECT=''
export WANDB_RUN_NAME=$(basename "$MODEL_PATH")_$(basename "$TRAIN_DATA")


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --main_process_port 8058 train.py \
    --deepspeed "ds_config/deepspeed_config_s3.json" \
    --train_only_last_turn True \
    --train_on_source False \
    --full_finetune True \
    --flash_attn True \
    --model_name_or_path ${MODEL_NAME} \
    --optim $TRAIN_OPTIM \
    --output_dir "${MODEL_PATH}-$(basename "$TRAIN_DATA")" \
    --num_train_epochs ${NUM_EPOCH} \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --data_seed 123 \
    --save_total_limit 10 \
    --evaluation_strategy "epoch" \
    --eval_steps 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --max_new_tokens 2048 \
    --dataloader_num_workers 32 \
    --group_by_length False \
    --logging_strategy steps \
    --remove_unused_columns False \
    --report_to wandb \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 16 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing true \
    --train_dataset ${TRAIN_DATA} \
    --eval_dataset ${EVAL_DATA} \
    --source_max_len 6144 \
    --target_max_len 2048 \
    --learning_rate 1e-5 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.5 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --use_auth_token true \
    --seed 0 2>&1|tee training.log
