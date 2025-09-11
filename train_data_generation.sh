#!/bin/bash

MODELNAME='Llama-3.1-8B-Instruct-selection...'
volume='/path/to/your/model/directory'

PORT=8058

model=/data/$MODELNAME
echo $model
docker run -d --gpus '"device=0,1,2,3"' \
    --shm-size 64g \
    -p $PORT:80 \
    -v $volume:/data \
    --name cot-selection \
    ghcr.io/huggingface/text-generation-inference:2.3.1 \
    --model-id $model \
    --max-batch-prefill-tokens 10240 \
    --max-input-length 10240 \
    --max-total-tokens 20480 \
&& sleep 2m

prompt=selection_IRI
mkdir -p $volume/train_data/$MODELNAME-$prompt

python3 generate_data.py \
    --port $PORT \
    --input-path $volume/train_data/rank_zephyr_training_data-parsed.jsonl \
    --prompt $prompt \
    --query-key query \
    --ctx-key passages \
    --model-name $MODELNAME \
    --start-idx 0 \
    --for-train \
    --output-name rank_zephyr_training_data \
    --output-dir $volume/train_data/$MODELNAME-$prompt

docker stop cot-selection
docker rm $(docker ps --filter status=exited -q)

python3 data_formatting.py \
    --input-path $volume/train_data/$MODELNAME-$prompt/rank_zephyr_training_data-0-39912-chat_temp0_generation.jsonl \
    --model-name $MODELNAME \
    --output-path $volume/train_data/$MODELNAME-$prompt-rank_zephyr_training_data-parsed.jsonl
