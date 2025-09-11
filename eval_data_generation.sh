#!/bin/bash

MODELNAME='Llama-3.1-8B-Instruct-selection...'
volume='/path/to/your/model/directory'

PORT=8058
for CKT in 284 350
do
    model=/data/$MODELNAME/checkpoint-$CKT
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

    mkdir -p $volume/results/$MODELNAME-ckt-$CKT

    for dataname in 'hotpotqa-dev_bge_results.jsonl' 'multihoprag_bge_results.jsonl' '2wikimultihopqa-dev_bge_results.jsonl' 'musique-dev_bge_results.jsonl'
    do
        echo $dataname
        python3 generate_data.py \
            --port $PORT \
            --input-path $volume/benchmark_with_ir_results/$dataname \
            --prompt selection_IRI \
            --query-key query \
            --ctx-key retrieval_list \
            --model-name selection_IRI-ckt-$CKT \
            --start-idx 0 \
            --output-name $MODELNAME-ckt-$CKT-$dataname \
            --output-dir $volume/results/$MODELNAME-ckt-$CKT
    done
    docker stop cot-selection
    docker rm $(docker ps --filter status=exited -q)
    sleep 1m
done