
BASEDIR='/path/to/your/directory'
MKEY='your_model_key_example'

echo $MKEY

dataname='2wikimultihopqa-bge' # or other.
echo $dataname

ORIGIN_DATA_PATH='/benchmark_with_ir_results/2wikimultihopqa-dev_bge_results.jsonl' # modify
python3 convert_rankify.py \
    --response-data /path/to/your/model/response.jsonl \
    --original-data $ORIGIN_DATA_PATH \
    --key $MKEY \
    --output-path /path/you/want/to/save/name-$dataname.jsonl
