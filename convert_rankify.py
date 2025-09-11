import json
import re
import numpy as np
import argparse
from tqdm import tqdm

def formatting(args):
    with open(args.response_data,'r') as f:
        data = [json.loads(line) for line in f]

    key=args.key
    with open(args.original_data,'r') as f:
        original_data = [json.loads(line) for line in f]

    hits = []
    selected_nums = []
    retrieved_hits = []
    k=10

    with open(args.output_path,'w') as f:
        for d, o in tqdm(zip(data, original_data)):
            assert d['question']==o['question']
            ranks = [int(rank)-1 for rank in re.findall(r'\[(\d+)\]',d[key].split('## Final')[-1]) if int(rank)<21][:k]
            nd={
                'question':o['question'],
                'answers':o['answers'],
                key: ranks+[],
                'original_contexts': o['contexts'],
            }
            if len(ranks) <k:
                for rank in range(20):
                    if rank not in ranks: ranks.append(rank)
                    if len(ranks)==k: break
            selected_nums.append(len(ranks))

            nd['contexts']=[o['contexts'][rank] for rank in ranks]
            nd[f'{key}_merged']=ranks+[]

            f.write(json.dumps(nd)+'\n')

            for rank in ranks:
                if o['contexts'][rank]['has_answer']:
                    hit=True
                    break
                else:
                    hit=False
            for psg in o['contexts'][:20]:
                if psg['has_answer']:
                    rhit=True
                    break
                else:
                    rhit=False
            
            hits.append(hit)
            retrieved_hits.append(rhit)
            
    print(np.mean(hits),len(hits), np.mean(selected_nums))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script with a file parameter.")
    parser.add_argument('--response-data', type=str, required=True, help='Response File Name')
    parser.add_argument('--original-data', type=str, required=True, help='Original data File Name')
    parser.add_argument('--output-path', type=str, required=True, help='Folder Path')
    parser.add_argument('--key', type=str, required=True, help='response key')
    args = parser.parse_args()
    formatting(args)
