from tqdm import tqdm
import time
import json
import re
import requests
from concurrent import futures
from functools import partial
import argparse
import random
random.seed(0)
num_concurrent=32
from openai import OpenAI




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name',type=str, default='example')
    parser.add_argument('--prompt',type=str, default='')
    parser.add_argument('--query-key',type=str, default='query')
    parser.add_argument('--ctx-key',type=str, default='context')
    parser.add_argument('--output-dir',type=str, default='./')
    parser.add_argument('--port',type=int, default=8080)
    parser.add_argument('--start-idx',type=int, default=0)
    parser.add_argument('--end-idx',type=int, default=-1)
    parser.add_argument('--model-name',type=str, default='')
    parser.add_argument('--input-path',type=str, default='')
    parser.add_argument('--for-train',action='store_true')
    
    args = parser.parse_args()
    return args


selection_sys_prompt = "You are RankLLM, an intelligent assistant that can rank and select passages based on their relevancy to the query."



selection_only_prompt="""I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}

Search Query: {question}.
Select the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [2] [1]. Only respond with the selection results, do not say any word or explain."""

woIRI_selection_CoT_prompt="""I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Select the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [2] [1].
Let's think step by step."""


selection_IRI_prompt = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Please follow the steps below:
Step 1. Please list up the information requirements to answer the query.
Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.
Step 3. Choose the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [4] [2]."""


rank_zephyr_sys_prompt="""You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."""
rank_zephyr_prompt="""I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {question}.

{context}

Search Query: {question}.

Rank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain."""

stepbystep_ranking_prompt = """I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Rank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be ### Final ranking: [] > [], e.g., ### Final ranking: [2] > [1]. 
Think step by step."""




sys_prompts = {
    'rankzephyr':rank_zephyr_sys_prompt,
    'rankzephyr_CoT':rank_zephyr_sys_prompt,
    'selection_IRI':selection_sys_prompt,
    'selection_woIRI':selection_sys_prompt, 
    'selection_only':selection_sys_prompt,
}

user_prompts = {
    'rankzephyr':rank_zephyr_prompt,
    'selection_IRI':selection_IRI_prompt,
    'rankzephyr_CoT':stepbystep_ranking_prompt,
    'selection_woIRI':woIRI_selection_CoT_prompt, #250212
    'selection_only':selection_only_prompt,
}




def evaluate(data, client, prompt, args):
    
    if args.ctx_key=='retrieval_list':
        ctxs = [re.sub(r"\n+"," ",f'[{idx+1}] {ctx["text"].strip()}') for idx, ctx in enumerate(data[args.ctx_key])]
        data[args.ctx_key] = ctxs
    if 'contexts' in args.ctx_key:
        ctx = data[args.ctx_key][0]
        if "title" in ctx.keys() and ctx["title"]!='':
            ctxs = [re.sub(r"\n+"," ",f'[{idx+1}] title: {ctx["title"]}\t{ctx["text"].strip()}') for idx, ctx in enumerate(data[args.ctx_key][:20])]
        else: 
            ctxs = [re.sub(r"\n+"," ",f'[{idx+1}] {ctx["text"].strip()}') for idx, ctx in enumerate(data[args.ctx_key][:20])]
        
        data[args.ctx_key] = ctxs
    if type(data[args.ctx_key]) is str:
        data[args.ctx_key] = [data[args.ctx_key]]
    
    user_input = user_prompts[prompt].format(question=data[args.query_key], context='\n\n\n'.join(data[args.ctx_key]), num=len(data[args.ctx_key][:20]))
    
    max_tokens = 4096
    if args.prompt =='rankzephyr': max_tokens = 512
    
    chat_completion = client.chat.completions.create(
        model='tgi',
        messages=[
            {'role':'system','content':sys_prompts[prompt]}, #rankzephyr
            {'role':'user', 'content': user_input}
        ],
        stream=False, 
        max_tokens=max_tokens,
    )
    output = chat_completion.choices[0].message
    if args.for_train:
        messages=[
            {'role':'system','content':sys_prompts[prompt]}, #rankzephyr
            {'role':'user', 'content': user_input},
        ]
        return output.content, messages
    return output.content

def main(args):
    url = f"http://127.0.0.1:{args.port}/v1"
    client = OpenAI(
        base_url=url,
        api_key='-'
    )

    with open(args.input_path,'r',encoding='utf-8') as f:
        if 'jsonl' in args.input_path:
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)

    if args.end_idx==-1:
        args.end_idx = len(data)

    with open(f'{args.output_dir}/{args.output_name}-{args.start_idx}-{args.end_idx}-chat_temp0_generation.jsonl','w') as wf:
        
        data = data[args.start_idx:args.end_idx]
        total = len(data)
        print(total, flush=True)
        
        response_data = []
        pass_rate = []
        _fn = partial(evaluate, client=client, prompt=args.prompt, args=args)
        num_concurrent = 64

        for idx in tqdm(range(0, total, num_concurrent)):
            batch = data[idx: idx+num_concurrent]
            with futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                outputs= list(executor.map(_fn, batch))
            
            for b, o in zip(batch, outputs):
                nd = {}

                if args.for_train:
                    o, messages = o
                    nd['messages']=messages
                for k,v in b.items():
                    nd[k]=v
                key = args.model_name
                nd[key] = ''
                try:
                    nd[key] = o
                except Exception as e:
                    print(e)
                
                response_data.append(nd)
                wf.write(json.dumps(nd,ensure_ascii=False)+'\n')
        

if __name__ =="__main__":
    args = parse_args()
    main(args)
