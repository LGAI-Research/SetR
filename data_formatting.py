import json
import re
import argparse
import random
random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path',type=str, default='')
    parser.add_argument('--model-name',type=str, default='')
    parser.add_argument('--output-path',type=str, default='')
    
    args = parser.parse_args()
    return args




def response_parsing(response, rank=False):
    if rank:
        ranks = re.findall(r'\#\#\#\s?Final ranking([\w\W]+)',response, re.IGNORECASE)[0].strip()
        for line in ranks.split('\n'):
            if '>' in line: ranks=line;break
        ranks = re.sub(r'\:\s','',ranks)
        
        ranks = re.sub(r'[^0-9\[\]>= ]','',ranks).strip()
        if ranks[0] != '[':
            ranks = re.sub(r'[^\[]','',ranks,count=1).strip()
        if '>' not in ranks:
            raise Exception('no ranking')
        output = ranks
    else:
        output = re.findall(r'\#\#\#\s?Final Selection([\w\W]+)',response, re.IGNORECASE)[0].strip()
        output = ' '.join(['['+rank+']' for rank in re.findall(r'\d+',output)])
    return output 

def main(args):

    for_train = []
    with open(args.input_path,'r') as f:
        for line in f:
            data = json.loads(line)
            nd = {
            'id':data['id'],
            'messages': data['messages']
            }
            try:
                response = data[args.model_name]
                assert response != ''
                # if rank only:
                #   response = response_parsing(response, rank=True)
                # if selection only:
                #   response = response_parsing(response)
                nd['messages'].append({'role':'assistant','content':response})
                
                for_train.append(nd)
            except Exception as e:
                print(e, flush=True)
                continue
    random.shuffle(for_train)
    split = int(len(for_train)*0.9)
    with open(f"{args.output_path}.train",'w') as f:
        for d in for_train[:split]: f.write(json.dumps(d)+'\n')
    with open(f"{args.output_path}.eval",'w') as f:
        for d in for_train[split:]: f.write(json.dumps(d)+'\n')


if __name__ =="__main__":
    args = parse_args()
    main(args)