# basic library
import copy
import re
import os
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
from tqdm import tqdm

# deep learning library
import transformers
import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100
debug_msg_flag = False


def rank0_print(*args):
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(*args)

@dataclass
class DataCollatorForSingleTurnCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool = False
    predict_with_generate: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}"
                   if not example['input'].startswith(self.tokenizer.bos_token)
                   else example['input']
                   for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels

        return data_dict


def conv2single_preprocess(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    train_only_last_turn=False,
):
    global debug_msg_flag
    outputs = []
    if 'input' in source and 'output' in source:
        outputs = [{
            'input': source['input'],
            'output': source['output'],
        }]
    else:
        conversations = source['messages']
        for j in range(len(conversations)):
            
            if train_only_last_turn:
                if j != len(conversations) - 1:
                    continue
            if conversations[j]['role'] == "assistant":  
                try:

                    input_prompt = tokenizer.apply_chat_template(conversations[:j], tokenize=False, add_generation_prompt=True)
                    output_text = conversations[j]['content']
                    outputs.append({
                        'input': input_prompt,
                        'output': output_text
                    })
                except Exception as e:
                    try:
                        outputs.append({
                            'input': '[EOS]\n'.join([c['content'] for c in conversations[:j]]+['']),
                            'output': conversations[j]['content']+'[EOS]'
                        })
                    except:
                        if not debug_msg_flag:
                            import traceback
                            traceback.print_exc()
                            debug_msg_flag = True
                        pass
                    
    return {"data": outputs}

