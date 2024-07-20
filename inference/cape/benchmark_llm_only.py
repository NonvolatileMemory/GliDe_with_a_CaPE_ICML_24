import argparse
import random
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    AutoConfig,
    set_seed
)

import transformers
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import datasets
import json
import tqdm
from transformers import LlamaConfig, LlamaTokenizer, AutoTokenizer
from torch.utils.data.dataloader import DataLoader
import contexttimer
import sys
sys.path.append("../..")
from glide.jointModel import JointModel
from glide.glideModel import initialize_past_key_values, get_json_list
from chat_io import RichChatIO
from conversation import default_conversation, plain
import contexttimer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--large_path",
        type=str,
        default=None,
        required=True,
        help="large path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        choices=["code", "finance", "gsm", "spider", "mtbench"],
        help="dataset",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        required=True,
        help="cuda device",
    )
    args = parser.parse_args()
    set_seed(1024)

    dataset_map = {
        "code": "code_search_net_test.json",
        "finance": "gbharti_finance-alpaca_eval.json",
        "gsm": "gsm8k_test.json",
        "spider": "spider_validation.json",
        "mtbench": "mt_bench_test.jsonl",
    }
    
    print(f"load model {args.large_path.split('/')[-1]}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.large_path, padding_size="left")

    model = AutoModelForCausalLM.from_pretrained(args.large_path, torch_dtype=torch.float16)
    model_name = args.large_path.split('/')[-1]
    
    infer_type = "rebuttal-llm-only"
    os.makedirs(f"benchmark/{infer_type}/{model_name}", exist_ok=True)

    tokenizer.pad_token = tokenizer.eos_token

    model.cuda(args.cuda)



    dataset = dataset_map[args.dataset]
    # for dataset in dataset_list:
    prompt_dataset = []
    if "mt" in dataset:
        dataset_data = get_json_list(os.path.join("raw_data", dataset))
        # prompt_dataset = []
        
        for d in dataset_data:
            prompt = f"A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user input. USER: {d['turns'][0]} ASSISTANT:"
            prompt_dataset.append(prompt)
    else:
        dataset_data = json.load(open(os.path.join("raw_data", dataset), "r", encoding="utf-8"))
        # prompt_dataset = []
        
        for d in dataset_data:
            prompt = f"A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user input. USER: {d['conversation'][0]['content']} ASSISTANT:"
            prompt_dataset.append(prompt)
                
    answers = []
    new_token_sum = 0
    generation_time_sum = 0
    metrics = []
    
    bar = tqdm.tqdm(
        range(len(dataset_data)),
        desc=f"{dataset} inference steps"
    )
    for prompt in prompt_dataset:
        tokenized = tokenizer(prompt, return_tensors="pt").to(model.device)
                
        generation_time = 0
        with contexttimer.Timer() as gen_t:
            outputs = model.generate(
                tokenized.input_ids,
                use_cache=True,
                max_new_tokens=128,
                # past_key_values=past_key_values,
                do_sample=False
            )
            
        generation_time += gen_t.elapsed
        generation_time_sum += generation_time
                
        answers.append({
            "prompt": prompt,
            "output": tokenizer.batch_decode(outputs[:,tokenized.input_ids.size(1):], skip_special_tokens=True)[0]
        })
        
        new_token_sum+=outputs.size(1) - tokenized.input_ids.size(1)
        
        bar.update()
        
    
    metrics = {
        "name": dataset,
        "size": len(dataset_data),
        "new_token_sum": new_token_sum,
        "time": generation_time_sum,
        "speed": new_token_sum/generation_time_sum,
    }
    
    with open(f"benchmark/{infer_type}/{model_name}/metrics_{dataset}.jsonl", mode='w', encoding='utf-8') as output:
        json.dump(metrics, output, indent=4, ensure_ascii=False)
    
    with open(f"benchmark/{infer_type}/{model_name}/inference_results_{dataset}.jsonl", mode='w', encoding='utf-8') as output:
        for info in answers:
            output.write(json.dumps(info, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()