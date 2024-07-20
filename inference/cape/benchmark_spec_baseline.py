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
from glide.jointModel import JointModel, smallLlamaForCausalLM, smallLlamaConfig
from glide.glideModel import initialize_past_key_values, get_json_list
from glide.modeling_llama_kv import LlamaForCausalLM
from chat_io import RichChatIO
from conversation import default_conversation, plain
from datetime import datetime


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
        "--small_path",
        type=str,
        default=None,
        required=True,
        help="small path",
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
    
    dataset_map = {
        "code": "code_search_net_test.json",
        "finance": "gbharti_finance-alpaca_eval.json",
        "gsm": "gsm8k_test.json",
        "spider": "spider_validation.json",
        "mtbench": "mt_bench_test.jsonl",
    }
    
    set_seed(1024)
    current_datetime = datetime.now()
    time_tag = current_datetime.strftime("%H%M%S")

    print(time_tag)

    print(f"load model {args.small_path.split('/')[-1]} and {args.large_path.split('/')[-1]}")
    tokenizer = AutoTokenizer.from_pretrained(args.large_path, padding_size="left")

    model_name = args.small_path.split("/")[-1]

    infer_type = "rebuttal-spec_baseline"
    num_beams = 5 if "beam" in infer_type else 1
    num_return_sequences = num_beams

    os.makedirs(f"benchmark/{infer_type}/{model_name+'-'+args.large_path.split('/')[-1]}", exist_ok=True)

    large_model = LlamaForCausalLM.from_pretrained(args.large_path, torch_dtype=torch.float16)
    ass_model = smallLlamaForCausalLM.from_pretrained(args.small_path, torch_dtype=torch.float16)

    # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    large_model.generation_config.pad_token_id = large_model.generation_config.eos_token_id
    ass_model.generation_config.pad_token_id = ass_model.generation_config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    large_model.cuda(args.cuda)
    ass_model.cuda(args.cuda)

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
    accept_length_sum = 0
    alpha_sum = 0
    prefix_time = 0
    generation_time = 0
    steps = 0
    metrics = []
    
    bar = tqdm.tqdm(
        range(len(dataset_data)),
        desc=f"{dataset} inference steps"
    )
    for prompt in prompt_dataset:
        tokenized = tokenizer(prompt, return_tensors="pt").to(large_model.device)

        outputs = ass_model.medusa_generate(
            tokenized.input_ids,
            large_model=large_model,
            max_steps=128,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            topk=None,
            temperature=1.0,
            do_beam_sample=False,
            do_speculative_sample=False,
            large_kv=False,
            steps_to_sample=None
        )
    
        answers.append({
            "prompt": prompt,
            "output": tokenizer.batch_decode(outputs["input_ids"][:,tokenized.input_ids.size(1):], skip_special_tokens=True)[0]
        })

        new_token_sum+=outputs["new_token_sum"].item()
        accept_length_sum+=outputs["accept_length_sum"].item()
        
        if isinstance(outputs["alpha_sum"], int):
            alpha_sum+=outputs["alpha_sum"]
        else:
            alpha_sum+=outputs["alpha_sum"].item()
            
        prefix_time+=outputs["prefix_time"]
        generation_time+=outputs["generation_time"]
        steps+=outputs["steps"]
        
        bar.update()
        
    
    metrics = {
        "name": dataset,
        "size": len(dataset_data),
        "steps": steps,
        "new_token_sum": new_token_sum,
        "accept_length_sum": accept_length_sum,
        "alpha_sum": alpha_sum,
        "prefix_time": prefix_time,
        "generation_time": generation_time,
        "mean_accepted_tokens": accept_length_sum / steps,
        "mean_accepted_tokens_all": new_token_sum / steps
    }
    
    print(metrics)
    
    with open(f"benchmark/{infer_type}/{model_name+'-'+args.large_path.split('/')[-1]}/metrics_{dataset+time_tag}_beam{num_beams}.jsonl", mode='w', encoding='utf-8') as output:
        json.dump(metrics, output, indent=4, ensure_ascii=False)
    
    with open(f"benchmark/{infer_type}/{model_name+'-'+args.large_path.split('/')[-1]}/inference_results_{dataset}.jsonl", mode='w', encoding='utf-8') as output:
        for info in answers:
            output.write(json.dumps(info, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()