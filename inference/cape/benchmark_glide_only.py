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
from glide.glideModel import initialize_past_key_values, get_json_list, smallLlamaForCausalLM
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
    parser.add_argument('--bin', nargs='+', type=int, required=True, default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], help="bin")
    
    args = parser.parse_args()
     
    current_datetime = datetime.now()
    time_tag = current_datetime.strftime("%H%M%S")

    print(time_tag)
    set_seed(1024)

    dataset_map = {
        "code": "code_search_net_test.json",
        "finance": "gbharti_finance-alpaca_eval.json",
        "gsm": "gsm8k_test.json",
        "spider": "spider_validation.json",
        "mtbench": "mt_bench_test.jsonl",
    }
    
    print(f"load model {args.small_path.split('/')[-1]} and {args.large_path.split('/')[-1]}")
    # large_path = "/home/ducunxiao/model/vicuna-7b-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(args.large_path, padding_size="left")

    # model_path = "/home/ducunxiao/kv-speculative/checkpoint/slimpajama-6b-vicuna-7b-sharegpt-sft-epoch-1/"
    # speculative_model = JointModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # model_name = model_path.split('/')[-1]
    # small_path = "/home/ducunxiao/model/glide-47m-vicuna-7b"
    model_name = args.small_path.split("/")[-1]
    large_model = LlamaForCausalLM.from_pretrained(args.large_path, torch_dtype=torch.float16)
    ass_model = smallLlamaForCausalLM.from_pretrained(args.small_path, torch_dtype=torch.float16)

    infer_type = "rebuttal-glide-only/"
    num_beams = 1 if "beam" in infer_type else 1
    num_return_sequences = num_beams
    batch_size = 1

    os.makedirs(f"benchmark/{infer_type}/{model_name}", exist_ok=True)

    # large_model = speculative_model.large_model.half()
    # ass_model = speculative_model.small_model.half()

    # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    large_model.generation_config.pad_token_id = large_model.generation_config.eos_token_id
    ass_model.generation_config.pad_token_id = ass_model.generation_config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    large_model.cuda(args.cuda)
    ass_model.cuda(args.cuda)

    # dataset_list = os.listdir("raw_data")
    # dataset_list = [dataset for dataset in dataset_list if "test" in dataset or "eval" in dataset or "validation" in dataset]
    # dataset_list.sort()

    # print(dataset_list)

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
    proposal_num = 0
    accept_length_sum = 0
    alpha_sum = 0
    prefix_time = 0
    generation_time = 0
    steps = 0
    algo_time = 0
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
            steps_to_sample=None,
            prob_to_top=args.bin,
            expand_cape=False,
            batch_size=batch_size,
        )
        
        answers.append({
            "prompt": prompt,
            "output": tokenizer.batch_decode(outputs["input_ids"][:,tokenized.input_ids.size(1):], skip_special_tokens=True)[0]
        })
        
        new_token_sum+=outputs["new_token_sum"].item()
        accept_length_sum+=outputs["accept_length_sum"].item()
        proposal_num += outputs["proposal_num"]
        if isinstance(outputs["alpha_sum"], int):
            alpha_sum+=outputs["alpha_sum"]
        else:
            alpha_sum+=outputs["alpha_sum"].item()
            
        prefix_time+=outputs["prefix_time"]
        generation_time+=outputs["generation_time"]
        steps+=outputs["steps"]
        algo_time+=outputs["algo_time"]
        
        bar.update()
        
    
    metrics = {
        "name": dataset,
        "size": len(dataset_data),
        "steps": steps,
        "new_token_sum": new_token_sum,
        "accept_length_sum": accept_length_sum,
        "proposal_num": proposal_num,
        "alpha_sum": alpha_sum,
        "prefix_time": prefix_time,
        "generation_time": generation_time,
        "algo_time": algo_time,
        "prob_to_conf": outputs["prob_to_top"],
        "speed": new_token_sum/ generation_time,
        "mean_accepted_tokens": accept_length_sum / steps,
        "mean_accepted_tokens_all": new_token_sum / steps
    }

    print(metrics)
    
    with open(f"benchmark/{infer_type}/{model_name}/metrics_{dataset+time_tag}_beam{num_beams}_batch{batch_size}.jsonl", mode='w', encoding='utf-8') as output:
        json.dump(metrics, output, indent=4, ensure_ascii=False)
    
    with open(f"benchmark/{infer_type}/{model_name}/inference_results_{dataset}.jsonl", mode='w', encoding='utf-8') as output:
        for info in answers:
            output.write(json.dumps(info, ensure_ascii=False)+"\n")
            
if __name__ == "__main__":
    main()
