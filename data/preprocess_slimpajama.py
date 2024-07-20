import os
import json
import tqdm
import random
from multiprocessing import cpu_count
from datasets import load_dataset

def format(x):
    x["source"] = ""
    x["target"] = x["text"]
    
    return x

base_save_dir = "dataset/slimpajama_sample/"
cache_dir = "dataset/cache/"
split = "train"
os.makedirs(base_save_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
dataset = load_dataset(path="dataset/slimpajama_sample", cache_dir=cache_dir, split=split, keep_in_memory=False, num_proc=cpu_count())
    
dataset = dataset.map(format, num_proc=cpu_count(), remove_columns=dataset.column_names)
dataset = dataset.shuffle(seed=42)

sample_num = 2000 * 1024
random.seed(42)
sample_num = min(sample_num, len(dataset))
dataset = dataset.select(random.sample(range(len(dataset)), sample_num))
dataset.cleanup_cache_files()
    
# dataset_split.save_to_disk(dataset_path=os.path.join(base_save_dir, split))
save_path = os.path.join(base_save_dir, split + f"_sample_{sample_num}.jsonl")
with open(save_path, mode='w', encoding='utf-8') as output:
    for info in tqdm.tqdm(dataset, desc="Processing slimredpajama sample", total=len(dataset)):
        output.write(json.dumps(info, ensure_ascii=False)+"\n")
