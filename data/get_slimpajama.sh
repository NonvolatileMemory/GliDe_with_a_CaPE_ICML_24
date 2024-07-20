#!/bin/bash

# Initialize the variable i
for ((i=1; i<=1; i++))
do
    # Construct the URL with the current value of i
    url="https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk1/example_train_${i}.jsonl.zst"
    
    # Use wget to download the file
    wget -P dataset/slimpajama_sample "$url"
done
