model="Llama2-7b"
tokenizer_dir="meta-llama/Llama-2-7b"

# Slimredpajama sample
dataset="slimpajama_sample"

# Format raw data
python preprocess_slimpajama.py

# Get full tokenization data
# Input is text
echo "Start tokenizing Dataset: $dataset (no truncation)"
python prepare_pretrain_dataset.py \
    --data_input_dirs dataset/$dataset/ \
    --tokenizer_dir  $tokenizer_dir \
    --data_cache_dir dataset/$dataset/$model/no_truncation_cache \
    --data_jsonl_output_dir dataset/$dataset/$model/no_truncation_jsonl \
    --data_arrow_output_dir dataset/$dataset/$model/no_truncation_arrow \
    --max_length -1 \
    --num_spliced_dataset_bins 10 \
    --cpus 9999 # will choose min(cpu_count(), 9999)
# max_length: -1 means we don't truncate

# If you want to merge different samples, uncomment the following.
# Input is full tokenization data (List of Integers)
# max_length=2048
# echo "Start tokenizing Dataset: $dataset (max length = $max_length)"
# python prepare_pretrain_dataset.py \
#     --data_input_dirs dataset/$dataset/$model/no_truncation_jsonl \
#     --tokenizer_dir  $tokenizer_dir \
#     --data_cache_dir dataset/$dataset/$model/$max_length/cache \
#     --data_jsonl_output_dir dataset/$dataset/$model/$max_length/jsonl \
#     --data_arrow_output_dir dataset/$dataset/$model/$max_length/arrow \
#     --max_length $max_length \
#     --num_spliced_dataset_bins 10 \
#     --concat \
#     --split_long_data \
#     --cpus 9999 # will choose min(cpu_count(), 9999)
# concat:  will merge sample 1 , sample 2 ... to max length
# split long data, do not drop any data
