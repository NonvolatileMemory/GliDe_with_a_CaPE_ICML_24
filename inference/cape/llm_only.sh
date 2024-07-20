large_paths=("/home/ducunxiao/model/vicuna-7b-v1.5" "/home/ducunxiao/model/vicuna-13b-v1.5" "/home/ducunxiao/model/vicuna-33b-v1.3")
datasets=("mtbench" "code" "finance" "gsm" "spider")

for dataset in "${datasets[@]}"; do
    for index in {0..2}; do
        echo "large path ${large_paths[$index]}"
        echo "dataset $dataset"
        python benchmark_llm_only.py \
            --large_path ${large_paths[$index]} \
            --dataset $dataset \
            --cuda 7 \
    done
done