large_paths=("/home/ducunxiao/model/vicuna-7b-v1.5" "/home/ducunxiao/model/vicuna-13b-v1.5" "/home/ducunxiao/model/vicuna-33b-v1.3")
small_paths=("/home/ducunxiao/model/llama-68m " "/home/ducunxiao/model/llama-68m " "/home/ducunxiao/model/llama-68m ")
datasets=("mtbench" "code" "finance" "gsm" "spider")

for dataset in "${datasets[@]}"; do
    for index in {0..2}; do
        echo "large path ${large_paths[$index]}"
        echo "small path ${small_paths[$index]}"
        echo "dataset $dataset"
        python benchmark_other_model_cape.py \
            --large_path ${large_paths[$index]} \
            --small_path ${small_paths[$index]} \
            --dataset $dataset \
            --cuda 1 \
            --bin 8 8 8 6 6 6 4 4 2 2 2 2
    done
done