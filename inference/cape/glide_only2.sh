large_paths=("/home/ducunxiao/model/vicuna-7b-v1.5" "/home/ducunxiao/model/vicuna-13b-v1.5" "/home/ducunxiao/model/vicuna-33b-v1.3")
small_paths=("/home/ducunxiao/model/glide-47m-vicuna-7b" "/home/ducunxiao/model/glide-47m-vicuna13b" "/home/ducunxiao/model/glide-vicuna33b")
datasets=("code")

for dataset in "${datasets[@]}"; do
    for index in {0..2}; do
        echo "large path ${large_paths[$index]}"
        echo "small path ${small_paths[$index]}"
        echo "dataset $dataset"
        python benchmark_glide_only.py \
            --large_path ${large_paths[$index]} \
            --small_path ${small_paths[$index]} \
            --dataset $dataset \
            --cuda 2 \
            --bin 1 1 1 1 1 1 1 1 1 1 1 1
    done
done