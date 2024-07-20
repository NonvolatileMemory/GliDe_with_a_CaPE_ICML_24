#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare dataset for continual pre-training
"""

import argparse
import json
import math
import os
import time
from multiprocessing import cpu_count

from datasets import dataset_dict, load_dataset
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from spliced_and_tokenized_dataset import (
    supervised_tokenize_pretrain,
    supervised_tokenize_pretrain_batched,
    ClosedToConstantLengthSplicedDataset,
)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input_dirs",
        type=str,
        required=True,
        default=None,
        help="Comma(i.e., ',') separated list of all data directories containing `.jsonl` data files.",
    )
    parser.add_argument(
        "--split_long_data",
        action="store_true",
        default=False,
        help="Split long data into serveral chunks",
    )
    parser.add_argument(
        "--tokenizer_dir", type=str, required=True, default=None, help="A directory containing the tokenizer"
    )
    parser.add_argument("--data_cache_dir", type=str, default="cache", help="Data cache directory")
    parser.add_argument(
        "--data_jsonl_output_dir",
        type=str,
        default="jsonl_output",
        help="Output directory of spliced dataset with jsonl format",
    )
    parser.add_argument(
        "--data_arrow_output_dir",
        type=str,
        default="arrow_output",
        help="Output directory of spliced dataset with arrow format",
    )
    parser.add_argument("--max_length", type=int, default=4096, help="Max length of each spliced tokenized sequence")
    parser.add_argument("--num_spliced_dataset_bins", type=int, default=10, help="Number of spliced dataset bins")
    parser.add_argument("--cpus", type=int, default=16, help="Number of spliced dataset bins")
    parser.add_argument(
        "--concat",
        action="store_true",
        default=False,
        help="Concat dataset",
    )
    args = parser.parse_args()

    if args.num_spliced_dataset_bins >= 100000:
        raise ValueError("Too many spliced divisions, must be smaller than 100000")
    
    cpus = min(args.cpus, cpu_count())

    assert not os.path.exists(args.data_cache_dir), f"Find existed data cache dir {args.data_cache_dir}"
    assert not os.path.exists(
        args.data_jsonl_output_dir
    ), f"Find existed jsonl data output dir {args.data_jsonl_output_dir}"
    assert not os.path.exists(
        args.data_arrow_output_dir
    ), f"Find existed arrow data output dir {args.data_arrow_output_dir}"
    os.makedirs(args.data_jsonl_output_dir)
    os.makedirs(args.data_arrow_output_dir)

    # Prepare to all input datasets
    input_data_paths = []
    input_data_dirs = args.data_input_dirs.split(",")
    for ds_dir in input_data_dirs:
        ds_dir = os.path.abspath(ds_dir)
        assert os.path.exists(ds_dir), f"Not find data dir {ds_dir}"
        ds_files = [name for name in os.listdir(ds_dir) if name.endswith(".jsonl")]
        ds_paths = [os.path.join(ds_dir, name) for name in ds_files]
        input_data_paths.extend(ds_paths)
        
    input_data_paths = [path for path in input_data_paths]
    
    print(input_data_paths)

    # Prepare to data splitting.
    train_splits = []
    split_interval = math.ceil(100 / args.num_spliced_dataset_bins)
    for i in range(0, 100, split_interval):
        start = i
        end = i + split_interval
        if end > 100:
            end = 100
        train_splits.append(f"train[{start}%:{end}%]")

    # Prepare to the tokenizer.
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    list_dataset = load_dataset(
        path="json",
        data_files=input_data_paths,
        cache_dir=os.path.join(args.data_cache_dir, "raw"),
        keep_in_memory=False,
        split=train_splits,
        num_proc=cpus,
    )
    
    before_count = 0
    after_count = 0
    count_max_length = 0
    for index, dataset in enumerate(list_dataset):
        assert isinstance(dataset, dataset_dict.Dataset)
        
        before_count += len(dataset)
        print(f"Before tokenization, length of the dataset: {len(dataset)}")
        print(dataset)
        
        if args.split_long_data:
            dataset = dataset.map(
                function=supervised_tokenize_pretrain_batched,
                fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length, "split": args.split_long_data},
                keep_in_memory=False,
                num_proc=min(len(dataset), cpus),
                batched=True,
                batch_size=1,
                remove_columns=dataset.column_names
            )
        else:
            dataset = dataset.map(
                function=supervised_tokenize_pretrain,
                fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length, "split": args.split_long_data},
                keep_in_memory=False,
                num_proc=min(len(dataset), cpus),
                remove_columns=dataset.column_names if args.max_length == -1 else []
            )

        print(f"After tokenization, length of the dataset: {len(dataset)}")
        
        if args.concat:
            dataset = dataset.sort(column_names=("seq_length"), reverse=False, keep_in_memory=False)
            spliced_dataset = ClosedToConstantLengthSplicedDataset(
                dataset=dataset, tokenizer=tokenizer, max_length=args.max_length, error_strict=False
            )
        else:
            spliced_dataset = dataset
        after_count += sum(1 for item in spliced_dataset)
        count_max_length += sum(1 for item in spliced_dataset if item["seq_length"] >= args.max_length)
        
        # Save each jsonl spliced dataset.
        output_index = "0" * (5 - len(str(index))) + str(index)
        output_name = f"part-{output_index}"
        output_jsonl_path = os.path.join(args.data_jsonl_output_dir, output_name + ".jsonl")
        st = time.time()
        with open(file=output_jsonl_path, mode="w", encoding="utf-8") as fp_writer:
            spliced_count = 0
            for spliced_data_point in spliced_dataset:
                spliced_count += 1
                fp_writer.write(json.dumps(spliced_data_point, ensure_ascii=False) + "\n")

        # Save each arrow spliced dataset
        output_arrow_path = os.path.join(args.data_arrow_output_dir, output_name)
        spliced_dataset = load_dataset(
            path="json",
            data_files=[output_jsonl_path],
            cache_dir=os.path.join(args.data_cache_dir, "spliced_and_tokenized"),
            keep_in_memory=False,
            num_proc=cpus,
            split="train",
        )
        spliced_dataset.save_to_disk(dataset_path=output_arrow_path, num_proc=min(len(spliced_dataset), cpus))
        
    print(f"Before tokenization: {before_count}")
    print(f"After tokenization: {after_count}")
    print(f"The number of data points >= {args.max_length}: {count_max_length}")


if __name__ == "__main__":
    main()
