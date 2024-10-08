#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Splicing multiple pre-tokenized sequence data points
"""

import bisect
import random
import warnings
import math
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

from datasets import dataset_dict
from torch.utils.data import ConcatDataset, Dataset, IterableDataset
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


IGNORE_INDEX = -100

DSType = Union[Dataset, ConcatDataset, dataset_dict.Dataset]


def supervised_tokenize_pretrain(
    data_point: Dict[str, str], tokenizer: LlamaTokenizer, ignore_index: int = None, max_length: int = 4096, split: bool = False
) -> Dict[str, Union[int, str, List[int]]]:
    """
    A tokenization function to tokenize an original pretraining data point as following:
        {"source": "", "target": "Beijing, the capital of the People's Republic of China, ...", "category": "geography"}
    """
    assert tokenizer.add_bos_token is False and tokenizer.add_eos_token is False, (
        "Initially set `tokenizer.add_bos_token` and `tokenizer.add_eos_token` to False, "
        "add <bos> and <eos> manually later"
    )
    if ignore_index is None:
        ignore_index = IGNORE_INDEX

    # Process raw text data
    if "input_ids" not in data_point:
        source_text = data_point["source"]  # `str`
        target_text = data_point["target"]  # `str`
        is_null_source = len(source_text) == 0

        source_text = tokenizer.bos_token + source_text
        target_text += tokenizer.eos_token
        sequence_text = source_text + target_text

        tokenized = tokenizer([sequence_text])["input_ids"]
        sequence_input_ids = tokenized[0]
        sequence_labels = deepcopy(sequence_input_ids)

    else:
        sequence_input_ids = data_point["input_ids"]
        sequence_labels = data_point["labels"]

    chunks_input_ids = None
    chunks_labels = None
    chunks_seq_lens = None


    length = len(sequence_input_ids)
    if "input_ids" not in data_point and max_length == -1:
        return dict(
            input_ids=sequence_input_ids,
            labels=sequence_labels,
            seq_length=length
        )

    # Sequence truncation
    if max_length > 0:
        if split:
            chunks_input_ids = []
            chunks_labels = []
            chunks_seq_lens = []
            for start in range(0, length, max_length):
                end = start + max_length
                if end > length:
                    end = length

                chunk_input_ids = sequence_input_ids[start: end]
                chunk_labels = sequence_labels[start: end]

                chunks_input_ids.append(chunk_input_ids)
                chunks_labels.append(chunk_labels)
                chunks_seq_lens.append(end-start)
            assert math.ceil(length/max_length) == len(chunks_input_ids)
        else:
            sequence_input_ids = sequence_input_ids[:max_length]
            sequence_labels = sequence_labels[:max_length]

    if chunks_input_ids is not None:
        return dict(
            input_ids=chunks_input_ids,
            labels=chunks_labels,
            seq_length=chunks_seq_lens
        )
    else:
        return dict(
            input_ids=sequence_input_ids,
            labels=sequence_labels,
            seq_length=len(sequence_input_ids)
        )

def supervised_tokenize_pretrain_batched(
    data_point: Dict[str, List], tokenizer: LlamaTokenizer, ignore_index: int = None, max_length: int = 4096, split: bool = False
) -> Dict[str, Union[int, str, List[int]]]:
    
    sequence_input_ids = data_point["input_ids"]
    sequence_labels = data_point["labels"]
    
    assert len(sequence_input_ids) == len(sequence_labels) == 1
    
    data_point = dict(
        input_ids=sequence_input_ids[0],
        labels=sequence_labels[0]
    )
    
    return supervised_tokenize_pretrain(data_point, tokenizer, ignore_index, max_length, split)
    

class ClosedToConstantLengthSplicedDataset(IterableDataset):
    """
    Define an iterable dataset that returns a (close to) constant length data point spliced from multiple
    original independent (pre-tokenized) data points.
    """

    def __init__(
        self,
        dataset: DSType,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        num_packed_sequences: int = 8,
        fetch_sequence_func: Callable[[Any], Tuple[List[int], List[int]]] = None,
        input_ids_field: str = "input_ids",
        labels_field: str = "labels",
        infinite: bool = False,
        shuffle: bool = True,
        error_strict: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        self.infinite = infinite
        self.max_buffer_size = max_length * num_packed_sequences  # e.g., 4096 * 16
        self.shuffle = shuffle

        # Callable[[Dict[str, Any]], Tuple[List[int], List[int]]],
        # A function that fetch sequence input_ids and labels from the original data point
        if fetch_sequence_func is None:
            self.fetch_sequence_func = lambda data_point: (data_point[input_ids_field], data_point[labels_field])
        else:
            self.fetch_sequence_func = fetch_sequence_func
        self.input_ids_field = input_ids_field
        self.labels_field = labels_field

        self.error_strict = error_strict
        self.current_size = 0  # `int`, current packed data size.

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterable[Dict[str, List[int]]]:
        iterator = iter(self.dataset)
        more_data_points = True
        while more_data_points is True:
            buffer, buffer_len = [], 0
            while True:
                # ending condition.
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    # `Tuple[List[int], List[int]]`
                    seq_input_ids, seq_labels = self.fetch_sequence_func(next(iterator))
                    buffer.append({self.input_ids_field: seq_input_ids, self.labels_field: seq_labels})
                    buffer_len += len(buffer[-1][self.input_ids_field])
                except StopIteration:
                    if self.infinite is True:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_data_points = False
                        break
            examples = []  # `List[Dict[str, List[int]]]`, save buffered spliced data points.
            spliced_input_ids, spliced_labels = [], []  # `List[int]`, `List[int]`
            for i, data_point in enumerate(buffer):
                # TODO(2023-09-18) check errors for each unspliced tokenized data point
                seq_input_ids = data_point[self.input_ids_field]
                seq_labels = data_point[self.labels_field]
                # Handle special case:
                # If the length of an original data point (i.e., input_ids length of a data point before splicing)
                # exceeds `max_length`, truncate it.
                if len(seq_input_ids) > self.max_length:
                    truncated_seq_input_ids = seq_input_ids[: self.max_length]
                    truncated_label_ids = seq_labels[: self.max_length]
                    if set(truncated_label_ids) == {IGNORE_INDEX}:
                        if self.error_strict is True:
                            raise ValueError(
                                f"Find an out-of-bounds length({len(seq_input_ids)}) data point "
                                f"with all label values as {IGNORE_INDEX}."
                            )
                        else:
                            warnings.warn(f"Filter an error truncated data point (labels all {IGNORE_INDEX})")
                            continue  # Skip the current error data point.
                    spliced_data_point = {
                        self.input_ids_field: truncated_seq_input_ids,
                        self.labels_field: truncated_label_ids,
                        "seq_length": len(truncated_seq_input_ids)
                    }
                    examples.append(spliced_data_point)
                    warnings.warn("Find a data point to be truncated.")
                    continue

                # Pre action judgment.
                if len(spliced_input_ids) + len(seq_input_ids) > self.max_length:
                    spliced_data_point = {
                        self.input_ids_field: spliced_input_ids,
                        self.labels_field: spliced_labels,
                        "seq_length": len(spliced_input_ids)
                    }  # `Dict[str, List[int]]`
                    # Update.
                    spliced_input_ids, spliced_labels = [], []
                    spliced_input_ids.extend(seq_input_ids)
                    spliced_labels.extend(seq_labels)
                    examples.append(spliced_data_point)
                else:
                    spliced_input_ids.extend(seq_input_ids)
                    spliced_labels.extend(seq_labels)
            # For residual spliced data point at the end of the data set
            if self.infinite is False and more_data_points is False and len(spliced_input_ids) > 0:
                examples.append({self.input_ids_field: spliced_input_ids, self.labels_field: spliced_labels, "seq_length": len(spliced_input_ids)})
            if self.shuffle:
                random.shuffle(examples)
            for spliced_data_point in examples:
                # TODO(2023-09-18): check errors for each spliced tokenized data point.
                self.current_size += 1
                yield spliced_data_point
