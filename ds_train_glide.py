#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import wandb
import json
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator
from flash_attention_path import replace_with_flash_attention
import os
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import causal_lm_model_to_fp32_loss, create_hf_model
from dschat.utils.perf import print_throughput
from glide.jointModel import JointModel
from glide.glideModel import smallLlamaForCausalLM, smallLlamaConfig
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import datasets
from utils import load_tokenized_dataset, setup_distributed_dataloader
import torch.distributed as dist

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor

def evaluation(model, eval_dataloader):
    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses += loss.float()
    losses = losses / (step + 1)
    try:
        losses = get_all_reduce_mean(losses)
    except:
        pass
    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")
    return perplexity, losses.item()

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--dataset',
                        type=str,
                        default='/home/lcdcx/share/datasets/yahma/alpaca-cleaned_train',
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path:dataset2-path ...')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--small_model_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--large_model_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=64,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=None,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--save_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1998,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="large_model.decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        default=True,
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        default=False,
                        help='Prints loss at each step.')
    # customized parser
    parser.add_argument("--large_lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--large_kv", action="store_true", default=False)
    parser.add_argument("--tarin_large", action="store_true", default=False)
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb_name")
    parser.add_argument("--reset_small", action="store_true", default=False)
    parser.add_argument("--wandb_proj", type=str, default="130m-7b-train-1", help="wandb_proj")
    parser.add_argument(
        "--use_remap",
        action="store_true",
        default=False,
        help="use remap layer in cross attention, large workload",
    )
    parser.add_argument(
        "--train_large",
        action="store_true",
        default=False,
        help="train large or not",
    )
    parser.add_argument(
        "--keep_same",
        action="store_true",
        default=False,
        help="keep the small cfg same as large",
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.save_dir is not None:
        with open(args.save_dir + '/train_cfg', "w") as f:
            json.dump(args.__dict__, f, indent=4)
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=False,
                                    tb_path=None,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family

    tokenizer = LlamaTokenizer.from_pretrained(args.small_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    if args.lora_dim > 0:
        args.train_large = True

    if args.large_kv:
        large_model = LlamaForCausalLM.from_pretrained(args.large_model_path)
        print_rank_0("load large model successful...")
        if not args.train_large:
            large_model = large_model.eval()
    else:
        large_model = None

    if args.reset_small:
        if args.large_kv:
            small_cfg = smallLlamaConfig.from_pretrained(args.small_model_path)
            small_cfg.large_hidden_size = large_model.config.hidden_size
            small_cfg.large_num_attention_heads = large_model.config.num_attention_heads
            small_cfg.large_head_dim =  large_model.config.hidden_size // large_model.config.num_attention_heads
            small_cfg.use_remap = args.use_remap
            
            if args.keep_same:
                small_cfg.hidden_size = large_model.config.hidden_size
                small_cfg.num_attention_heads = large_model.config.num_attention_heads
                small_cfg.intermediate_size = large_model.config.intermediate_size
                small_cfg.num_key_value_heads = small_cfg.num_attention_heads
        small_model = smallLlamaForCausalLM(small_cfg)
    else:
        small_model = smallLlamaForCausalLM.from_pretrained(args.small_model_path)

    model = JointModel(None, large_model, small_model, args.train_large)

    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        causal_lm_model_to_fp32_loss(model)

    if args.lora_dim > 0:
        model.large_model = convert_linear_layer_to_lora(model.large_model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model.large_model = only_optimize_lora_parameters(model.large_model)
            # model = make_model_gradient_checkpointing_compatible(model)
        
        # for name, param in model.small_model.named_parameters():aa
        #     param.requires_grad = True

    # Prepare the data
    dataset_path = [os.path.join(args.dataset, d) for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d)) and d.startswith("part")]
    train_dataset = load_tokenized_dataset(dataset_path) #load_tokenized_dataset(args.dataset.split(':'))
    #train_dataset = train_dataset.filter(lambda x: x is not None).filter(lambda x: len(x['input_ids']) < 2048)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        # eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        # eval_sampler = DistributedSampler(eval_dataset)
    
    def merge_collate_fn(batch):

        # Unzip the batch to separate input_ids, attention_mask, and labels
        input_ids, labels = zip(*[(b['input_ids'], b['labels']) for b in batch])
        ignore_idex = -100
        max_len = 2048
        # Pad the sequences. This requires the sequences to be converted into lists of tensors.
        padded_input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in input_ids], batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_attention_masks = padded_input_ids.ne(tokenizer.pad_token_id)
        padded_labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lab) for lab in labels], batch_first=True, padding_value=ignore_idex)
        return {'input_ids': padded_input_ids, 'attention_mask': padded_attention_masks, 'labels': padded_labels}
    
    train_dataloader = setup_distributed_dataloader(
        dataset=train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=merge_collate_fn,
    )
    # train_dataloader = DataLoader(train_dataset,
    #                              collate_fn=merge_collate_fn,
    #                              sampler=train_sampler,
    #                              batch_size=args.per_device_eval_batch_size)
    # eval_dataloader = DataLoader(eval_dataset,
    #                              collate_fn=merge_collate_fn,
    #                              sampler=eval_sampler,
    #                              batch_size=args.per_device_eval_batch_size)


    if model.large_model is not None:
        replace_with_flash_attention(model=model.large_model)
    replace_with_flash_attention(model=model.small_model)

    small_params = model.small_model.parameters()

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam

    if args.train_large:
        # large_params = model.large_model.parameters()
        # optimizer_grouped_parameters = [
        #     {'params': small_params, 'lr': args.lr, 'betas': (0.9, 0.95), 'weight_decay': args.weight_decay, 'adamw_mode': True},
        #     {'params': large_params, 'lr': args.large_lr, 'betas': (0.9, 0.95), 'weight_decay': args.weight_decay, 'adamw_mode': True}
        # ]
        # optimizer = AdamOptimizer(
        #     optimizer_grouped_parameters
        # )
        optimizer = AdamOptimizer(
                    [{'params': model.parameters(), 'lr': args.lr, 'betas': (0.9, 0.95), 'weight_decay': args.weight_decay, 'adamw_mode': True}])
    else:
        optimizer = AdamOptimizer(
            [{'params': small_params, 'lr': args.lr, 'betas': (0.9, 0.95), 'weight_decay': args.weight_decay, 'adamw_mode': True}])


    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.accumulation_steps)
    
    if args.num_warmup_steps is None:
        args.num_warmup_steps = int(args.num_train_epochs * 0.025 * (len(train_dataloader) // args.accumulation_steps))
        print_rank_0(f"Warmup steps is set to {args.num_warmup_steps}")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    args.output_dir = args.save_dir
    print_rank_0("***** Running training *****", args.global_rank)
    if torch.distributed.get_rank() == 0:
        if args.use_wandb:
            wandb.login(key='f8dcb864761d01f96d010f20990179a6f473fd70')
            if args.wandb_proj is None:
                wandb.init(project='default_proj')
            else:
                wandb.init(project=args.wandb_proj, name=args.wandb_name)
            args_dict = vars(args)
            wandb.config.update(args_dict)

    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity, eval_loss = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", args.global_rank)

    total_loss = torch.tensor(0.0).to(torch.cuda.current_device())
    large_total_loss = torch.tensor(0.0).to(torch.cuda.current_device())

    num_steps_per_epoch = len(train_dataloader)
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0


    save_tag = 'last'
    buffer_save_tag = 'last_2'
    use_buffer = 0

    if os.path.exists(os.path.join(args.save_dir, save_tag)) or os.path.exists(os.path.join(args.save_dir, buffer_save_tag)):
        try:
            print_rank_0(f"to continuie pretrain, loading ckpt from {save_tag}")
            _, client_sd = model.load_checkpoint(os.path.join(args.save_dir), save_tag)
            start_epoch, start_step, sampler_start_idx = client_sd['epoch'], client_sd['step'], client_sd['start_index']
        except RuntimeError as e:
            print_rank_0(f"load unsucessfully, change to continuie pretrain, loading ckpt from {buffer_save_tag}")
            _, client_sd = model.load_checkpoint(os.path.join(args.save_dir), buffer_save_tag)
            start_epoch, start_step, sampler_start_idx = client_sd['epoch'], client_sd['step'], client_sd['start_index']

    train_dataloader.sampler.set_start_index(start_index=sampler_start_idx)
    print_rank_0(f"Loaded sample at index {sampler_start_idx}")



    for epoch in range(start_epoch, args.num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch=epoch)
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        import time
        pbar = tqdm(desc=f"Epoch {epoch}", disable=not torch.distributed.get_rank() == 0, initial=start_step, total=num_steps_per_epoch)

        for step, batch in enumerate(train_dataloader, start=start_step):
            
            global_step = (epoch * num_steps_per_epoch) + (step + 1)

            start = time.time()
            batch = to_device(batch, device)
            batch_output, large_output = model(batch)

            loss = batch_output.loss
            total_loss += loss.detach().item()

            if args.large_kv:
                large_loss = large_output.loss
            else:
                large_loss = torch.tensor([0.])
            
            if args.train_large:
                large_logits = (large_output.logits).view(-1, large_output.logits.size(-1))
                large_loss = (large_logits.detach().softmax(dim=-1)) * ( large_logits.log_softmax(dim=-1) )
                large_loss = -large_loss.sum(dim=-1).mean()
            large_total_loss += large_loss.detach().item()

            if args.train_large:
                whole_loss = large_loss + loss
            else:
                whole_loss = loss

            model.backward(whole_loss)
            model.step()
            end = time.time()

            all_reduce_mean(tensor=total_loss)
            all_reduce_mean(tensor=large_total_loss)
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
    
            if args.use_wandb:
                if torch.distributed.get_rank() == 0:
                    wandb.log({"loss": total_loss.item(), "large loss": large_total_loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

            if args.print_loss and step % 32 == 1:
                if torch.distributed.get_rank() == 0:
                    print(
                        f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, lr = {lr_scheduler.get_last_lr()}, loss = {total_loss}, large loss = {large_total_loss}"
                    )

            if (args.save_interval > 0 and (step + 1) % (args.save_interval) == 0):
                print_rank_0('saving the internal model ...', args.global_rank)
                model = convert_lora_to_linear_layer(model)
                client_state = {'epoch': epoch, 'step': step+1, 'start_index': (step + 1) * args.per_device_train_batch_size}
                if use_buffer % 2 == 1: 
                    model.save_checkpoint(args.save_dir, tag=save_tag, client_state=client_state)
                else:
                    model.save_checkpoint(args.save_dir, tag=buffer_save_tag, client_state=client_state)
                use_buffer += 1
                print_rank_0('save successful')
                if args.global_rank == 0:
                    save_hf_format(model, tokenizer, args)
                    print_rank_0('save huggingface model successful')
            # if args.zero_stage == 3:
            #     # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            #     save_zero_three_model(model,
            #                           args.global_rank,
            #                           args.save_dir,
            #                           zero_stage=args.zero_stage)

            total_loss.fill_(0.0)
            large_total_loss.fill_(0.0)
            pbar.update()

        # Evaluate perplexity on the validation set.
        # print_rank_0(
        #     f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
        #     args.global_rank)
        # perplexity, eval_loss = evaluation(model, eval_dataloader)
        # print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", args.global_rank)
        model.tput_timer.update_epoch_count()
        sampler_start_idx = 0

    if args.save_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            args.output_dir = args.save_dir
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.save_dir,
                                  zero_stage=args.zero_stage)

if __name__ == "__main__":
    main()
