from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    AutoConfig,
    TextIteratorStreamer
)
from threading import Thread
import transformers
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import datasets
from torch.utils.data.dataloader import DataLoader
import contexttimer
import sys
sys.path.append("../..")
from glide.jointModel import JointModel, smallLlamaForCausalLM, smallLlamaConfig
from glide.glideModel import initialize_past_key_values, get_json_list
from glide.modeling_llama_kv import LlamaForCausalLM
from chat_io import RichChatIO
from conversation import default_conversation, plain

print("load model")

small_path = "/home/ducunxiao/model/glide-47m-vicuna-7b"
large_path = "/home/ducunxiao/model/vicuna-7b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(large_path, padding_size="left")

infer_type = "greedy"

large_kv = False
llm_only = True
if llm_only:
    large_model = AutoModelForCausalLM.from_pretrained(large_path, torch_dtype=torch.float16)
    ass_model = None
else:
    large_model = LlamaForCausalLM.from_pretrained(large_path, torch_dtype=torch.float16)
    ass_model = smallLlamaForCausalLM.from_pretrained(small_path, torch_dtype=torch.float16)
    ass_model.generation_config.pad_token_id = ass_model.generation_config.eos_token_id


# set pad_token_id to eos_token_id because GPT2 does not have a PAD token
large_model.generation_config.pad_token_id = large_model.generation_config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

large_model.cuda(0)
if ass_model is not None:
    ass_model.cuda(0)

chatio = RichChatIO()
conv = default_conversation

def normal_stream(streamer, conv):
    conv.messages[-1][-1] = ""
    for new_text in streamer:
        conv.messages[-1][-1] += new_text
        yield conv.messages[-1][-1]

while True:
    inp = chatio.prompt_for_input(conv.roles[0])

    if not inp:
        print('prompt should not be empty!')
        continue
        
    if inp.strip() == "clear":
        conv.clear()
        os.system("clear")
        continue
        
    if inp.strip() == "exit":
        print('End of chat.')
        break
    
    query_text = inp.strip()
                
    conv.append_message(conv.roles[0], query_text)
    conv.append_message(conv.roles[1], None)

    chatio.prompt_for_output(conv.roles[1])
    
    tokenized = tokenizer(conv.get_prompt(), return_tensors="pt").to(large_model.device)

    if llm_only:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        generation_kwargs = dict(input_ids=tokenized.input_ids, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=large_model.generate, kwargs=generation_kwargs)
        thread.start()

        outputs = chatio.stream_output_normal(normal_stream(streamer, conv))
        conv.messages[-1][-1]=outputs.strip()
    else:
        outputs = chatio.stream_output(
            ass_model.medusa_generate_stream(
                tokenized.input_ids,
                large_kv=large_kv,
                large_model=large_model,
                max_steps=128,
                num_beams=1,
                num_return_sequences=1,
                topk=None,
                temperature=1.0,
                do_beam_sample=True,
                do_speculative_sample=False,
                prob_to_top=[8, 8, 8, 6, 6, 6, 4, 4, 2, 2, 2, 2],
                expand_cape=False,
                tokenizer_path=large_path,
            )
        )
        
        conv.messages[-1][-1]=outputs.strip()
        
        large_model.past_key_values_data.fill_(0)
        large_model.current_length_data.fill_(0)
        ass_model.past_key_values_data.fill_(0)
        ass_model.current_length_data.fill_(0)
