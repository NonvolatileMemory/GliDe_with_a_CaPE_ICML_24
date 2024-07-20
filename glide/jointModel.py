import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn
from transformers import LlamaConfig, LlamaTokenizer
from glide.glideModel import smallLlamaForCausalLM, smallLlamaConfig

from typing import Any, Dict, Optional, Union

from glide.modeling_llama_kv import LlamaForCausalLM
class jointLlamaConfig(PretrainedConfig):

    model_type = "JointModel"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        small_cfg=None,
        large_cfg=None,
        pass_cfg=False,
        **kwargs,
    ):
        if pass_cfg:
            self.small_cfg = small_cfg
            self.large_cfg = large_cfg
        else:
            if small_cfg is None:
                small_cfg = {}
            self.small_cfg = smallLlamaConfig.from_dict(small_cfg)
            if large_cfg is not None:
                self.large_cfg = LlamaConfig.from_dict(large_cfg) 
            else:
                self.large_cfg = None

class JointModel(PreTrainedModel):
    config_class = jointLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["smallLlamaForCausalLM", "LlamaForCausalLM"]
    _skip_keys_device_placement = "past_key_values"
    def __init__(self, config, large_model=None, small_model=None, train_large=False):
        # super(JointModel, self).__init__(config)
        nn.Module.__init__(self)
        # two ways to initialize
        # directly init from config
        # init from given large_model and small_model
        if config is not None:
            self.config = config
            assert self.config.small_cfg is not None, "at least provide small cfg"
            self.small_model = smallLlamaForCausalLM(self.config.small_cfg)
            if self.config.large_cfg is not None:
                self.large_model = LlamaForCausalLM(self.config.large_cfg)
            else:
                self.large_model = None
        else:
            self.large_model = large_model
            self.small_model = small_model
            self.config = jointLlamaConfig(small_model.config, large_model.config if large_model is not None else None, pass_cfg=True)
        
        self.train_large = train_large

        # if self.large_model is not None:
        #     self.large_model.gradient_checkpointing_enable()
        # self.small_model.gradient_checkpointing_enable()
        self.max_len = 2048

    def forward(self, batch):
        # trunct
        batch['input_ids'] = batch['input_ids'][:, :self.max_len]
        batch['attention_mask'] = batch['attention_mask'][:, :self.max_len]
        batch['labels'] = batch['labels'][:, :self.max_len]

        try:
            if 'large_key_values' in batch.keys():
                batch['large_key_values'] = large_kv
            if self.large_model is not None:
                if self.train_large:
                    large_output = self.large_model(input_ids=batch['input_ids'],
                        attention_mask =batch['attention_mask'],
                        labels = batch['labels'],
                        use_cache=True)
                    large_kv = large_output.past_key_values
                else:
                    with torch.no_grad():
                        large_output = self.large_model(input_ids=batch['input_ids'],
                            attention_mask =batch['attention_mask'],
                            labels = batch['labels'],
                            use_cache=True)
                    large_kv = large_output.past_key_values
                    for layerkv in large_kv:
                        layerkv = (layerkv[0].detach(), layerkv[1].detach())
                batch['large_key_values'] = large_kv #(large_kv[-1][0], large_kv[-1][1])  # len = 32

            else:
                large_output = None
            
            batch_output = self.small_model(**batch)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM: input shape is {batch['input_ids'].size()}")
                torch.cuda.empty_cache()
            else:
                print(batch['input_ids'])

        if large_output is not None:
            if torch.isnan(large_output.loss):
                print(batch['input_ids'].size())
                print(batch['input_ids'])
                print(batch['labels'])
        return batch_output, large_output


    def gradient_checkpointing_enable(self):
        a = 1
