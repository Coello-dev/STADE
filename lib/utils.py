import torch
import torch.nn as nn
import psutil


class Catcher(nn.Module):
    def __init__(self, module, inps, cache):
        super().__init__()
        self.module = module
        self.inps = inps
        self.cache = cache

    def forward(self, inp, **kwargs):
        inp = inp.cpu()
        self.inps[self.cache["i"]] = inp.cpu()
        self.cache["i"] += 1
        # Move everything to cpu otherwise it will stay in cuda later on
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.cpu()
        if "attention_mask" in kwargs:
            self.cache["catcher_attention_mask"] = kwargs["attention_mask"]
        else:
            self.cache["catcher_attention_mask"] = None
        self.cache["catcher_position_ids"] = kwargs["position_ids"]
        raise ValueError
