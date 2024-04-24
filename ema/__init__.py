from collections import OrderedDict
from copy import deepcopy

import torch

# inspired by https://www.zijianhu.com/post/pytorch/ema/


class EMA(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, decay: float):
        super().__init__()
        self.decay = decay
        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        assert self.training

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())
        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        assert model_params.keys() == shadow_params.keys()
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, param in model_params.items():
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def forward(self, inputs):
        if self.training:
            return self.model(inputs)
        else:
            return self.shadow(inputs)
