import torch
import torch.nn as nn
from lib.prune_zoo.basepruner import BasePruner


# Define Wanda class
class Wanda(BasePruner):
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, structure="row", **kwargs):
        super().__init__(layer=layer, structure=structure)

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def process_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()

    def get_weight_metric(self, **kwargs):
        return torch.abs(self.layer.weight.data) * torch.sqrt(
            self.scaler_row.reshape((1, -1))
        )


# Define Wanda Mean class
class WandaMean(BasePruner):
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, structure="row", **kwargs):
        super().__init__(layer=layer, structure=structure)

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def get_statistics(self):
        return self.mean_inp / self.nsamples

    def get_weight_metric(self, **kwargs):
        return torch.abs(
            self.layer.weight.data
        ) * self.get_statistics().reshape((1, -1))

    def process_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.type(torch.float32)
        self.mean_inp += torch.sum(torch.abs(inp), dim=1)
        self.nsamples += tmp

    def free(self):
        self.mean_inp = None
        torch.cuda.empty_cache()
