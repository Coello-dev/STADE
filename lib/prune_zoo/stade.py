import torch
import torch.nn as nn
from lib.prune_zoo.basepruner import BasePruner


# Define WrappedGPT class
class Stade(BasePruner):
    def __init__(self, layer, structure="row", **kwargs):
        super().__init__(layer=layer, structure=structure)

        self.mean_inp = torch.zeros((self.columns), device=self.dev)
        self.var_inp = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def process_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[1]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.type(torch.float32)
        new_nsamples = self.nsamples + batch_size
        if self.nsamples:
            new_mean_inp = (
                self.mean_inp * self.nsamples / new_nsamples
                + torch.sum(inp, dim=1) / new_nsamples
            )
            self.var_inp = (
                (self.nsamples - 1) * self.var_inp / (new_nsamples - 1)
                + self.nsamples * (self.mean_inp**2) / (new_nsamples - 1)
                - new_nsamples * (new_mean_inp**2) / (new_nsamples - 1)
                + torch.sum(inp**2, dim=1) / (new_nsamples - 1)
            )
            # self.var_inp = ((self.nsamples-1)*self.var_inp + self.nsamples*(self.mean_inp**2) - new_nsamples*(new_mean_inp**2) + torch.sum(inp**2, dim=1))/(new_nsamples-1)
        else:
            new_mean_inp = torch.sum(inp, dim=1) / new_nsamples
            self.var_inp = torch.sum(
                (inp - new_mean_inp.reshape(-1, 1)) ** 2, dim=1
            ) / (new_nsamples - 1)

        self.mean_inp = new_mean_inp
        self.nsamples = new_nsamples

    def get_statistics(self):
        return self.mean_inp, self.var_inp

    def get_weight_metric(self, **kwargs):
        _, inp_var = self.get_statistics()
        return (self.layer.weight.data**2) * inp_var.reshape((1, -1))

    def update_weights(self, W_mask, **kwargs):
        W = self.layer.weight.data
        bias_data = (
            self.mean_inp @ (W * W_mask.type(torch.float32)).transpose(0, 1)
        ).type(W.dtype)
        if self.layer.bias is None:
            self.layer.bias = torch.nn.Parameter(bias_data, requires_grad=False)
        else:
            self.layer.bias = torch.nn.Parameter(
                self.layer.bias + bias_data, requires_grad=False
            )

    def free(self):
        self.mean_inp = None
        self.var_inp = None
        torch.cuda.empty_cache()


# Define WrappedGPT class
class Stade_Star(BasePruner):
    """
    Ablation version of STADE from the appendix
    """

    def __init__(self, layer, structure="row", **kwargs):
        super().__init__(layer=layer, structure=structure)

        self.mean_inp = torch.zeros((self.columns), device=self.dev)
        self.var_inp = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def process_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[1]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.type(torch.float32)
        new_nsamples = self.nsamples + batch_size
        if self.nsamples:
            new_mean_inp = (
                self.mean_inp * self.nsamples / new_nsamples
                + torch.sum(inp, dim=1) / new_nsamples
            )
            self.var_inp = (
                (self.nsamples - 1) * self.var_inp / (new_nsamples - 1)
                + self.nsamples * (self.mean_inp**2) / (new_nsamples - 1)
                - new_nsamples * (new_mean_inp**2) / (new_nsamples - 1)
                + torch.sum(inp**2, dim=1) / (new_nsamples - 1)
            )
            # self.var_inp = ((self.nsamples-1)*self.var_inp + self.nsamples*(self.mean_inp**2) - new_nsamples*(new_mean_inp**2) + torch.sum(inp**2, dim=1))/(new_nsamples-1)
        else:
            new_mean_inp = torch.sum(inp, dim=1) / new_nsamples
            self.var_inp = torch.sum(
                (inp - new_mean_inp.reshape(-1, 1)) ** 2, dim=1
            ) / (new_nsamples - 1)

        self.mean_inp = new_mean_inp
        self.nsamples = new_nsamples

    def get_statistics(self):
        return self.mean_inp, self.var_inp

    def get_weight_metric(self, **kwargs):
        inp_mean, inp_var = self.get_statistics()
        inp_weight = inp_var + inp_mean**2
        return (self.layer.weight.data**2) * inp_weight.reshape((1, -1))

    def free(self):
        self.mean_inp = None
        self.var_inp = None
        torch.cuda.empty_cache()
