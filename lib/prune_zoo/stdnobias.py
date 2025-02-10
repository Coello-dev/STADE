import torch
import torch.nn as nn


# Define WrappedGPT class
class STDNOBIAS:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, **kwargs):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.mean_inp = torch.zeros((self.columns), device=self.dev)
        self.var_inp = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
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
            new_mean_inp = self.mean_inp*self.nsamples/new_nsamples + torch.sum(inp, dim=1)/new_nsamples
            self.var_inp = (self.nsamples-1)*self.var_inp/(new_nsamples-1) + self.nsamples*(self.mean_inp**2)/(new_nsamples-1) - new_nsamples*(new_mean_inp**2)/(new_nsamples-1) + torch.sum(inp**2, dim=1)/(new_nsamples-1)
            # self.var_inp = ((self.nsamples-1)*self.var_inp + self.nsamples*(self.mean_inp**2) - new_nsamples*(new_mean_inp**2) + torch.sum(inp**2, dim=1))/(new_nsamples-1)
        else:
            new_mean_inp = torch.sum(inp, dim=1)/new_nsamples
            self.var_inp = torch.sum((inp-new_mean_inp.reshape(-1,1))**2, dim=1)/(new_nsamples-1)

        self.mean_inp = new_mean_inp
        self.nsamples = new_nsamples

    def get_statistics(self):
        if self.nsamples <= 1:
            return self.mean_inp, torch.zeros((self.columns), device=self.dev)
        else:
            return self.mean_inp, self.var_inp
        
    def prune(
        self, sparsity, prune_n=0, prune_m=0, **kwargs
    ):
        
        inp_mean, inp_var = self.get_statistics()
        W_metric = (self.layer.weight.data ** 2) * (inp_mean.reshape((1,-1))**2 + inp_var.reshape((1,-1)))
        W_mask = (torch.zeros_like(W_metric) == 1)

        if prune_n != 0:
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity)]
            W_mask.scatter_(1, indices, True)

        self.layer.weight.data[W_mask] = 0  ## set weights to zero

    def free(self):
        self.mean_inp = None
        self.var_inp = None
        torch.cuda.empty_cache()