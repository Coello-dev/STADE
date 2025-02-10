import torch
import torch.nn as nn


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


# Define Wanda class
class Wanda:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", **kwargs):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()

    def prune(self, sparsity, prune_n, prune_m, use_variant=False, **kwargs):

        W_metric = torch.abs(self.layer.weight.data) * torch.sqrt(self.scaler_row.reshape((1,-1)))

        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        if prune_n != 0:
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            if use_variant:
                # wanda variant 
                tmp_metric = torch.cumsum(sort_res[0], dim=1)
                sum_before = W_metric.sum(dim=1)

                alpha = 0.4
                alpha_hist = [0., 0.8]
                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                while (torch.abs(cur_sparsity - sparsity)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                    if cur_sparsity > sparsity:
                        alpha_new = (alpha + alpha_hist[0]) / 2.0
                        alpha_hist[1] = alpha
                    else:
                        alpha_new = (alpha + alpha_hist[1]) / 2.0
                        alpha_hist[0] = alpha

                    alpha = alpha_new 
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
            else:
                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity)]
                W_mask.scatter_(1, indices, True)
        
        self.layer.weight.data[W_mask] = 0