import torch


class RandomPrune:

    def __init__(self, layer, **kwargs):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

    def add_batch(self, **kwargs):
        pass

    def free(self):
        torch.cuda.empty_cache()

    def prune(self, sparsity, prune_n=0, prune_m=0, **kwargs):

        W = self.layer.weight.data
        W_metric = torch.randn_like(W)
        if prune_n != 0:
            W_mask = (torch.zeros_like(W)==1)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            thresh = torch.sort(W_metric.flatten().to(self.dev))[0][int(W.numel()*sparsity)]
            W_mask = (W_metric<=thresh)

        W[W_mask] = 0.0
        self.layer.weight.data = W