import torch


class BasePruner:
    def __init__(self, layer, structure="row", **kwargs):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.structure = structure
        assert structure in [
            "row",
            "column",
            "all",
            "all_outnorm",
            "all_outnorm2",
            "all_inpnorm",
            "all_inpnorm2",
        ], f"Invalid structure {structure}"
        if self.structure in ["all_outnorm", "all_outnorm2"]:
            self.out_norm = torch.zeros((self.rows), device=self.dev)
        elif self.structure in ["all_inpnorm", "all_inpnorm2"]:
            self.inp_norm = torch.zeros((self.columns), device=self.dev)

    def add_batch(self, **kwargs):
        self.process_batch(**kwargs)
        self.store_metric(**kwargs)

    def store_metric(self, inp, out):
        if self.structure in ["all_outnorm", "all_outnorm2"]:
            self.out_norm += torch.norm(out, p=2, dim=1) ** 2
        elif self.structure in ["all_inpnorm", "all_inpnorm2"]:
            self.inp_norm += torch.norm(out, p=2, dim=0) ** 2

    def process_batch(self, **kwargs):
        raise NotImplementedError

    def get_weight_metric(self, **kwargs):
        raise NotImplementedError

    def free(self):
        torch.cuda.empty_cache()

    def update_weights(self, **kwargs):
        pass

    def prune(self, sparsity, prune_n=0, prune_m=0, **kwargs):
        W = self.layer.weight.data
        W_metric = self.get_weight_metric()
        W_mask = torch.zeros_like(W) == 1
        if prune_n != 0:
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii : (ii + prune_m)].float()
                    W_mask.scatter_(
                        1,
                        ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                        True,
                    )
        else:
            if self.structure == "row":
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, : int(W_metric.shape[1] * sparsity)]
                W_mask.scatter_(1, indices, True)
            elif self.structure == "column":
                sort_res = torch.sort(W_metric, dim=0, stable=True)
                indices = sort_res[1][: int(W_metric.shape[1] * sparsity), :]
                W_mask.scatter_(1, indices, True)
            elif self.structure == "all":
                thresh = torch.sort(W_metric.flatten().to(self.dev))[0][
                    int(W.numel() * sparsity)
                ]
                W_mask = W_metric <= thresh
                thresh = thresh.cpu()
            elif self.structure == "all_outnorm":
                out_norm = self.out_norm.reshape(-1, 1)
                thresh = torch.sort(W_metric.flatten().to(self.dev) * out_norm)[
                    0
                ][int(W.numel() * sparsity)]
                W_mask = W_metric <= thresh
                thresh = thresh.cpu()
            elif self.structure == "all_outnorm2":
                out_norm = torch.sqrt(self.out_norm.reshape(-1, 1))
                thresh = torch.sort(W_metric.flatten().to(self.dev) * out_norm)[
                    0
                ][int(W.numel() * sparsity)]
                W_mask = W_metric <= thresh
                thresh = thresh.cpu()
            elif self.structure == "all_inpnorm":
                inp_norm = self.inp_norm.reshape(1, -1)
                thresh = torch.sort(W_metric.flatten().to(self.dev) * inp_norm)[
                    0
                ][int(W.numel() * sparsity)]
                W_mask = W_metric <= thresh
                thresh = thresh.cpu()
            elif self.structure == "all_inpnorm2":
                inp_norm = torch.sqrt(self.inp_norm.reshape(1, -1))
                thresh = torch.sort(W_metric.flatten().to(self.dev) * inp_norm)[
                    0
                ][int(W.numel() * sparsity)]
                W_mask = W_metric <= thresh
                thresh = thresh.cpu()

        self.layer.weight.data[W_mask] = 0.0
        self.update_weights(W_mask=W_mask)
