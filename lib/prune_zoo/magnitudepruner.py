import torch
from lib.prune_zoo.basepruner import BasePruner


class MagnitudePruner(BasePruner):
    def __init__(self, layer, structure="row", **kwargs):
        super().__init__(layer=layer, structure=structure)

    def process_batch(self, **kwargs):
        pass

    def get_weight_metric(self, **kwargs):
        return torch.abs(self.layer.weight.data)
