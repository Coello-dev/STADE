from .wanda import Wanda
from .stade import Stade, Stade_Star


# Define Wanda class
class StadeW:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", **kwargs):
        # LLama model
        if layer_name in [
            "self_attn.o_proj",
            "mlp.down_proj",
            "self_attn.out_proj",
            "fc2",
        ]:
            self.pruner = Stade(layer=layer, layer_name=layer_name)
        elif layer_name in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "fc1",
        ]:
            self.pruner = Wanda(layer=layer, layer_name=layer_name)
        else:
            raise ValueError(f"Invalid layer name: {layer_name}")

    def add_batch(self, **kwargs):
        self.pruner.add_batch(**kwargs)

    def free(self):
        self.pruner.free()

    def prune(self, sparsity, prune_n, prune_m, **kwargs):
        return self.pruner.prune(sparsity, prune_n, prune_m, **kwargs)


# Define Wanda class
class StadeW_Star:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", **kwargs):
        # LLama model
        if layer_name in [
            "self_attn.o_proj",
            "mlp.down_proj",
            "self_attn.out_proj",
            "fc2",
        ]:
            self.pruner = Stade_Star(layer=layer, layer_name=layer_name)
        elif layer_name in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "fc1",
        ]:
            self.pruner = Wanda(layer=layer, layer_name=layer_name)
        else:
            raise ValueError(f"Invalid layer name: {layer_name}")

    def add_batch(self, **kwargs):
        self.pruner.add_batch(**kwargs)

    def free(self):
        self.pruner.free()

    def prune(self, sparsity, prune_n, prune_m, **kwargs):
        return self.pruner.prune(sparsity, prune_n, prune_m, **kwargs)
