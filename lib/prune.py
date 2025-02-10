import time 
import torch 
import torch.nn as nn
import numpy as np

from .prune_zoo.sparsegpt import SparseGPT
from .prune_zoo.wanda import Wanda
from .prune_zoo.randomprune import RandomPrune
from .prune_zoo.magnitudeprune import MagnitudePrune
from .prune_zoo.std import STD
from .prune_zoo.stdnobias import STDNOBIAS
from .prune_zoo.stade import STADE, STADENOBIAS

from .data import get_loaders 
from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(args, model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if 'llama' in args.model:
        layers = model.model.layers
    elif 'opt' in args.model:
        layers = model.model.decoder.layers
    else:
        raise Exception(f'Invalid model name: {args.model}')
    count = 0 
    total_params = 0
    log = ''
    for i in range(len(layers)):
        print(f"\n--------layer [{i}/{len(layers)}]--------")
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_count = (W==0).sum().item()
            W_params = W.numel()

            count += (W==0).sum().item()
            total_params += W.numel()
            print(f"{name} sparsity {float(W_count)/W_params:.6f}")

    model.config.use_cache = use_cache

def prepare_calibration_input(args, model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if 'llama' in args.model:
        layers = model.model.layers
    elif 'opt' in args.model:
        layers = model.model.decoder.layers
    else:
        raise Exception(f'Invalid model: {args.model}')

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inp = inp.cpu()
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            # Move everything to cpu otherwise it will stay in cuda later on
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    kwargs[key] = value.cpu()
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            batch_i = batch[0].to(device)
            model(batch_i)
        except ValueError:
            pass
    
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    # Move to cuda
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)
    if isinstance(position_ids, torch.Tensor):
        position_ids = position_ids.to(device)

    return inps, outs, attention_mask, position_ids


@torch.no_grad()
def prune_process(args, model, tokenizer, dev, prune_n=0, prune_m=0):

    # Get pruning method
    layer_wrapper_dict = {
        'random': RandomPrune,
        'magnitude': MagnitudePrune,
        'sparsegpt': SparseGPT,
        'wanda': Wanda,
        'stade': STADE,
        'stadenobias': STADENOBIAS,
        'std': STD,
        'stdnobias': STDNOBIAS,
    }
    layer_wrapper = layer_wrapper_dict[args.prune_method]
    print(f'Pruning method: {args.prune_method}')

    print("Loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("Calibration dataset loaded")

    inps, outs, attention_mask, position_ids = prepare_calibration_input(args=args, model=model, dataloader=dataloader, device=dev, nsamples=args.nsamples)


    # Make it gpu memory bound efficient
    inps = inps.cpu()
    outs = outs.cpu()

    if 'llama' in args.model:
        layers = model.model.layers

        # Get position embeddings
        model.model.rotary_emb.to(dev)
        position_embeddings = model.model.rotary_emb(inps[0].unsqueeze(0), position_ids)
        model.model.rotary_emb.cpu()

    elif 'opt' in args.model:
        layers = model.model.decoder.layers
    else:
        raise Exception(f'Invalid model: {args.model}')
    
    if not args.filter_layer is None:
        print(f'Layers filtered by {args.filter_layer}')

    print('Start prunning...')

    for i in range(len(layers)):
        layer = layers[i]
        layer.to(dev)
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            if args.filter_layer is None:
                gpts[name] = layer_wrapper(layer=subset[name], layer_name=name, layer_block=layer)
            elif name in args.filter_layer:
                gpts[name] = layer_wrapper(layer=subset[name], layer_name=name, layer_block=layer)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp=inp[0].data, out=out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                inp_j = inps[j].to(dev)
                if 'llama' in args.model:
                    layer(inp_j.unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                elif 'opt' in args.model:
                    layer(inp_j.unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                else:
                    raise Exception(f'Invalid model: {args.model}')
                inp_j = inp_j.cpu()
    
        for h in handles:
            h.remove()

        for name in gpts:
            gpts[name].prune(sparsity=args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            inp_j = inps[j]
            inp_j = inp_j.to(dev)
            if 'llama' in args.model:
                outs_j = layer(inp_j.unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
            elif 'opt' in args.model:
                outs_j = layer(inp_j.unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            else:
                raise Exception(f'Invalid model: {args.model}')
            inp_j = inp_j.cpu()
            outs_j = outs_j.cpu()
            outs[j] = outs_j

        layer.cpu()
        layers[i] = layer 
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()