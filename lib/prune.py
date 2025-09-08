import time
import torch
import torch.nn as nn
import numpy as np

from .prune_zoo.sparsegpt import SparseGPT
from .prune_zoo.sparsegptnoupdate import SparseGPTNoupdate
from .prune_zoo.wanda import Wanda
from .prune_zoo.randompruner import RandomPruner
from .prune_zoo.magnitudepruner import MagnitudePruner
from .prune_zoo.stade import Stade, Stade_Star
from .prune_zoo.stadew import StadeW, StadeW_Star

from .utils import Catcher
from .data import get_loaders
# from .forward_pass import forward_pass_gpu_constrained


def find_layers(module, layers=[nn.Linear], name=""):
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
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def check_sparsity(args, model, path=None):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "llama" in args.model:
        layers = model.model.layers
    elif "Qwen" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        raise Exception(f"Invalid model name: {args.model}")
    count = 0
    total_params = 0
    log = ""
    for i in range(len(layers)):
        print(f"\n--------layer [{i}/{len(layers)}]--------")
        if not path is None:
            log += f"\n--------layer {i}--------\n"
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_count = (W == 0).sum().item()
            W_params = W.numel()

            count += (W == 0).sum().item()
            total_params += W.numel()

            if subset[name].bias is None:
                print(f"{name} sparsity {float(W_count) / W_params:.6f}")
                if not path is None:
                    log += f"{name} sparsity {float(W_count) / W_params:.6f}\n"

            else:
                B = subset[name].bias.data
                B_count = (B != 0).sum().item()
                # print(f"{name} sparsity {float(W_count)/W_params:.6f}")
                print(
                    f"{name} sparsity w bias {float(W_count - B_count) / W_params:.6f} | bias abs sum {torch.abs(B).sum().item():.4f}"
                )
                if not path is None:
                    # log += f"{name} sparsity {float(W_count)/W_params:.6f}\n"
                    log += f"{name} sparsity w bias {float(W_count - B_count) / W_params:.6f} | bias abs sum {torch.abs(B).sum().item():.4f}\n"

    model.config.use_cache = use_cache
    if path is None:
        return float(count) / total_params
    else:
        return float(count) / total_params, log


def prepare_calibration_input(args, model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "llama" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    elif "Qwen" in args.model:
        layers = model.model.layers
    else:
        raise Exception(f"Invalid model: {args.model}")

    # dev = model.hf_device_map["model.embed_tokens"]
    # if "model.embed_tokens" in model.hf_device_map:
    #     device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    print(f"Device: {device}")
    # inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {
        "i": 0,
        "catcher_attention_mask": None,
        "catcher_position_ids": None,
    }

    layers[0] = Catcher(module=layers[0], inps=inps, cache=cache)
    for batch in dataloader:
        try:
            batch_i = batch[0].to(device)
            model = model(batch_i)
            # forward_pass_gpu_constrained(
            #     args=args, model=model, input_ids=batch_i, device=device
            # )
        except ValueError:
            pass

    outs = torch.zeros_like(layers[0].inps)
    attention_mask = layers[0].cache["catcher_attention_mask"]
    position_ids = layers[0].cache["catcher_position_ids"]
    model.config.use_cache = use_cache

    # Remove model from cuda
    batch_i.cpu()
    model.cpu()

    # Move to cuda
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)
    if isinstance(position_ids, torch.Tensor):
        position_ids = position_ids.to(device)

    layers[0] = layers[0].module

    return inps, outs, attention_mask, position_ids


@torch.no_grad()
def prune_process(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    # Get pruning method
    layer_wrapper_dict = {
        # Easy baseline
        "random": RandomPruner,
        "magnitude": MagnitudePruner,
        # Wanda
        "wanda": Wanda,
        # STADE
        "stade": Stade,
        "stadew": StadeW,
        "stade_star": Stade_Star,
        "stadew_star": StadeW_Star,
        # SparseGPT
        "sparsegpt": SparseGPT,
        "sparsegptnoupdate": SparseGPTNoupdate,
    }
    layer_wrapper = layer_wrapper_dict[args.prune_method]
    print(f"Pruning method: {args.prune_method}")

    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("Dataset loaded")

    inps, outs, attention_mask, position_ids = prepare_calibration_input(
        args=args,
        model=model,
        dataloader=dataloader,
        device=dev,
        nsamples=args.nsamples,
    )
    print("Calibration dataset prepared")

    # Make it gpu memory bound efficient
    inps = inps.cpu()
    outs = outs.cpu()

    if "llama" in args.model:
        layers = model.model.layers

        # Get position embeddings
        model.model.rotary_emb.to(dev)
        inps = inps.to(dev)
        position_embeddings = model.model.rotary_emb(
            inps[0].unsqueeze(0), position_ids
        )
        inps = inps.cpu()
        model.model.rotary_emb.cpu()

    elif "opt" in args.model:
        layers = model.model.decoder.layers

    elif "Qwen" in args.model:
        layers = model.model.layers
        model.model.rotary_emb.to(dev)
        inps = inps.to(dev)
        position_embeddings = model.model.rotary_emb(inps, position_ids)
        inps = inps.cpu()
        model.model.rotary_emb.cpu()

    else:
        raise Exception(f"Invalid model: {args.model}")

    if not args.filter_layer is None:
        print(f"Layers filtered by {args.filter_layer}")

    avg_layer_time = 0
    old_outs = [0.0 for _ in range(len(outs))]
    coss_diff = []
    l2_diff = []

    print("Start prunning...")

    for i in range(len(layers)):
        start_layer_time = time.time()
        layer = layers[i]
        layer.to(dev)
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            if args.filter_layer is None:
                gpts[name] = layer_wrapper(
                    layer=subset[name], layer_name=name, layer_block=layer
                )
            elif name in args.filter_layer:
                gpts[name] = layer_wrapper(
                    layer=subset[name], layer_name=name, layer_block=layer
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp=inp[0].data, out=out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        batch_avg_time = 0.0
        for j in range(args.nsamples):
            batch_start_time = time.time()
            with torch.no_grad():
                inp_j = inps[j].to(dev)
                if "llama" in args.model:
                    outs_j = layer(
                        inp_j.unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
                elif "opt" in args.model:
                    outs_j = layer(
                        inp_j.unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
                elif "Qwen" in args.model:
                    outs_j = layer(
                        inp_j.unsqueeze(0),
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
                else:
                    raise Exception(f"Invalid model: {args.model}")
                inp_j = inp_j.cpu()
                outs_j = outs_j.cpu()
                old_outs[j] = outs_j
            batch_end_time = time.time()
            batch_avg_time = batch_avg_time * j / (j + 1) + (
                batch_end_time - batch_start_time
            ) / (j + 1)
            # if j%20 == 0:
            #     print(f'Bacth [{j+1}/{args.nsamples}] | AVG: {batch_avg_time} | Current: {batch_end_time-batch_start_time}')
        for h in handles:
            h.remove()

        for name in gpts:
            gpts[name].prune(
                sparsity=args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()
            print(f"Layer {i} {name}")

        l2_loss_layer = 0.0
        cos_sim_layer = 0.0
        for j in range(args.nsamples):
            inp_j = inps[j]
            inp_j = inp_j.to(dev)
            if "llama" in args.model:
                outs_j = layer(
                    inp_j.unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            elif "opt" in args.model:
                outs_j = layer(
                    inp_j.unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            elif "Qwen" in args.model:
                outs_j = layer(
                    inp_j.unsqueeze(0),
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                )[0]
            else:
                raise Exception(f"Invalid model: {args.model}")
            inp_j = inp_j.cpu()
            old_j = old_outs[j]
            old_j = old_j.to(dev)
            l2_loss_layer += torch.nn.functional.mse_loss(old_j, outs_j)
            cos_sim_layer += torch.nn.functional.cosine_similarity(
                old_j, outs_j
            ).mean()
            old_j = old_j.cpu()
            outs_j = outs_j.cpu()
            outs[j] = outs_j

        print(
            f"Prunning {i} | Layer L2 loss: {l2_loss_layer / args.nsamples} | Layer Cos sim: {cos_sim_layer / args.nsamples}"
        )
        coss_diff.append(cos_sim_layer / args.nsamples)
        l2_diff.append(l2_loss_layer / args.nsamples)

        layer.cpu()
        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        end_layer_time = time.time()
        avg_layer_time = avg_layer_time * i / (i + 1) + (
            end_layer_time - start_layer_time
        ) / (i + 1)
        print(
            f"Layer [{i + 1}/{len(layers)}] | AVG: {avg_layer_time} | Current: {end_layer_time - start_layer_time}"
        )
        torch.cuda.empty_cache()

    # Return logs
    return coss_diff, l2_diff
