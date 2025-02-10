# Import necessary modules
import time
import torch
import torch.nn as nn

from tqdm import tqdm

# Import get_loaders function from data module within the same directory
from .data import get_loaders 
from .forward_pass import forward_pass_gpu_constrained

from collections import defaultdict
import fnmatch


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


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(args, model, testloader, 1, device)
    return ppl_test


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(args, model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous().to(device)
        shift_labels = inputs[:, 1:].to(device)

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(args, model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    with tqdm(range(0,nsamples,bs), unit='batch') as tepoch:
        for i in tepoch:

            # Show epoch descriptor
            tepoch.set_description(f'Wikitex evaluation | Batch {i}')


            # Calculate end index
            j = min(i+bs, nsamples)

            # Prepare inputs and move to device
            inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
            inputs = inputs.reshape(j-i, model.seqlen)

            # Forward pass through the model
            output = model(inputs).logits
            lm_logits = output.logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous().to(device)
            shift_labels = inputs[:, 1:].to(device)

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            shift_labels = shift_labels
            loss = loss.cpu()

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * model.seqlen * (j-i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

            # Show logs
            tepoch.set_postfix(ppl=torch.exp(torch.stack(nlls).sum() / (j * model.seqlen)).item())

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(args, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False, device=None):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    #task_names = pattern_match(task_list, tasks.ALL_TASKS)
    task_names = task_list
    model_args = f"pretrained=llm_weights/temp,cache_dir=./llm_weights/temp"
    limit = None 
    if "70b" in args.model or "65b" in args.model:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained=llm_weights/temp,cache_dir=./llm_weights/temp,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model='hf',
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=device,
        limit=limit,
        check_integrity=False,
    )

    return results 