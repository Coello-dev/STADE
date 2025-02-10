import argparse
import os
import sys
import time
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from importlib.metadata import version

from lib.prune import prune_process, prune_ablate, check_sparsity
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=False

def get_llm(model_name, cache_dir="llm_weights"):
    
    llm_path = cache_dir + '/' + model_name
    print(f"Model weight path: {llm_path}")

    model = AutoModelForCausalLM.from_pretrained(
        llm_path, 
        torch_dtype=torch.float16, 
        local_files_only=True,
        low_cpu_mem_usage=False
    )
    print(f"Model loaded succesfully")

    model.seqlen = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    print(f"Tokenizer loaded")

    return model, tokenizer

def get_argparser():

    # Get argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLama model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.') #Default:128
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default='llm_weights/temp', help='Path to save the pruned model.')
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--filter_layer",type=str,default=None,help="If none given all layers will be pruned. Otherwise only the given ones.",
                        choices=[None,"self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","self_attn.out_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj","mlp.fc1","mlp.fc2"])
    parser.add_argument('--weighted_matrices', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')
    args = parser.parse_args()

    # Show argparser
    parser_info = [f" - {k}: {v} \n" for k, v in args.__dict__.items()]
    print(f'Argparser:\n{"".join(parser_info)}')

    # Return args
    return args


def main():

    # Starting time
    print(f'Experiment starting time: {time.strftime("%H:%M:%S", time.localtime())}')

    # Check enviroment state
    print(f'Python system info: {sys.version}')
    print(f"File location using os.getcwd(): {os.getcwd()}")

    # Check GPU access
    use_cuda = torch.cuda.is_available()
    print(f'Torch version: {torch.__version__}, Cuda available: {use_cuda}')
    print(f"GPU info from pytorch: {torch.cuda.get_device_properties('cuda').name}")

    # Get args
    args = get_argparser()
    args.save = args.save.replace(":","_")

    # Check GPU meory information
    device_cuda = torch.device(f'cuda:{args.cuda_device}')

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    print(f"Loading llm model {args.model}")
    model, tokenizer = get_llm(args.model, args.cache_dir, device=device_cuda)
    model.eval()

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "dense":
            print(f'Using pruning method {args.prune_method}, a.k.a. NO prunning is done')
        else:
            prune_process(args=args, model=model, tokenizer=tokenizer, dev=device_cuda, prune_n=prune_n, prune_m=prune_m)
    else:
        print(f'Sparsity {args.sparsity_ratio}, a.k.a. NO prunning is done')  
        
    ################################################################
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")

    if not args.skip_wiki:
        print("*"*30)
        sparsity_ratio = check_sparsity(args=args, model=model, path=save_filepath)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*"*30)
    ################################################################
        ppl_test = eval_ppl(args, model, tokenizer, device_cuda)
        print(f"wikitext perplexity {ppl_test}")

        # Save model result in file
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test\tfilter_layer", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{args.filter_layer}\n", file=f, flush=True)

    if args.eval_zero_shot:

        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args=args, task_list=task_list, num_fewshot=num_shot, use_accelerate=accelerate, device=f'cuda:{args.cuda_device}')
        print("********************************")
        print("zero_shot evaluation results")
        print(results['results'])


        save_filepath_zero = os.path.join(args.save, f"zero_{args.prune_method}.txt")
        results
        with open(save_filepath_zero, "w") as f:
            print(f"dataset\taccuracy", file=f, flush=True)
            results_list = []
            for dataset, result_dataset in results['results'].items():
                print(f"{dataset}\t{result_dataset['acc,none']:.6f}", file=f, flush=True)
                results_list.append(result_dataset['acc,none'])
            results_avg = np.array(results_list).mean()
            print(f"Average\t{results_avg:.6f}", file=f, flush=True)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()