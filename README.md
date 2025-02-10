# Pruning LLMs by Weights and Activations
Official PyTorch implementation of **STADE**, as presented in our paper:

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
Below is an example command for pruning Llama-3.2-1B with Wanda to execute unstructured 50% sparsity with STADE.
```
python main.py \
    --meta-llama/Llama-3.2-1B \
    --prune_method stade \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_3.2_1b/0.5/stade/ 
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA/OPT model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--cuda_device`: Specifies the cuda device number.
- `--prune_method`: We have implemented the pruning methods [`random`, `magnitude`, `wanda`, `sparsegpt`, `std`, `stdnobias`, `stade`, `stadenobias`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--save`: Specifies the directory where the result will be stored.

For structured N:M sparsity, set the argument `--sparsity_type` to "2:4" or "4:8". An illustrative command is provided below:
```
python main.py \
    --meta-llama/Llama-3.2-1B \
    --prune_method stade \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_3.2_1b/2_4/stade/ 
```

### Zero-Shot Evaluation
For evaluating zero-shot tasks, we modify the [EleutherAI LM Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master) framework so that it could evaluate pruned LLM models. Make sure to download, extract and install this custom `lm_eval` package from the source code.

## Acknowledgement
This repository is build upon the [Wanda](https://github.com/locuslab/wanda) repository.
