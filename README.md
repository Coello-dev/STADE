# Pruning LLMs with STADE
Official PyTorch implementation of **STADE**, as presented in our paper:
[STADE: Standard Deviation as a Pruning Metric](https://arxiv.org/abs/2503.22451)

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
- `--prune_method`: We have implemented the following pruning methods [`random`, `magnitude`, `wanda`, `sparsegpt`, `stade`, `stadew`, `stade_plus`].
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

## Citation

// If you found this work useful, please consider citing:

```
@misc{mecke2025stadestandarddeviationpruning,
      title={STADE: Standard Deviation as a Pruning Metric},
      author={Diego Coello de Portugal Mecke and Haya Alyoussef and Maximilian Stubbemann and Ilia Koloiarov and Tom Hanika and Lars Schmidt-Thieme},
      year={2025},
      eprint={2503.22451},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.22451},
}
```

## Acknowledgement
This repository is build upon the [Wanda](https://github.com/locuslab/wanda) repository.
