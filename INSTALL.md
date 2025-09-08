# Installation  
Step 1: Create a new conda environment:
```
conda create -n prune_llm python=3.9.20
conda activate prune_llm
```
Step 2: Install relevant packages
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #torch:2.5.1+cu118, torchvision:2.5.1+cu118, torchaudio:2.5.1+cu118
pip install transformers datasets wandb sentencepiece #transformers:2.5.1+cu118, datasets:2.5.1+cu118, 
pip install accelerate #accelerate:0.18.0
pip install functorch #functorch: 2.0.0
# only required for zero_shot
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e . 
```
There are known [issues](https://github.com/huggingface/transformers/issues/22222) with the transformers library on loading the LLaMA tokenizer correctly. Please follow the mentioned suggestions to resolve this issue.