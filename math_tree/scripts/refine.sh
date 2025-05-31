
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate autoformalization


# Get the LLM URL
llm_url,model_name=$(python3 llm/deploy_llm.py)

# unset http_proxy and https_proxy
unset http_proxy
unset https_proxy

# Refine the text
python3 block_tree/refine.py --model_name $model_name --batch_size 10 --llm_url $llm_url

# echo "$refined_text"