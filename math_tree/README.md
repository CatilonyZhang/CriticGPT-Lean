# math_tree

## TODO List
- [x] Coding: Statement Refinement via Doubao

- [x] Data Processing: Refine the first batch of informal dataset: Omni-math

- [ ] Refine more informal math data 

- [ ] Autoformalization: Finish the Omni-math's statement autoformalization, with ``sorry'' 

- [ ] Whole proof generation: run deepseek-prover to obtain the first benchmarking on Omni-math

- [ ] InternLM demo checking https://github.com/project-numina/autoformalizer/pull/156

- [ ] BFS with Value function https://github.com/project-numina/autoformalizer/pull/157

- [x] Autoformalization with compiler feedback 

- [x] Autoformalization infering pipeline 

- [ ] Autoformalization training via qwen-coder 14B (In this case we can use VML)

- [ ] deepseek-like MCTS https://github.com/deepseek-ai/DeepSeek-Prover-V1.5/blob/main/prover/algorithms/rmax_tree_search.py

## Statement Refinement

To quick start
```shell
cd refine
python refine.py --config refine/config/omnibench_refine.yaml --output-dir eval_logs
```

To refine other informal math datasets, refer to `refine/config/omnibench_refine.yaml`. 
Create a new config file and 
Change the corresponding key-value in the config.

```yaml
# Dataset configuration
data_path: 'KbsdJames/Omni-MATH'
data_split: ['test']
informal_statement: 'refined_statement'
answer: 'answer'
informal_proof: 'solution'
difficulty: 'difficulty'
domain: 'domain'

has_header: true # if false, adds a default header

# llm_config
batch_size: 32
model_path: 'ep-20250128143818-wq8s5'
sampling_config:
  n: 1
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.95
  stop: ['```']
```
For any model, no matter vllm or web-api, it is recommand to use OpenAI APIs. 



## Autoformalization
```python
cd autoformalizer/autoformalizer/model_utils

export math_tree=/lustre/fast/fast/txiao/zly/lean/math_tree # your project dir

python -m autoformalizer.model_utils.infer_hf_dataset infer_hf_dataset \
    --model_path="/home/mert/models/Qwen7BCoder_AutoformalizerV1B4"  \
    --dataset_id="AI-MO/AutoformalizationEvalV2" \
    --output_dataset_id="AI-MO/AutoformalizationEvalV2" \
    --output_dataset_branch="Qwen7B_AutoformalizerV1B4"
```





## Some Reminders
### huggingface_hub 
1. `huggingface_hub >=0.23` would lead to the issue 
```
NotImplementedError: FileSystem does not appear to support flock; use SoftFileLock instead
```
when try to cache the downloaded ckpt in the local dir

A recommand way to download the privete model is
```
git-lfs clone git@hf.co:AI-MO/Qwen7BCoder_AutoformalizerV3B0
git-lfs clone git@hf.co:<model_name>
```

### Mathlib, Lean, Repl installation
Lean installation
```
cd ~
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env
```

REPL installation
```
git clone https://github.com/leanprover-community/repl.git && cd repl && git checkout adbbfcb9d4e61c12db96c45d227de92f21cc17dd
lake build
cd ..
```

Mathlib installation
```
cd ~/repl/test/Mathlib
bash test.sh

go to ``autoformalizer/autoformalizer/eval_utils/constants.py``
base = os.environ.get("AUTOFORMALIZER_WORKSPACE")
path_to_repl = f"{base}/repl/.lake/build/bin/repl"
path_to_mathlib = f"{base}/mathlib4"

```





