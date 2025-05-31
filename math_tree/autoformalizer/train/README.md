# How to run a training?

1. First, ensure that you have installed the conda environment using the `autoformalization.yml` file in the top directory of this repository. You can do this by running the following command:

```bash
conda env create -f autoformalization.yml
conda activate autoformalization
```

2. In order for LLaMa-Factory to recognize your dataset, you need to add your dataset to `data/dataset_info.json` file. For example, for the [AutoformalizationV1_Alpaca](https://huggingface.co/datasets/AI-MO/AutoformalizationV1_Alpaca) instruction fine-tuning dataset, this looks as follows:

```json
  "autoformalization_alpaca": {
    "hf_hub_url": "AI-MO/AutoformalizationV1_Alpaca",
    "ms_hub_url": "AI-MO/AutoformalizationV1_Alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
```

For full description of how to do this for other dataset formats, check out [LLaMa-Factory data documentation](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md).

3. Finally, you can set your training configuration in `config/your_folder/name_of_your_config_file.yaml` and run the training using the following command:

```bash
cd train
FORCE_TORCHRUN=1 llamafactory-cli train config/name_of_your_config_file.yaml 
```

An example of a training configuration file can be found in `config/mert/Qwen7B_Autoformalizer1.yaml`. It is recommended that you create your own folder under config, put your `.yaml` file there and run as described above. To find out the full set of parameters for the YAML configuration file, run `llamafactory-cli train -h` or check out the [examples at LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md).
