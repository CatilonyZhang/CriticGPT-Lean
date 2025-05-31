import fire
from huggingface_hub import HfApi


def push_to_hub(model_dir: str, output_id: str):
    """
    Pushes the model to the hub
    Parameters:
    model_dir (str): The directory of the model
    output_id (str): The Hugging Face model ID (AI-MO/Model-Name)
    """
    api = HfApi()
    api.upload_large_folder(
        folder_path=model_dir, repo_id=output_id, repo_type="model", private=True
    )


if __name__ == "__main__":
    """
    Example usage:
    python -m autoformalizer.model_utils.push_to_hub \
    --model_dir='/DATA/disk7/mert/models/Qwen2.5-PTFull-241229/checkpoint-60000' \
    --output_id='AI-MO/Qwen2.5-PTFull-241229' \
    --output_branch='main'
    """
    fire.Fire(push_to_hub)
