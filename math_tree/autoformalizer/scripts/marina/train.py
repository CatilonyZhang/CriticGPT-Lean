"""
Script for fine-tuning a causal language model on a single example using supervised fine-tuning (SFT).
This script is designed to overfit on a small dataset, which can be useful for debugging. It runs on a single GPU H100 card. 
Peak memory usage is aroung 50GB during training
usage:
>>> python train.py --model_name=deepseek-ai/DeepSeek-Prover-V1.5-RL --epochs=1 --batch_size=1 --lr=3e-5
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import Dataset
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
import fire
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


example = {
        'prompt': """Complete the following Lean 4 code:

```lean4
import Mathlib

theorem algebra_18895 :
  (2 - 2 * Complex.I) * (5 + 5 * Complex.I) = 20 := by
""",
        'response': """  ring
  norm_num
  simp [Complex.I_mul_I]
  norm_num
"""
    }
DATASET = Dataset.from_dict({"prompt": [example["prompt"]] * 8, "response": [example["response"]] * 8})


def init_accelerator():
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,  # Use ZeRO Stage 3 for full sharding
        offload_optimizer_device="cpu",  # Offload optimizer states to CPU to fit in one GPU
        offload_param_device="none",  # none: Keep parameters on GPU, cpu: Offload params to CPU
        gradient_accumulation_steps=1,  # Number of accumulation steps
    )
    accelerator = Accelerator(mixed_precision="bf16", deepspeed_plugin=deepspeed_plugin)
    return accelerator


def load_model(model_name="deepseek-ai/DeepSeek-Prover-V1.5-RL"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    logger.info("Model and tokenizer loaded successfully!")
    return model, tokenizer


def tokenize_dataset(dataset, tokenizer):
    def tokenize(example):
        return {
            "input_ids": tokenizer(example["prompt"])['input_ids'],
            "labels": tokenizer(example["response"])['input_ids']
        }
    return dataset.map(tokenize)


def generate_sample(accelerator, model, tokenizer, prompt):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
        outputs = model.generate(
            **inputs, max_length=500, num_beams=5, early_stopping=True, pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def main(
    model_name="deepseek-ai/DeepSeek-Prover-V1.5-RL", 
    dataset = DATASET,
    epochs=10, 
    batch_size=1, 
    lr=5e-5, 
    sample_every=1
    ):
    accelerator =  init_accelerator()
    model, tokenizer =  load_model()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    def collate_fn(batch):
        inputs = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        combined = torch.tensor([i + l for i, l in zip(inputs, labels)], dtype=torch.long)
        return {"input_ids": combined, "labels": combined.clone()}

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # training loop
    step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            model.train()
            with accelerator.autocast():
                loss = model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            logger.info(f"Epoch {epoch} Step {step}. Loss: {loss.item()}")
            if step % sample_every == 0:
                sample = generate_sample(accelerator, model, tokenizer, example["prompt"])
                with open(f"sample_step_{step:03}.txt", "w") as f:
                    f.write(sample)
            step += 1


if __name__ == "__main__":
    fire.Fire(main)


'''
>>> python train.py --epochs=1 --batch_size=1 --lr=3e-5
Epoch 0 Step 0. Loss: 1.4719412326812744
Epoch 0 Step 1. Loss: 0.44800832867622375
Epoch 0 Step 2. Loss: 0.11573266237974167
Epoch 0 Step 3. Loss: 0.06833910942077637
Epoch 0 Step 4. Loss: 0.006531677674502134
Epoch 0 Step 5. Loss: 0.001125045120716095
Epoch 0 Step 6. Loss: 0.0005079088732600212
Epoch 0 Step 7. Loss: 0.0002948239562101662
'''


''' STEP 7
Complete the following Lean 4 code:

```lean4
import Mathlib

theorem algebra_18895 :
  (2 - 2 * Complex.I) * (5 + 5 * Complex.I) = 20 := by
  ring
  norm_num
  simp [Complex.I_mul_I]
  norm_num
```
'''