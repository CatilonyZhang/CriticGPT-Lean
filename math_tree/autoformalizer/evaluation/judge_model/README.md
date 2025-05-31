# Judge Model Evaluation Tool

This tool provides automated evaluation of judge models for formal mathematics verification. It analyzes model performance in assessing the correctness of mathematical formalizations.

## judge_model_filter Overview
The Judge Model Evaluation tool compares natural language mathematical statements with their Lean 4 formalizations, using various language models (GPT-4, Claude, Gemini, QwQ, etc.) to evaluate the accuracy of the translations.

### Quick Start

For local model evaluation, simply run:

```bash
# Launch the evaluation
bash judge_model_filter.sh
```

After completion, stop the servers

```bash
bash stop_servers.sh
```

#### Customized Usage
```python
from typing import Dict, List, Callable
from judge_model_filter import get_judge_feedback, parse_formalization_status

# Custom prompt template example
def custom_prompt_template(row: Dict) -> str:
    return f"""
    Context: {row.get('context', '')}
    Problem: {row['problem']}
    Formalization: {row['autoformalization_preview']}
    Additional Notes: {row.get('notes', '')}
    """

# Custom response parser example
def custom_parser(response: str) -> str:
    if "VALID" in response.upper():
        return "Valid"
    elif "INVALID" in response.upper():
        return "Invalid"
    return "Unknown"

# Run with custom configuration
get_judge_feedback(
    input_dataset_id="your/dataset",
    input_dataset_branch="main",
    output_dataset_id="your/results",
    output_dataset_branch="evaluation",
    
    # Customization options
    system_prompt="You are a mathematical formalization expert...",  # Optional system prompt
    prompt_template=custom_prompt_template,  # Custom prompt formatting
    columns_to_use=["problem", "autoformalization_preview", "context", "notes"],  # Required columns
    parse_response=custom_parser,  # Custom response parser
    
    # Additional settings
    num_votes=3,  # Number of votes per example
    temperature=0.0,  # Model temperature
    sample_size=100,  # Optional: process subset of data
    verbose=True
)
```

## judge_model_eval Overview

The Judge Model Evaluation tool compares natural language mathematical statements with their Lean 4 formalizations, using various language models (GPT-4, Claude, Gemini, QwQ, etc.) to evaluate the accuracy of the translations.

## Workflow

The evaluation process consists of three main steps:

1. Launch model servers (for local models)
2. Run evaluation
3. Stop servers (if using local models)

### 1. Launch Model Servers

For local models, use the provided server launch script:

```bash
bash launch_servers.sh
```

This script will:
- Start vLLM servers on specified GPUs
- Generate configuration YAML files in the `config/` directory
- Each model gets its own configuration file (e.g., `config/Qwen7BCoder_JudgemodelV1.yaml`)

Example YAML configuration generated:
```yaml
endpoints:
  - model_path: "Qwen/Qwen7BCoder_JudgemodelV1"
    url: "http://localhost:8082/v1"
  
  - model_path: "Qwen/Qwen7BCoder_JudgemodelV1"
    url: "http://localhost:8083/v1"
```

### 2. Run Evaluation

Use the evaluation script with a configuration file:

```bash
# For local models
bash judge_model_eval.sh config/Qwen7BCoder_JudgemodelV1.yaml

# For cloud APIs (e.g., Claude)
bash judge_model_eval.sh config/claude.yaml
```

Example cloud API configuration (`config/claude.yaml`):
```yaml
endpoints:
  - model_path: "claude-3"
    url: "https://api.anthropic.com/v1"
    api_key: "${ANTHROPIC_API_KEY}"
```

### 3. Stop Servers

After evaluation, stop local model servers:

```bash
bash stop_servers.sh
```




## Configuration Options

| Parameter             | Description                 | Default                           |
| --------------------- | --------------------------- | --------------------------------- |
| `--input_dataset_id`  | Input dataset identifier    | `AI-MO/formalized_preview_241210` |
| `--input_branch`      | Input dataset branch        | `branch`                          |
| `--output_dataset_id` | Output dataset identifier   | Same as input_dataset_id          |
| `--model_name`        | Model to use for evaluation | `gpt-4o`                          |
| `--verbose`           | Enable detailed logging     | `False`                           |
| `--qwq_port`         | Port for QwQ vLLM server   | Value from QWQ_PORT env var       |

## Supported Models

- OpenAI Models (GPT-4, GPT-3.5)
- Anthropic Models (Claude)
- Google Models (Gemini)
- O1 Models (O1-mini, O1-preview)
- QwQ Models (requires local vLLM server)

## Output Format

The tool generates evaluation results in a new branch of the specified dataset. The output branch name follows the format:

```
model_{model_name}_evaluation_{input_branch}
```

## Environment Variables Reference

| Variable          | Description                     | Required For      |
| ---------------- | ------------------------------- | ---------------- |
| OPENAI_API_KEY   | OpenAI API key                  | OpenAI models    |
| ANTHROPIC_API_KEY| Anthropic API key               | Claude models    |
| QWQ_MODEL_PATH   | Path to QwQ model directory     | QwQ models       |
| QWQ_PORT         | Port for vLLM server           | QwQ models       |