import json
from typing import Dict, Any
from openai import OpenAI
from math_tree.autoformalizer.autoformalizer.eval_utils.constants import gpt_verification_system_prompt

class LLMEvaluator:
    """LLM 评估器
    
    负责：
    1. 评估代码的正确性
    2. 检查语义匹配度
    3. 提供改进建议
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = OpenAI(
            api_key=config.get("openai_api_key", ""),
            base_url=config.get("openai_base_url", "https://api.openai.com/v1")
        )
    
    def evaluate(self, problem: str, lean_code: str, compilation_result: Dict[str, Any]) -> Dict[str, Any]:
        """评估代码质量和正确性"""
        messages = [
            {"role": "system", "content": gpt_verification_system_prompt},
            {"role": "user", "content": f"""请评估以下 Lean4 代码的质量和正确性：

原始问题：
{problem}

Lean4 代码：
{lean_code}

编译结果：
{json.dumps(compilation_result, ensure_ascii=False, indent=2)}

请详细分析：
1. 代码是否准确捕捉了原始问题的数学含义
2. 是否存在逻辑或语义错误
3. 代码结构和风格是否合适
4. 如果有问题，应该如何改进"""}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=self.config.get("openai_model", "gpt-4"),
            messages=messages,
            temperature=0.2
        )
        
        return {
            "evaluation": response.choices[0].message.content,
            "is_semantically_correct": "incorrect" not in response.choices[0].message.content.lower()
        } 