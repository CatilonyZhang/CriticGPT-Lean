import os
import yaml
import json
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
from autoformalizer.clients import lean4_client
from autoformalizer.eval_utils.lean_feedback import has_error, parallel_lean4_feedback
from evaluation.evaluation_constants import LEAN4_DEFAULT_HEADER
from autoformalizer.eval_utils.constants import gpt_verification_system_prompt

class MathAutoCoder:
    """数学问题自动形式化 Pipeline
    
    支持:
    1. 数学问题到 Lean4 代码的转换
    2. Lean4 代码验证
    3. GPT 反馈和改进
    """
    
    def __init__(self, config_path: str):
        """初始化 MathAutoCoder
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.openai_client = OpenAI(
            api_key=self.config.get("openai_api_key", ""),
            base_url=self.config.get("openai_base_url", "https://api.openai.com/v1")
        )
        
        # Lean4 客户端配置
        if self.config.get("lean_feedback", "local") == "server":
            self.lean_client = lean4_client.Lean4Client(
                "https://kimina.saas.moonshot.cn/lean4-evaluator",
                api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY"),
            )
        
        # 使用代码库中的专业提示词
        self.default_system_prompt = """You are an expert Lean 4 programmer and mathematician, known for your precision and perfectionism.
Your task is to convert mathematical problems into formal Lean 4 proofs.

Please pay special attention to:
1. The type Nat of natural numbers in Lean 4 includes 0
2. Subtraction of Nat in Lean 4 is truncated
3. Division of Nat or Int in Lean 4 rounds down
4. Use appropriate type coercions when needed (e.g. Int, Rat, Real)

Ensure your formalization:
1. Is syntactically correct Lean 4 code
2. Captures the exact mathematical meaning
3. Uses appropriate mathematical types and operators
4. Includes all necessary imports and declarations"""

        # 代码生成提示词模板
        self.code_generation_template = """Complete the following Lean 4 code:

```lean4
{formal_statement}
"""
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration: {str(e)}")

    def _prepare_formal_statement(self, math_problem: str) -> str:
        """准备形式化声明
        
        Args:
            math_problem: 数学问题描述
            
        Returns:
            形式化声明
        """
        # 提取定理名称和类型信息
        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": f"""请分析以下数学问题，并提供：
1. 合适的定理名称
2. 所需的类型和变量声明
3. 定理的形式化表述

问题：
{math_problem}"""}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=self.config.get("openai_model", "gpt-4"),
            messages=messages,
            temperature=0.2  # 使用较低的温度以获得更确定的结果
        )
        
        return response.choices[0].message.content

    def math_to_lean(self, math_problem: str) -> str:
        """将数学问题转换为 Lean4 代码
        
        Args:
            math_problem: 数学问题描述
            
        Returns:
            Lean4 形式化代码
        """
        # 1. 准备形式化声明
        formal_statement = self._prepare_formal_statement(math_problem)
        
        # 2. 生成完整代码
        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": self.code_generation_template.format(
                formal_statement=LEAN4_DEFAULT_HEADER + formal_statement
            )}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=self.config.get("openai_model", "gpt-4"),
            messages=messages
        )
        
        lean_code = LEAN4_DEFAULT_HEADER + response.choices[0].message.content
        return lean_code
    
    def verify_lean_code(self, lean_code: str) -> Dict[str, Any]:
        """验证 Lean4 代码
        
        Args:
            lean_code: Lean4 代码
            
        Returns:
            验证结果
        """
        try:
            if self.config.get("lean_feedback", "local") == "local":
                results = parallel_lean4_feedback(
                    lean_codes=[lean_code],
                    num_workers=self.config.get("lean_workers", 1),
                    max_retries=self.config.get("lean_retries", 1),
                    timeout=self.config.get("lean_timeout", 60),
                    memory_limit=self.config.get("lean_memory_limit", 512)
                )
                has_error_status = has_error(results[0], accept_sorry=False)
                return {
                    "is_valid": not has_error_status,
                    "feedback": results[0]
                }
            else:
                result = lean4_client.batch_verify_proof(
                    self.lean_client,
                    [{
                        "uuid": "test",
                        "proof_id": "test",
                        "proof": lean_code
                    }],
                    timeout=self.config.get("lean_timeout", 60),
                    num_threads=self.config.get("lean_workers", 1)
                )[0]
                
                return {
                    "is_valid": result.get("is_valid_no_sorry", False),
                    "feedback": result.get("lean_feedback", "")
                }
        except Exception as e:
            return {
                "is_valid": False,
                "feedback": f"验证过程出错: {str(e)}"
            }
    
    def gpt_review(self, math_problem: str, lean_code: str, verification_result: Dict[str, Any]) -> str:
        """GPT 审查结果
        
        Args:
            math_problem: 原始数学问题
            lean_code: 生成的 Lean4 代码
            verification_result: 验证结果
            
        Returns:
            审查意见
        """
        messages = [
            {"role": "system", "content": gpt_verification_system_prompt},
            {"role": "user", "content": f"""请审查以下数学问题的形式化结果：

原始问题：
{math_problem}

生成的 Lean4 代码：
{lean_code}

验证结果：
{json.dumps(verification_result, ensure_ascii=False, indent=2)}

请详细分析：
1. 形式化是否准确捕捉了原始问题的数学含义
2. 是否正确处理了类型转换（特别是 Nat、Int、Real 之间的转换）
3. 是否存在语法或逻辑错误
4. 如果有错误，具体应该如何修改"""}
        ]
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.get("openai_model", "gpt-4"),
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"审查过程出错: {str(e)}"
    
    def process(self, math_problem: str, max_attempts: int = 3) -> Dict[str, Any]:
        """处理完整的数学问题形式化流程
        
        Args:
            math_problem: 数学问题
            max_attempts: 最大尝试次数
            
        Returns:
            处理结果
        """
        attempts = []
        best_result = None
        
        for attempt in range(max_attempts):
            # 1. 转换为 Lean4 代码
            lean_code = self.math_to_lean(math_problem)
            
            # 2. 验证代码
            verification_result = self.verify_lean_code(lean_code)
            
            # 3. GPT 审查
            review = self.gpt_review(math_problem, lean_code, verification_result)
            
            current_result = {
                "attempt": attempt + 1,
                "lean_code": lean_code,
                "verification_result": verification_result,
                "review": review
            }
            
            attempts.append(current_result)
            
            # 如果验证通过，就不需要继续尝试
            if verification_result["is_valid"]:
                best_result = current_result
                break
                
            # 如果这是最后一次尝试，选择验证结果最好的版本
            if attempt == max_attempts - 1:
                best_result = min(attempts, 
                    key=lambda x: len(str(x["verification_result"].get("feedback", ""))))
        
        return {
            "original_problem": math_problem,
            "best_result": best_result,
            "all_attempts": attempts
        } 