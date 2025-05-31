from typing import Dict, Any
from openai import OpenAI
from math_tree.llm.prompt.blocktree_prompt import blocktree_prompt
from math_tree.llm.prompt.lean2blocktree import lean2blocktree

class MathCoder:
    """数学问题编码器
    
    负责：
    1. 精炼数学问题
    2. 生成 BlockTree 结构
    3. 转换为 Lean4 代码
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openai_client = OpenAI(
            api_key=config.get("openai_api_key", ""),
            base_url=config.get("openai_base_url", "https://api.openai.com/v1")
        )
        
        # 系统提示词
        self.system_prompt = """You are an expert mathematician and Lean4 programmer.
Your task is to:
1. Refine mathematical problems into clear, formal statements
2. Generate structured block trees for proof organization
3. Convert block trees into syntactically correct Lean4 code

Please ensure:
1. Mathematical precision and rigor
2. Clear logical structure
3. Proper type handling (Nat, Int, Real)
4. Complete imports and declarations"""
        
    def refine_problem(self, problem: str) -> str:
        """精炼数学问题，使其更加形式化和清晰"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""请将以下数学问题转换为更加形式化和清晰的表述：

问题：
{problem}

要求：
1. 明确所有变量的类型和范围
2. 清晰表述问题的条件和目标
3. 使用标准的数学符号
4. 保持数学严谨性"""}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=self.config.get("openai_model", "gpt-4"),
            messages=messages,
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    def generate_blocktree(self, refined_problem: str) -> Dict[str, Any]:
        """生成问题的 BlockTree 结构"""
        messages = [
            {"role": "system", "content": blocktree_prompt},
            {"role": "user", "content": f"请为以下数学问题生成 BlockTree 结构：\n\n{refined_problem}"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=self.config.get("openai_model", "gpt-4"),
            messages=messages,
            temperature=0.2
        )
        
        block_tree = response.choices[0].message.content
        return {
            "problem": refined_problem,
            "block_tree": block_tree
        }
        
    def blocktree_to_lean(self, block_tree: Dict[str, Any]) -> str:
        """将 BlockTree 转换为 Lean4 代码"""
        messages = [
            {"role": "system", "content": lean2blocktree},
            {"role": "user", "content": f"""请将以下 BlockTree 结构转换为 Lean4 代码：

原始问题：
{block_tree['problem']}

BlockTree 结构：
{block_tree['block_tree']}

要求：
1. 生成完整的 Lean4 代码
2. 包含所有必要的导入语句
3. 正确处理类型转换
4. 使用适当的证明策略"""}
        ]
        
        response = self.openai_client.chat.completions.create(
            model=self.config.get("openai_model", "gpt-4"),
            messages=messages,
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    def process(self, problem: str) -> Dict[str, Any]:
        """处理完整的编码流程"""
        # 1. 精炼问题
        refined_problem = self.refine_problem(problem)
        
        # 2. 生成 BlockTree
        block_tree = self.generate_blocktree(refined_problem)
        
        # 3. 转换为 Lean4 代码
        lean_code = self.blocktree_to_lean(block_tree)
        
        return {
            "original_problem": problem,
            "refined_problem": refined_problem,
            "block_tree": block_tree,
            "lean_code": lean_code
        } 