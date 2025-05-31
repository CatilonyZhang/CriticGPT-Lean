import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from math_tree.pipeline import MathPipeline

def main():
    # 配置 OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # 初始化 Pipeline
    config_path = project_root / "config" / "math_pipeline_config.yaml"
    pipeline = MathPipeline(config_path)
    
    # 测试数学问题
    math_problems = [
        """
        证明：对于任意实数 a 和 b，如果 a + b = 1 且 a² + b² = 1，那么 a = b = 1/2。
        """,
        """
        证明：如果一个正整数 n 是完全平方数，那么它的所有正因数的个数是奇数。
        """,
        """
        设 f(x) 是定义在实数集上的连续函数，对任意实数 x 都有 f(f(x)) = x。
        证明：f 是严格单调的，且 f(x) = f⁻¹(x)。
        """
    ]
    
    # 处理每个问题
    for i, problem in enumerate(math_problems, 1):
        print(f"\n处理问题 {i}:")
        print("=" * 50)
        print(problem.strip())
        print("-" * 50)
        
        # 处理问题
        result = pipeline.process(problem)
        
        # 打印最佳结果
        best_result = result["best_result"]
        print(f"\n最佳结果 (第 {best_result['attempt']} 次尝试):")
        
        print("\n精炼后的问题：")
        print(best_result["coding_result"]["refined_problem"])
        
        print("\nBlockTree 结构：")
        print(best_result["coding_result"]["block_tree"])
        
        print("\nLean4 代码：")
        print(best_result["coding_result"]["lean_code"])
        
        print("\n编译结果：")
        print(json.dumps(best_result["compilation_result"], ensure_ascii=False, indent=2))
        
        print("\nLLM 评估：")
        print(best_result["evaluation_result"]["evaluation"])

if __name__ == "__main__":
    main() 