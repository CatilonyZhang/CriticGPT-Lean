import os
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from math_autocoder import MathAutoCoder

def save_result(result: dict, output_dir: Path):
    """保存结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"result_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    # 配置 OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # 创建输出目录
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    # 初始化 MathAutoCoder
    config_path = project_root / "config" / "math_autocoder_config.yaml"
    autocoder = MathAutoCoder(config_path)
    
    # 测试数学问题
    math_problems = [
        """
        证明：对于任意实数 a 和 b，如果 a + b = 1 且 a² + b² = 1，那么 a = b = 1/2。
        """,
        """
        证明：如果一个正整数 n 是完全平方数，那么它的所有正因数的个数是奇数。
        """
    ]
    
    # 处理每个问题
    for i, problem in enumerate(math_problems, 1):
        print(f"\n处理问题 {i}:")
        print("=" * 50)
        print(problem.strip())
        print("-" * 50)
        
        # 处理问题
        result = autocoder.process(problem)
        
        # 打印最佳结果
        best_result = result["best_result"]
        print(f"\n最佳结果 (第 {best_result['attempt']} 次尝试):")
        print("\nLean4 代码：")
        print(best_result["lean_code"])
        print("\n验证结果：")
        print(json.dumps(best_result["verification_result"], ensure_ascii=False, indent=2))
        print("\nGPT 审查意见：")
        print(best_result["review"])
        
        # 保存结果
        save_result(result, output_dir)

if __name__ == "__main__":
    main() 