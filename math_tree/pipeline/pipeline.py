import os
import yaml
import json
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

from .coder import MathCoder
from .compiler import LeanCompiler
from .evaluator import LLMEvaluator

class MathPipeline:
    """数学问题自动形式化流水线
    
    集成：
    1. MathCoder - 编码器
    2. LeanCompiler - 编译器
    3. LLMEvaluator - 评估器
    """
    
    def __init__(self, config_path: str):
        """初始化 Pipeline
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.coder = MathCoder(self.config)
        self.compiler = LeanCompiler(self.config)
        self.evaluator = LLMEvaluator(self.config)
        
        # 创建输出目录
        self.output_dir = Path(self.config.get("output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML configuration: {str(e)}")
    
    def process(self, problem: str, max_attempts: int = 3) -> Dict[str, Any]:
        """处理数学问题
        
        Args:
            problem: 数学问题
            max_attempts: 最大尝试次数
            
        Returns:
            处理结果
        """
        attempts = []
        best_result = None
        
        for attempt in range(max_attempts):
            # 1. 编码
            coding_result = self.coder.process(problem)
            
            # 2. 编译
            compilation_result = self.compiler.compile(coding_result["lean_code"])
            
            # 3. 评估
            evaluation_result = self.evaluator.evaluate(
                problem,
                coding_result["lean_code"],
                compilation_result
            )
            
            current_result = {
                "attempt": attempt + 1,
                "coding_result": coding_result,
                "compilation_result": compilation_result,
                "evaluation_result": evaluation_result
            }
            
            attempts.append(current_result)
            
            # 如果编译和语义评估都通过，就不需要继续尝试
            if compilation_result["is_valid"] and evaluation_result["is_semantically_correct"]:
                best_result = current_result
                break
            
            # 如果这是最后一次尝试，选择结果最好的版本
            if attempt == max_attempts - 1:
                best_result = max(attempts, 
                    key=lambda x: (x["compilation_result"]["is_valid"], 
                                 x["evaluation_result"]["is_semantically_correct"]))
        
        result = {
            "original_problem": problem,
            "best_result": best_result,
            "all_attempts": attempts
        }
        
        # 保存结果
        self._save_result(result)
        
        return result
    
    def _save_result(self, result: Dict[str, Any]):
        """保存处理结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"result_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2) 