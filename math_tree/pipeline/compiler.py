import os
from typing import Dict, Any
from math_tree.autoformalizer.autoformalizer.clients import lean4_client
from math_tree.autoformalizer.autoformalizer.eval_utils.lean_feedback import has_error, parallel_lean4_feedback

class LeanCompiler:
    """Lean4 编译器
    
    支持：
    1. 本地编译
    2. 远程服务器编译
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get("lean_feedback", "local")
        
        if self.mode == "server":
            self.client = lean4_client.Lean4Client(
                url=config.get("lean_server_url", "https://kimina.saas.moonshot.cn/lean4-evaluator"),
                api_key=os.environ.get("MOONSHOT_LEAN4_API_KEY")
            )
    
    def compile(self, lean_code: str) -> Dict[str, Any]:
        """编译 Lean4 代码"""
        if self.mode == "local":
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
            result = self.client.one_pass_verify(lean_code, timeout=self.config.get("lean_timeout", 60))
            return {
                "is_valid": not has_error(result.get("response", {}), accept_sorry=False),
                "feedback": result
            } 