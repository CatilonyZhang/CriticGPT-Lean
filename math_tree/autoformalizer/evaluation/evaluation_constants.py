from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from vllm import SamplingParams
import json

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

@dataclass
class VLLMConfig:
    """Configuration for VLLM model initialization"""
    tensor_parallel_size: int = 1
    swap_space: int = 0
    gpu_memory_utilization: float = 0.95
    dtype: str = "auto"

@dataclass
class SamplingConfig:
    """Configuration for text generation sampling"""
    n_samples: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 1.05  # defaults to 0, set 1.05 to supress the repeat output problem in coder-base model 
    stop: Optional[Union[str, List[str]]] = None

@dataclass
class ProofVerificationResult:
    """Structured result from proof verification"""
    problem_id: str
    proof: str
    is_valid: bool
    output: str
    sample_id: Optional[int] = -1
    lean_feedback: Optional[Dict[str, Any]] = ''
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))

@dataclass
class GenerationTask:
    problem_ids: List[str]
    prompts: List[str]
    sampling_params: SamplingParams

@dataclass
class GenerationResult:
    problem_id: str
    sample_id: int
    output: str

@dataclass
class ProofVerificationTask:
    problem_ids: List[str]
    sample_ids: List[int]
    proofs: List[str]
