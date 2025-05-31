import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class ModelParams:
    model_path: List[str]
    url: List[str]
    n_retry: int = 0
    temperature: float = 1.0
    max_length: int = 2048
    prompt_type: str = "text"
    tokenizer_path: str = (
        "/mnt/moonfs/wanghaiming-m2/models/deepseekprover/DeepSeek-Prover-V1.5-RL"
    )


@dataclass
class VerifierParams:
    url: str
    timeout: int = 60


@dataclass
class SearchParams:
    search_method: str = "bfs"
    num_sampled_tactics: int = 64
    max_expansions: int = 100
    search_timeout: int = 36000
    step_timeout: Optional[int] = None
    serialize_interval: int = 30
    resume_from_checkpoint: bool = False


@dataclass
class DatasetConfig:
    name: str
    split: str = field(default="train")
    select_range: Optional[int] = None  # Optional, can be added if needed


@dataclass
class Config:
    # Experiment configurations
    platform: str = "moonshot"
    job_id: str = "initial_step_v1"
    max_workers: int = 10
    verbose: bool = False

    # Dataset settings
    datasets: List[DatasetConfig] = field(default_factory=list)
    pass_at: int = 4

    # Sharding and Filtering
    shard_size: Optional[int] = None
    filter_threshold: Optional[int] = None
    add_negation: Optional[bool] = False

    # Model arguments
    model_params: ModelParams = field(default_factory=ModelParams)

    # Verifier parameters
    verifier_params: VerifierParams = field(default_factory=VerifierParams)

    # Search parameters
    search_params: SearchParams = field(default_factory=SearchParams)

    # Directories
    working_dir: str = "/mnt/moonfs/wanghaiming-m2/jobs/{job_id}"
    huggingface_cache_dir: str = "/mnt/moonfs/wanghaiming-m2/.cache/ttmmpp"

    # Debugging
    logging_dir: str = "{working_dir}/log/proof_search/"
    statement_dir: str = "{working_dir}/statements"
    verified_proofs_dir: str = "{working_dir}/verified_proofs"
    # proof_record_dir: str = "{working_dir}/proof_records.jsonl"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Load configuration from a YAML file and instantiate the Config class.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Returns:
            Config: An instance of the Config class populated with YAML data.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Replace placeholders
        job_id = config_dict.get("job_id", cls.job_id)
        working_dir = config_dict.get("working_dir", cls.working_dir).format(
            job_id=job_id
        )

        # Format other directories based on working_dir
        logging_dir = config_dict.get("logging_dir", cls.logging_dir).format(
            working_dir=working_dir
        )
        statement_dir = config_dict.get("statement_dir", cls.statement_dir).format(
            working_dir=working_dir
        )
        verified_proofs_dir = config_dict.get(
            "verified_proofs_dir", cls.verified_proofs_dir
        ).format(working_dir=working_dir)
        # proof_record_dir = config_dict.get('proof_record_dir', cls.proof_record_dir).format(working_dir=working_dir)

        # Create necessary directories
        pathlib.Path(working_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(statement_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(verified_proofs_dir).mkdir(parents=True, exist_ok=True)

        # Handle nested model_params
        model_params_dict = config_dict.get("model_params", {})
        model_params = ModelParams(**model_params_dict)

        # Handle nested verifier_params
        verifier_params_dict = config_dict.get("verifier_params", {})
        verifier_params = VerifierParams(**verifier_params_dict)

        # Handle nested search_params
        search_params_dict = config_dict.get("search_params", {})
        search_params = SearchParams(**search_params_dict)

        # Handle datasets list
        datasets_list = config_dict.get("datasets", [])
        datasets = [DatasetConfig(**ds) for ds in datasets_list]

        # Initialize Config
        config = cls(
            platform=config_dict.get("platform", cls.platform),
            job_id=job_id,
            max_workers=config_dict.get("max_workers", cls.max_workers),
            verbose=config_dict.get("verbose", cls.verbose),
            datasets=datasets,
            pass_at=config_dict.get("pass_at", cls.pass_at),
            shard_size=config_dict.get("shard_size", cls.shard_size),
            filter_threshold=config_dict.get("filter_threshold", cls.filter_threshold),
            add_negation=config_dict.get("add_negation", cls.add_negation),
            model_params=model_params,
            verifier_params=verifier_params,
            search_params=search_params,
            working_dir=working_dir,
            huggingface_cache_dir=config_dict.get(
                "huggingface_cache_dir", cls.huggingface_cache_dir
            ),
            logging_dir=logging_dir,
            statement_dir=statement_dir,
            verified_proofs_dir=verified_proofs_dir,
            # proof_record_dir=proof_record_dir,
        )

        return config
