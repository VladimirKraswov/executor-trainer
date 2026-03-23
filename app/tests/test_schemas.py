import pytest
import sys
from unittest.mock import MagicMock

# Mock heavy/missing dependencies
sys.modules["torch"] = MagicMock()
sys.modules["unsloth"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["trl"] = MagicMock()

from app.bootstrap.schemas import JobConfig, ModelConfig, DatasetConfig, TrainingConfig, LoraConfig, OutputsConfig

def test_minimal_config():
    config_dict = {
        "job_name": "test-job",
        "model": {
            "source": "huggingface",
            "repo_id": "Qwen/Qwen2.5-7B-Instruct"
        },
        "dataset": {
            "source": "local",
            "train_path": "/data/train.json"
        },
        "training": {
            "method": "lora"
        },
        "lora": {
            "target_modules": ["q_proj", "v_proj"]
        },
        "outputs": {
            "base_dir": "/output"
        }
    }
    config = JobConfig.model_validate(config_dict)
    assert config.job_name == "test-job"
    assert config.model.repo_id == "Qwen/Qwen2.5-7B-Instruct"

def test_invalid_model_config():
    with pytest.raises(ValueError):
        ModelConfig(source="local", local_path=None)

def test_invalid_dataset_config():
    with pytest.raises(ValueError):
        DatasetConfig(source="url", train_url=None)

def test_outputs_fill_defaults():
    outputs = OutputsConfig(base_dir="/output")
    assert outputs.logs_dir == "/output/logs"
    assert outputs.lora_dir == "/output/lora"

def test_logical_base_model_id():
    m = ModelConfig(source="huggingface", repo_id="Qwen/Qwen2.5-7B-Instruct")
    assert m.logical_base_model_id == "Qwen/Qwen2.5-7B-Instruct"

    m = ModelConfig(source="local", local_path="/app/model", base_model="Qwen/Qwen2.5-7B-Instruct")
    assert m.logical_base_model_id == "Qwen/Qwen2.5-7B-Instruct"

    m = ModelConfig(source="local", local_path="/app/model")
    assert m.logical_base_model_id is None
