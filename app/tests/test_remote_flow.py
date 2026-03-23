import json
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Mock heavy dependencies
sys.modules["unsloth"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["trl"] = MagicMock()

from app import runner
from app.bootstrap.schemas import JobConfig

@pytest.fixture
def config_dict():
    return {
        "job_id": "job-test-001",
        "job_name": "test-job",
        "mode": "remote",
        "model": {
            "source": "local",
            "local_path": "/app",
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        },
        "dataset": {
            "source": "url",
            "train_url": "http://example.com/train.json",
            "val_url": "http://example.com/val.json",
        },
        "training": {
            "method": "lora",
        },
        "lora": {
            "target_modules": ["q_proj", "v_proj"],
        },
        "outputs": {
            "base_dir": "test_outputs",
        },
        "postprocess": {
            "merge_lora": True,
        },
        "upload": {
            "enabled": True,
            "target": "url",
            "upload_url": "http://example.com/upload",
            "url_targets": {
                "summary_url": "http://example.com/upload/summary",
                "logs_url": "http://example.com/upload/logs",
                "merged_archive_url": "http://example.com/upload/merged",
            },
        },
        "report_url": "http://example.com/report",
    }

@patch("app.runner.resolve_config_list")
@patch("app.runner.Reporter")
@patch("app.runner.AssetManager")
@patch("app.runner.UploadRunner")
@patch("app.runner.PublishRunner")
@patch("app.pipeline.train_runner.run_training")
@patch("app.runner.try_hf_login")
@patch("app.runner.setup_logging")
@patch("pathlib.Path.open", new_callable=mock_open)
@patch("app.runner.write_json")
def test_full_pipeline_success(
    mock_write_json,
    mock_path_open,
    mock_setup_logging,
    mock_hf_login,
    mock_run_training,
    mock_publish_runner_cls,
    mock_upload_runner_cls,
    mock_asset_manager_cls,
    mock_reporter_cls,
    mock_resolve_config_list,
    config_dict
):
    cfg = JobConfig.model_validate(config_dict)
    mock_resolve_config_list.return_value = ([cfg], "http://example.com/config.json")
    mock_setup_logging.return_value = (Path("test_outputs/logs/trainer.log"), [])

    mock_run_training.return_value = {
        "status": "success",
        "job_name": cfg.job_name,
        "lora_dir": "test_outputs/lora/test-job",
        "merged_dir": "test_outputs/merged/test-job",
    }

    mock_upload_runner = mock_upload_runner_cls.return_value
    mock_upload_runner.upload_non_summary_artifacts.return_value = (
        {"logs": {"url": "http://example.com/upload/logs", "path": "trainer.log"}},
        {},
    )
    mock_upload_runner.upload_summary.return_value = {
        "summary": {"url": "http://example.com/upload/summary", "path": "job-result.json"}
    }

    mock_publish_runner = mock_publish_runner_cls.return_value

    with patch("sys.argv", ["runner.py", "--config", "http://example.com/config.json"]):
        runner.main()

    reporter = mock_reporter_cls.return_value
    reporter.report_status.assert_any_call(
        "started",
        message="Training pipeline started",
        stage="bootstrap",
        progress=0,
    )
    reporter.report_status.assert_any_call(
        "finished",
        message="Training pipeline finished successfully",
        stage="finished",
        progress=100,
    )
    reporter.report_final.assert_called_once()

    mock_asset_manager_cls.return_value.prepare_dataset.assert_called_once()
    mock_upload_runner.upload_non_summary_artifacts.assert_called_once()
    mock_upload_runner.upload_summary.assert_called_once()
