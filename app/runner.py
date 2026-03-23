from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple

import torch

from .adapters.hf_utils import try_hf_login
from .adapters.log_streamer import LogStreamer
from .adapters.reporter import Reporter
from .bootstrap.bootstrap_loader import load_remote_job_config
from .bootstrap.config_loader import load_config, load_config_bundle
from .pipeline.asset_manager import AssetManager
from .pipeline.publish_runner import PublishRunner
from .pipeline.upload_runner import UploadRunner
from .pipeline.utils import cleanup_runtime

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def tail_file(file_path: Path, lines: int = 50) -> str:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return "".join(f.readlines()[-lines:])
    except Exception:
        return "Could not retrieve logs"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def setup_logging(
    logs_dir: str,
    job_id: Optional[str] = None,
    job_name: Optional[str] = None,
    logs_url: Optional[str] = None,
    logs_bearer_token: Optional[str] = None,
) -> Tuple[Path, List[logging.Handler]]:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "trainer.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    handlers: List[logging.Handler] = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]

    if logs_url and job_id and job_name:
        streamer = LogStreamer(
            logs_url=logs_url,
            job_id=job_id,
            job_name=job_name,
            bearer_token=logs_bearer_token,
        )
        streamer.setFormatter(formatter)
        handlers.append(streamer)

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )
    return log_file, handlers


def teardown_logging(handlers: List[logging.Handler]) -> None:
    root = logging.getLogger()
    for handler in handlers:
        try:
            root.removeHandler(handler)
        except Exception:
            pass
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass


def start_heartbeat(reporter: Reporter, interval: int = 30) -> Tuple[Event, Thread]:
    stop_event = Event()

    def heartbeat_loop():
        while not stop_event.wait(interval):
            try:
                reporter.report_status(
                    "running",
                    message="Heartbeat",
                    stage="heartbeat",
                    extra={"heartbeat": True},
                )
            except Exception:
                pass

    thread = Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    return stop_event, thread


def stop_heartbeat(stop_event: Optional[Event], thread: Optional[Thread]) -> None:
    if stop_event:
        stop_event.set()
    if thread:
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass


def apply_run_output_paths(cfg) -> Tuple[str, Path]:
    run_name = f"{cfg.job_name}_{utc_compact_timestamp()}"
    base_root = Path(cfg.outputs.base_dir)
    run_root = base_root / run_name

    cfg.outputs.base_dir = str(run_root)
    cfg.outputs.logs_dir = str(run_root / "logs")
    cfg.outputs.lora_dir = str(run_root / "lora")
    cfg.outputs.checkpoints_dir = str(run_root / "checkpoints")
    cfg.outputs.metrics_dir = str(run_root / "metrics")
    cfg.outputs.merged_dir = str(run_root / "merged")
    cfg.outputs.quantized_dir = str(run_root / "quantized")
    cfg.outputs.eval_dir = str(run_root / "evaluation")
    cfg.outputs.downloads_dir = str(run_root / "downloads")

    return run_name, run_root


class Pipeline:
    def __init__(self, cfg: Any, config_source: str, bootstrap_meta: Optional[Dict[str, Any]] = None):
        self.cfg = copy.deepcopy(cfg)
        self.config_source = config_source
        self.bootstrap_meta = bootstrap_meta or {}
        self.run_name, self.run_root = apply_run_output_paths(self.cfg)

        if self.bootstrap_meta.get("status_url"):
            self.cfg.reporting.status.url = self.bootstrap_meta["status_url"]
            self.cfg.reporting.status.enabled = True
        if self.bootstrap_meta.get("progress_url"):
            self.cfg.reporting.progress.url = self.bootstrap_meta["progress_url"]
            self.cfg.reporting.progress.enabled = True
        if self.bootstrap_meta.get("final_url"):
            self.cfg.reporting.final.url = self.bootstrap_meta["final_url"]
            self.cfg.reporting.final.enabled = True
        if self.bootstrap_meta.get("logs_url"):
            self.cfg.reporting.logs.url = self.bootstrap_meta["logs_url"]
            self.cfg.reporting.logs.enabled = True

        auth_token = self.bootstrap_meta.get("callback_auth_token")
        if auth_token:
            for cb in [self.cfg.reporting.status, self.cfg.reporting.progress, self.cfg.reporting.final, self.cfg.reporting.logs]:
                if not cb.auth.bearer_token:
                    cb.auth.bearer_token = auth_token

        self.log_file, self.log_handlers = setup_logging(
            self.cfg.outputs.logs_dir,
            job_id=self.cfg.job_id or self.cfg.job_name,
            job_name=self.cfg.job_name,
            logs_url=self.cfg.reporting.logs.url if self.cfg.reporting.logs.enabled else None,
            logs_bearer_token=self.cfg.reporting.logs.auth.bearer_token,
        )

        self.reporter = Reporter(self.cfg)
        self.asset_manager = AssetManager(self.cfg)
        self.uploader = UploadRunner(self.cfg)
        self.publisher = PublishRunner(self.cfg)

        self.heartbeat_stop, self.heartbeat_thread = start_heartbeat(
            self.reporter,
            interval=self.cfg.reporting.heartbeat_interval
        )

        self.effective_config_path = Path(self.cfg.outputs.logs_dir) / "effective-job.json"
        self.result_path = Path(self.cfg.outputs.base_dir) / "job-result.json"

        self.started_at = utc_now_iso()
        self.training_result: Dict[str, Any] = {}
        self.evaluation_result: Optional[Dict[str, Any]] = None
        self.result: Dict[str, Any] = {
            "status": "started",
            "job_id": self.cfg.job_id or self.cfg.job_name,
            "job_name": self.cfg.job_name,
            "run_name": self.run_name,
            "started_at": self.started_at,
            "finished_at": None,
            "config_source": self.config_source,
            "training": {},
            "evaluation": None,
            "artifacts": {
                "run_root": str(self.run_root),
                "log_file": str(self.log_file),
                "effective_config_path": str(self.effective_config_path),
                "result_path": str(self.result_path),
            },
            "uploads": {},
            "upload_errors": {},
            "externalRefs": [],
        }

    def run(self) -> Dict[str, Any]:
        try:
            self.reporter.report_status(
                "started",
                message="Training pipeline started",
                stage="bootstrap",
                progress=0,
            )

            write_json(self.effective_config_path, json.loads(self.cfg.model_dump_json()))
            logger.info("==> config loaded")
            logger.info("==> job_name: %s", self.cfg.job_name)
            logger.info("==> run_name: %s", self.run_name)
            logger.info("==> job_id: %s", self.cfg.job_id or self.cfg.job_name)
            logger.info("==> config source: %s", self.config_source)
            logger.info("==> run output dir: %s", self.run_root)

            self.reporter.report_status(
                "running",
                message="Validating Hugging Face access",
                stage="hf_login",
                progress=2,
            )
            try_hf_login()
            self.publisher.ensure_hf_ready()

            pipeline_cfg = self.cfg.pipeline
            has_steps = bool(pipeline_cfg and pipeline_cfg.steps)

            if has_steps:
                logger.info("==> executing pipeline steps")
                for step in pipeline_cfg.steps:
                    if step.enabled:
                        self._execute_step(step.key, step.kind)
                    else:
                        logger.info("==> step %s disabled, skipping", step.key)
            else:
                logger.warning("==> pipeline.steps missing, falling back to legacy sequence")
                self._run_legacy_sequence()

            self._finalize_success()
            return self.result

        except Exception as exc:
            return self._handle_error(exc)
        finally:
            self._cleanup()

    def _execute_step(self, step_key: str, step_kind: str) -> None:
        logger.info("==> running step: %s (kind: %s)", step_key, step_kind)

        if step_kind == "prepare_assets":
            self.reporter.report_status("running", message="Preparing assets", stage="prepare_assets", progress=5)
            self.asset_manager.prepare_dataset(self.cfg)
            self.asset_manager.prepare_evaluation_dataset(self.cfg)
            cleanup_runtime("prepare-assets")

        elif step_kind == "training":
            from .pipeline.train_runner import run_training
            self.training_result = run_training(self.cfg, reporter=self.reporter)
            if self.cfg.model.logical_base_model_id and not self.training_result.get("base_model_id"):
                self.training_result["base_model_id"] = self.cfg.model.logical_base_model_id
            self.result["training"] = self.training_result
            cleanup_runtime("training")

        elif step_kind == "evaluation":
            from .pipeline.eval_runner import run_evaluation
            cleanup_runtime("pre-evaluation")
            try:
                self.evaluation_result = run_evaluation(self.cfg, self.training_result, reporter=self.reporter)
                self.result["evaluation"] = self.evaluation_result
            except Exception as e:
                logger.error("Evaluation failed but continuing: %s", e)
                self.evaluation_result = {
                    "enabled": True,
                    "status": "failed",
                    "error": str(e),
                }
                self.result["evaluation"] = self.evaluation_result
                self.result["evaluation_error"] = str(e)
            cleanup_runtime("evaluation")

        elif step_kind == "publish_hf":
            self.reporter.report_status("running", message="Publishing to HF", stage="publish", progress=90)
            try:
                hf_model_uploads = self.publisher.upload_to_huggingface(self.training_result)
                if hf_model_uploads:
                    self.result["uploads"].update(hf_model_uploads)

                hf_metadata_uploads = self.publisher.upload_hf_metadata(
                    log_file=str(self.log_file),
                    effective_config_path=str(self.effective_config_path),
                    result_path=str(self.result_path),
                    training_result=self.training_result,
                    eval_result=self.evaluation_result,
                )
                if hf_metadata_uploads:
                    self.result["uploads"].update(hf_metadata_uploads)
            except Exception as e:
                logger.error("Publishing to HF failed but continuing: %s", e)
                self.result["upload_errors"]["huggingface"] = str(e)
            cleanup_runtime("publish")

        elif step_kind == "upload_artifacts":
            self.reporter.report_status("running", message="Uploading artifacts", stage="upload", progress=95)
            try:
                extra_uploads, extra_upload_errors = self.uploader.upload_non_summary_artifacts(
                    log_file=str(self.log_file),
                    effective_config_path=str(self.effective_config_path),
                    training_result=self.training_result,
                    eval_result=self.evaluation_result,
                )
                if extra_uploads:
                    self.result["uploads"].update(extra_uploads)
                if extra_upload_errors:
                    self.result["upload_errors"].update(extra_upload_errors)
            except Exception as e:
                logger.error("Uploading artifacts failed but continuing: %s", e)
                self.result["upload_errors"]["artifacts"] = str(e)
            cleanup_runtime("upload")

    def _run_legacy_sequence(self) -> None:
        p = self.cfg.pipeline
        if not p or p.prepare_assets.enabled:
            self._execute_step("prepare_assets", "prepare_assets")
        if not p or p.training.enabled:
            self._execute_step("training", "training")
        if (p.evaluation.enabled if p else self.cfg.evaluation.enabled):
            self._execute_step("evaluation", "evaluation")
        if (p.publish.enabled if p else self.cfg.huggingface.enabled):
            self._execute_step("publish", "publish_hf")
        if (p.upload.enabled if p else self.cfg.upload.enabled):
            self._execute_step("upload", "upload_artifacts")

    def _finalize_success(self) -> None:
        self.result["finished_at"] = utc_now_iso()

        evaluation_failed = bool(self.result.get("evaluation_error")) or (
            isinstance(self.result.get("evaluation"), dict)
            and self.result["evaluation"].get("status") == "failed"
        )

        self.result["status"] = "partial_failed" if evaluation_failed else "success"
        write_json(self.result_path, self.result)

        p = self.cfg.pipeline
        should_upload = p.upload.enabled if p else self.cfg.upload.enabled
        if should_upload and self.cfg.upload.target == "url" and self.cfg.upload.url_targets.summary_url:
            try:
                summary_upload = self.uploader.upload_summary(str(self.result_path))
                if summary_upload:
                    self.result["uploads"].update(summary_upload)
                    write_json(self.result_path, self.result)
            except Exception as exc:
                logger.exception("summary upload failed")
                self.result["upload_errors"]["summary"] = str(exc)
                write_json(self.result_path, self.result)

        if evaluation_failed or self.result["upload_errors"]:
            logger.warning("==> pipeline finished with warnings")
        else:
            logger.info("==> pipeline finished successfully")

        final_message = (
            "Training pipeline finished with evaluation errors"
            if evaluation_failed
            else "Training pipeline finished successfully"
        )

        status_extra = {"evaluation_error": self.result.get("evaluation_error")} if evaluation_failed else {}

        self.reporter.report_status(
            "finished",
            message=final_message,
            stage="finished",
            progress=100,
            extra=status_extra,
        )
        self.reporter.report_final(self.result, status=self.result["status"])

    def _handle_error(self, exc: Exception) -> Dict[str, Any]:
        error_msg = str(exc)
        stack_trace = traceback.format_exc()

        logger.error("FATAL ERROR: %s", error_msg)
        logger.error(stack_trace)

        self.result["status"] = "failed"
        self.result["finished_at"] = utc_now_iso()
        self.result["error"] = error_msg
        write_json(self.result_path, self.result)

        self.reporter.report_error(error_msg, logs=tail_file(self.log_file, 50))
        self.reporter.report_final(self.result, status="failed")
        return self.result

    def _cleanup(self) -> None:
        stop_heartbeat(self.heartbeat_stop, self.heartbeat_thread)
        try:
            self.reporter.close()
        except Exception:
            pass
        teardown_logging(self.log_handlers)
        cleanup_runtime("job-finalize")


def resolve_single_remote_config(args) -> Tuple[Any, str, Dict[str, Any]]:
    if args.job_config_url:
        cfg, meta = load_remote_job_config(args.job_config_url)
        return cfg, args.job_config_url, meta

    if args.config:
        cfg = load_config(args.config)
        return cfg, args.config, {}

    raise ValueError("Either --config or --job-config-url must be provided")


def resolve_config_list(args) -> Tuple[List[Any], str]:
    if args.job_config_url:
        cfg, _, _ = resolve_single_remote_config(args)
        return [cfg], args.job_config_url

    if args.config:
        cfgs = load_config_bundle(args.config)
        return cfgs, args.config

    raise ValueError("Either --config or --job-config-url must be provided")


def run_single_job(cfg, config_source: str, bootstrap_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cleanup_runtime("previous-job")
    pipeline = Pipeline(cfg, config_source, bootstrap_meta)
    return pipeline.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--job-config-url", required=False)
    args = parser.parse_args()

    try:
        if args.job_config_url:
            cfg, config_source, bootstrap_meta = resolve_single_remote_config(args)
            results = [run_single_job(cfg, config_source, bootstrap_meta)]
            summary_root = Path(cfg.outputs.base_dir).parent if Path(cfg.outputs.base_dir).name else Path("/output")
        else:
            cfgs, config_source = resolve_config_list(args)
            if not cfgs:
                raise ValueError("No jobs found in config bundle")

            summary_root = Path(cfgs[0].outputs.base_dir)
            results = []
            for index, cfg in enumerate(cfgs, start=1):
                cleanup_runtime(f"before-batch-job-{index}")
                print(f"==> batch job {index}/{len(cfgs)}: {cfg.job_name}")
                results.append(run_single_job(cfg, config_source, {}))
                cleanup_runtime(f"after-batch-job-{index}")
    except Exception as exc:
        print(f"ERROR: failed to execute jobs: {exc}")
        sys.exit(1)

    batch_finished_at = utc_now_iso()
    batch_started_at = min((r.get("started_at") for r in results if r.get("started_at")), default=batch_finished_at)
    success_count = sum(1 for r in results if r.get("status") == "success")
    failed_count = sum(1 for r in results if r.get("status") == "failed")
    partial_failed_count = sum(1 for r in results if r.get("status") == "partial_failed")

    batch_summary = {
        "status": "success" if failed_count == 0 and partial_failed_count == 0 else "partial_failed",
        "started_at": batch_started_at,
        "finished_at": batch_finished_at,
        "total_jobs": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "partial_failed_count": partial_failed_count,
        "results": results,
    }

    summary_root.mkdir(parents=True, exist_ok=True)
    batch_summary_path = summary_root / f"batch-summary_{utc_compact_timestamp()}.json"
    write_json(batch_summary_path, batch_summary)

    print(json.dumps(batch_summary, indent=2, ensure_ascii=False))

    if failed_count > 0 or partial_failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()