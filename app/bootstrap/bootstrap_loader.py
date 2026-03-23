from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import requests

from .schemas import JobConfig
from ..pipeline.utils import retry

logger = logging.getLogger(__name__)


@retry(tries=3, delay=2, backoff=2, logger=logger)
def load_remote_job_config(url: str) -> Tuple[JobConfig, Dict[str, Any]]:
    logger.info("==> loading remote config from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Remote config must be a JSON object")

    # Some backends wrap the config in a 'config' or 'job' field
    config_data = payload.get("config") or payload.get("job") or payload
    meta = payload.get("meta") or {}

    # Extract callback/logging info from meta or top-level payload if present
    # This logic allows backend to override/provide endpoints dynamically
    resolved_meta = {
        "logs_url": meta.get("logs_url") or payload.get("logs_url"),
        "status_url": meta.get("status_url") or payload.get("status_url"),
        "progress_url": meta.get("progress_url") or payload.get("progress_url"),
        "final_url": meta.get("final_url") or payload.get("final_url"),
        "callback_auth_token": (
            meta.get("callback_auth_token")
            or payload.get("callback_auth_token")
            or meta.get("bearer_token")
            or payload.get("bearer_token")
        ),
    }

    # Merge resolved_meta into meta
    meta.update({k: v for k, v in resolved_meta.items() if v})

    return JobConfig.model_validate(config_data), meta
