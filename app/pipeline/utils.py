from __future__ import annotations

import gc
import logging
import shutil
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")

def retry(
    exceptions: tuple[type[Exception], ...] = (Exception,),
    tries: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Retry decorator with exponential backoff.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

def cleanup_runtime(stage: str | None = None) -> None:
    if stage:
        logger.info("==> [cleanup] stage: %s", stage)

    try:
        gc.collect()
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning("Error during GPU cleanup: %s", e)

def get_disk_usage(path: str | Path) -> dict[str, float]:
    total, used, free = shutil.disk_usage(path)
    return {
        "total_gb": total / (1024**3),
        "used_gb": used / (1024**3),
        "free_gb": free / (1024**3),
    }

def check_disk_space(path: str | Path, min_free_gb: float = 5.0) -> bool:
    usage = get_disk_usage(path)
    free_gb = usage["free_gb"]
    if free_gb < min_free_gb:
        logger.warning("Low disk space on %s: %.2f GB free, required %.2f GB", path, free_gb, min_free_gb)
        return False
    return True

def get_gpu_memory() -> list[dict[str, float]]:
    if not torch.cuda.is_available():
        return []

    memos = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(i)
        allocated = torch.cuda.memory_allocated(i)
        free = total - allocated # This is not strictly true as some might be used by other processes

        # More accurate free memory using torch.cuda.mem_get_info
        try:
            free_info, total_info = torch.cuda.mem_get_info(i)
            free = free_info
        except Exception:
            pass

        memos.append({
            "device": i,
            "total_gb": total / (1024**3),
            "free_gb": free / (1024**3),
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
        })
    return memos

def check_gpu_memory(min_free_gb: float = 2.0) -> bool:
    memos = get_gpu_memory()
    if not memos:
        return True # Assume OK if no GPU (though training will likely fail later)

    for memo in memos:
        if memo["free_gb"] < min_free_gb:
            logger.warning("Low GPU memory on device %d: %.2f GB free, required %.2f GB", memo["device"], memo["free_gb"], min_free_gb)
            return False
    return True
