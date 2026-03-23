import pytest
from app.pipeline.utils import retry
from unittest.mock import MagicMock

def test_retry_success():
    mock_func = MagicMock(return_value="success")
    decorated = retry(tries=3, delay=0.1)(mock_func)
    result = decorated()
    assert result == "success"
    assert mock_func.call_count == 1

def test_retry_eventual_success():
    mock_func = MagicMock(side_effect=[ValueError("fail"), "success"])
    decorated = retry(exceptions=(ValueError,), tries=3, delay=0.1)(mock_func)
    result = decorated()
    assert result == "success"
    assert mock_func.call_count == 2

def test_retry_failure():
    mock_func = MagicMock(side_effect=ValueError("fail"))
    decorated = retry(exceptions=(ValueError,), tries=3, delay=0.1)(mock_func)
    with pytest.raises(ValueError):
        decorated()
    assert mock_func.call_count == 3
