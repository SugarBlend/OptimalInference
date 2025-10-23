import pytest
import collections

def pytest_configure(config: pytest.Config) -> None:
    config._test_results = collections.deque(maxlen=2)
