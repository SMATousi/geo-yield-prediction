import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def single_threaded_torch():
    """Keep the small CPU contract tests fast and deterministic in CI."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)
