import os
import sys
from pathlib import Path

import pytest

# Keep the footprint small and deterministic on CI runners.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def engine():
    """One model load for the whole session; loading gpt2 dominates test time."""
    import torch

    torch.set_num_threads(1)
    from server.model import LLMEngine

    return LLMEngine()
