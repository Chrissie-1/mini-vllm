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


# Every assertion here is about engine mechanics -- cache surgery, batch
# invariance, stopping rules -- none of which depend on the weights being any
# good. TEST_MODEL swaps in a tiny randomly-initialised GPT-2 so the suite runs
# on a constrained machine and in CI without a 500 MB download.
TEST_MODEL = os.environ.get("TEST_MODEL", "gpt2")


@pytest.fixture(scope="session")
def engine():
    """One model load for the whole session; loading weights dominates test time."""
    import torch

    torch.set_num_threads(1)
    from server.model import LLMEngine

    return LLMEngine(model_name=TEST_MODEL)
