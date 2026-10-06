import pytest

from nyc311 import ingest
from nyc311.config import Config


@pytest.fixture(scope="session")
def sample_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    return Config.from_args(["--data", "data/sample/311_batch_*.json", "--out", str(root / "out"),
                             "--work", str(root / "work"), "--threads", "4"])


@pytest.fixture(scope="session")
def ingested(sample_cfg):
    return ingest.run(sample_cfg)
