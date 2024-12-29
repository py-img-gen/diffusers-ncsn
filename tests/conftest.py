import pathlib

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed() -> int:
    return 19950815


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


@pytest.fixture
def project_dir(root_dir: pathlib.Path) -> pathlib.Path:
    dirpath = root_dir / "outputs"
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


@pytest.fixture
def lib_dir(root_dir: pathlib.Path) -> pathlib.Path:
    return root_dir / "src" / "ncsn"
