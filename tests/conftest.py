import pytest

from cup_model.demo import generate
from cup_model.store import Dataset, Store


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    generate(path, days=30, seed=7)
    store = Store(":memory:")
    store.import_jsonl(path)
    data = Dataset(store.records())
    store.close()
    return data
