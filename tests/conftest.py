import pytest

from app.data import generate_dataset
from app.model import ModelConfig, TrainConfig, Vocab, train


@pytest.fixture(scope="session")
def small_data():
    return generate_dataset(1500, seed=7)


@pytest.fixture(scope="session")
def tiny_model(small_data):
    """A small model trained for a few seconds, so tests never need the committed checkpoint."""
    courses = lambda split: [s["courses"] for s in small_data[split]]
    model, vocab, history = train(courses("train"), courses("val"), TrainConfig(epochs=12, batch_size=64),
                                  ModelConfig(vocab_size=len(Vocab()), d_model=32, dropout=0.0),
                                  log=lambda *_: None)
    return model, vocab, history
