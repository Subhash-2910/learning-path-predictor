"""Milestone 2: the course language model."""
import torch

from app.catalog import IDS
from app.model import BOS, EOS, PAD, CourseGPT, ModelConfig, Vocab, batchify, load, save


def test_vocab_has_one_token_per_course_and_round_trips():
    vocab = Vocab()
    assert len(vocab) == 3 + len(IDS)
    ids = vocab.encode(["CS101", "MA101"])
    assert ids[0] == BOS and ids[-1] == EOS
    assert vocab.decode(ids) == ["CS101", "MA101"]


def test_batchify_pads_on_the_right_and_truncates():
    batch = batchify([[1, 5, 2], [1, 6, 7, 8, 2]], max_len=4)
    assert batch.tolist() == [[1, 5, 2, PAD], [1, 6, 7, 8]]


def test_the_model_is_causal():
    """Changing a later course must not change the prediction at an earlier position."""
    torch.manual_seed(0)
    model = CourseGPT(ModelConfig(vocab_size=len(Vocab()), d_model=16, dropout=0.0)).eval()
    a = torch.tensor([[BOS, 3, 4, 5, 6]])
    b = torch.tensor([[BOS, 3, 4, 9, 10]])
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert la.shape == (1, 5, len(Vocab()))
    assert torch.allclose(la[:, :3], lb[:, :3], atol=1e-6)
    assert not torch.allclose(la[:, 3:], lb[:, 3:])


def test_attention_weights_can_be_kept_and_match_the_fast_path():
    torch.manual_seed(0)
    model = CourseGPT(ModelConfig(vocab_size=len(Vocab()), d_model=16, dropout=0.0)).eval()
    ids = torch.tensor([[BOS, 3, 4, 5]])
    with torch.no_grad():
        fast, slow = model(ids), model(ids, keep_attention=True)
    assert torch.allclose(fast, slow, atol=1e-5)
    weights = model.blocks[0].last_attention
    assert weights.shape == (1, 4, 4, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 4, 4))
    assert float(weights[0, 0, 0, 1:].sum()) == 0.0  # the first position sees only itself


def test_training_lowers_the_validation_loss(tiny_model):
    _, _, history = tiny_model
    assert history["best_val_loss"] < 0.7 * history["start_val_loss"]
    assert history["epochs"][-1]["train_loss"] < history["epochs"][0]["train_loss"]


def test_save_and_load_give_the_same_predictions(tiny_model, tmp_path):
    model, vocab, _ = tiny_model
    save(model, vocab, tmp_path / "m.pt")
    loaded, loaded_vocab = load(tmp_path / "m.pt")
    ids = torch.tensor([vocab.encode(["CS101", "CS102"], eos=False)])
    with torch.no_grad():
        assert torch.allclose(model(ids), loaded(ids))
    assert loaded_vocab.tokens == vocab.tokens
