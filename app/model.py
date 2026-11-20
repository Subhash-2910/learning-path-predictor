"""Milestone 2: pretrain a small GPT-style language model on course-ID sequences.

The "language" here has one word per course. A student history becomes a sentence:

    <bos> CS101 MA101 ST101 DS201 <eos>

and the model learns ordinary next-token prediction on it, exactly like a text model learns the next word.
It is small on purpose (about 110k parameters: 2 layers, 4 heads, 64 dimensions) so it trains on a laptop CPU
in about a minute and the tests can train one in seconds.

Training uses the usual language-model recipe: AdamW, linear warm-up then cosine decay of the learning rate,
dropout, loss ignored on padding, and the checkpoint with the lowest validation loss is the one kept.

    python -m app.model      # trains on dataset/train.jsonl -> model/course_gpt.pt, results/training.json
"""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.catalog import IDS

MODEL_PATH = Path("model/course_gpt.pt")
SPECIALS = ["<pad>", "<bos>", "<eos>"]
PAD, BOS, EOS = 0, 1, 2


class Vocab:
    """Token ids: 0-2 are the special tokens, then one id per course in catalogue order."""

    def __init__(self, course_ids=IDS):
        self.tokens = SPECIALS + list(course_ids)
        self.index = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def encode(self, courses, eos: bool = True) -> list[int]:
        return [BOS] + [self.index[c] for c in courses] + ([EOS] if eos else [])

    def decode(self, ids) -> list[str]:
        return [self.tokens[i] for i in ids if i >= len(SPECIALS)]

    @property
    def course_token_ids(self) -> list[int]:
        return list(range(len(SPECIALS), len(self.tokens)))


@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int = 24
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1


class Block(nn.Module):
    """Pre-norm transformer block: causal self-attention, then a two-layer MLP, both with residuals."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout))
        self.drop = nn.Dropout(cfg.dropout)
        self.last_attention = None  # filled when forward(..., keep_attention=True)

    def forward(self, x, keep_attention: bool = False):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        if keep_attention:  # the slow, explicit way, so the weights can be looked at (milestone 5)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d // self.n_heads)
            scores = scores.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1),
                                        float("-inf"))
            weights = scores.softmax(-1)
            self.last_attention = weights.detach()
            y = weights @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(b, t, d)))
        return x + self.mlp(self.ln2(x))


class CourseGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied: one vector per course, used for both input and output
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, keep_attention: bool = False):
        t = ids.shape[1]
        x = self.drop(self.tok(ids) + self.pos(torch.arange(t, device=ids.device)))
        for block in self.blocks:
            x = block(x, keep_attention)
        return self.head(self.ln(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    seed: int = 0


def batchify(sequences: list[list[int]], max_len: int) -> torch.Tensor:
    """Right-pad token lists into one tensor; anything longer than `max_len` keeps its first tokens."""
    width = min(max(len(s) for s in sequences), max_len)
    out = torch.full((len(sequences), width), PAD, dtype=torch.long)
    for i, s in enumerate(sequences):
        s = s[:width]
        out[i, :len(s)] = torch.tensor(s)
    return out


def lm_loss(model: CourseGPT, batch: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy; positions whose target is padding do not count."""
    logits = model(batch[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), ignore_index=PAD)


@torch.no_grad()
def evaluate_loss(model: CourseGPT, sequences, batch_size: int = 512) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(sequences), batch_size):
        batch = batchify(sequences[i:i + batch_size], model.cfg.max_len)
        n = int((batch[:, 1:] != PAD).sum())
        total += float(lm_loss(model, batch)) * n
        count += n
    return total / count


def train(train_courses: list[list[str]], val_courses: list[list[str]], cfg: TrainConfig = TrainConfig(),
          model_cfg: ModelConfig | None = None, log=print) -> tuple[CourseGPT, Vocab, dict]:
    torch.manual_seed(cfg.seed)
    vocab = Vocab()
    model = CourseGPT(model_cfg or ModelConfig(vocab_size=len(vocab)))
    train_ids = [vocab.encode(c) for c in train_courses]
    val_ids = [vocab.encode(c) for c in val_courses]

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    steps_per_epoch = math.ceil(len(train_ids) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch

    def lr_at(step):  # linear warm-up, then cosine down to 10% of the peak
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    generator = torch.Generator().manual_seed(cfg.seed)
    history = {"epochs": [], "start_val_loss": evaluate_loss(model, val_ids)}
    best, best_state, step, started = float("inf"), None, 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(train_ids), generator=generator).tolist()
        running = 0.0
        for i in range(0, len(order), cfg.batch_size):
            batch = batchify([train_ids[j] for j in order[i:i + cfg.batch_size]], model.cfg.max_len)
            loss = lm_loss(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            step += 1
        val = evaluate_loss(model, val_ids)
        history["epochs"].append({"epoch": epoch, "train_loss": round(running / steps_per_epoch, 4),
                                  "val_loss": round(val, 4)})
        log(f"epoch {epoch:2d}  train {running / steps_per_epoch:.4f}  val {val:.4f}")
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
    model.load_state_dict(best_state)
    model.eval()
    history.update({"best_val_loss": round(best, 4), "best_val_perplexity": round(math.exp(best), 3),
                    "steps": step, "seconds": round(time.time() - started, 1), "parameters": model.n_params(),
                    "train_config": asdict(cfg), "model_config": asdict(model.cfg)})
    return model, vocab, history


def save(model: CourseGPT, vocab: Vocab, path: Path = MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "tokens": vocab.tokens, "state_dict": model.state_dict()}, path)


def load(path: Path = MODEL_PATH) -> tuple[CourseGPT, Vocab]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    vocab = Vocab(blob["tokens"][len(SPECIALS):])
    model = CourseGPT(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, vocab


if __name__ == "__main__":
    from app.data import load_split

    train_c = [s["courses"] for s in load_split("train")]
    val_c = [s["courses"] for s in load_split("val")]
    model, vocab, history = train(train_c, val_c)
    save(model, vocab)
    Path("results").mkdir(exist_ok=True)
    Path("results/training.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"{history['parameters']:,} parameters, best val loss {history['best_val_loss']} "
          f"(perplexity {history['best_val_perplexity']}) at epoch {history['best_epoch']}, "
          f"{history['seconds']} s -> {MODEL_PATH}")
