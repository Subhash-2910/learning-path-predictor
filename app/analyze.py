"""Milestone 5: which course "itemsets" has the model learned?

Three looks inside the trained model:

1. **Itemsets, by generation.** Let the model write thousands of new student histories, mine their frequent
   itemsets with the exact Apriori miner from the market-basket drill (`thilimarketbasket.mine`), and compare
   with the itemsets mined from the real training data. If the model learned which courses go together, the
   two lists match and the supports agree. The planted bundles and paths are checked by name.
2. **Embeddings.** Every course is one 64-number vector (shared by the input and output layers). Courses the
   model treats alike point the same way: nearest neighbours, a 2D map, and same-track vs other-track
   similarity.
3. **Attention.** For one history, which earlier courses the last position looks at when predicting.

    python -m app.analyze   # -> results/itemsets.{json,md}, results/embeddings.png, results/attention.png
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from thilimarketbasket.mine import frequent_itemsets

from app.catalog import BUNDLES, BY_ID, GOALS, IDS, ROOT, TRACKS, prereqs_met
from app.data import valid_order
from app.model import BOS, EOS, PAD
from app.predict import Recommender

RESULTS = ROOT / "results"
MIN_SUPPORT = 0.05


# ----- 1. Itemsets by generation ----------------------------------------------------------------------------

@torch.no_grad()
def sample_histories(rec: Recommender, n: int, seed: int = 0, temperature: float = 1.0) -> list[list[str]]:
    """Let the model write `n` student histories from scratch, one course at a time, until it emits <eos>."""
    gen = torch.Generator().manual_seed(seed)
    model, max_len = rec.model, rec.model.cfg.max_len
    ids = torch.full((n, 1), BOS, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    for _ in range(max_len - 1):
        logits = model(ids)[:, -1] / temperature
        logits[:, [PAD, BOS]] = float("-inf")
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=gen).squeeze(1)
        nxt[done] = PAD
        ids = torch.cat([ids, nxt[:, None]], 1)
        done |= nxt == EOS
        if done.all():
            break
    return [rec.vocab.decode([t for t in row.tolist() if t not in (PAD, BOS, EOS)]) for row in ids]


def popularity_histories(real: list[list[str]], n: int, seed: int = 0) -> list[list[str]]:
    """Baseline with no learned structure: histories of the real lengths, filled with courses drawn by
    popularity alone. Any itemset it gets right is explained by popular courses being frequent."""
    counts = Counter(c for s in real for c in s)
    ids = list(counts)
    p = np.array([counts[i] for i in ids], dtype=float)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    lengths = [len(real[i % len(real)]) for i in range(n)]
    return [[str(c) for c in rng.choice(ids, size=k, replace=False, p=p)] for k in lengths]


def support_of(itemset, baskets) -> float:
    return sum(itemset <= b for b in baskets) / len(baskets)


def compare_itemsets(real: list[list[str]], generated: list[list[str]], min_support: float = MIN_SUPPORT,
                     max_size: int = 4) -> dict:
    """Frequent itemsets (2+ courses) mined from both, and how well the generated ones match the real ones."""
    real_sets, gen_sets = [set(s) for s in real], [set(s) for s in generated]
    found_real = {k: v for k, v in frequent_itemsets(real, min_support, max_size).items() if len(k) > 1}
    found_gen = {k: v for k, v in frequent_itemsets(generated, min_support, max_size).items() if len(k) > 1}
    both = set(found_real) & set(found_gen)
    union = sorted(set(found_real) | set(found_gen), key=lambda k: (-found_real.get(k, 0), len(k), sorted(k)))
    r = np.array([found_real.get(k, support_of(k, real_sets)) for k in union])
    g = np.array([found_gen.get(k, support_of(k, gen_sets)) for k in union])
    by_size = {}
    for size in range(2, max_size + 1):
        rs = {k for k in found_real if len(k) == size}
        gs = {k for k in found_gen if len(k) == size}
        by_size[size] = {"real": len(rs), "generated": len(gs), "both": len(rs & gs)}
    return {
        "min_support": min_support,
        "real_itemsets": len(found_real), "generated_itemsets": len(found_gen), "in_both": len(both),
        "recall": len(both) / max(1, len(found_real)), "precision": len(both) / max(1, len(found_gen)),
        "support_correlation": float(np.corrcoef(r, g)[0, 1]) if len(union) > 2 else float("nan"),
        "mean_abs_support_error": float(np.mean(np.abs(r - g))) if len(union) else float("nan"),
        "by_size": by_size,
        "top": [{"itemset": sorted(k), "real": round(float(rv), 4), "generated": round(float(gv), 4)}
                for k, rv, gv in list(zip(union, r, g))[:20]],
    }


def planted_checks(real: list[list[str]], generated: list[list[str]]) -> dict:
    """The structure we planted on purpose, measured in the real data and in the model's own histories."""
    def follow_rate(seqs, first, second):
        with_first = [s for s in seqs if first in s]
        hits = sum(s[s.index(first) + 1:s.index(first) + 2] == [second] for s in with_first)
        return hits / max(1, len(with_first))

    real_sets, gen_sets = [set(s) for s in real], [set(s) for s in generated]
    bundles = [{"bundle": f"{a} → {b}", "planted": p, "real": round(follow_rate(real, a, b), 3),
                "generated": round(follow_rate(generated, a, b), 3)} for a, b, p in BUNDLES]
    paths = [{"path": goal, "first_four": path[:4],
              "real": round(support_of(set(path[:4]), real_sets), 4),
              "generated": round(support_of(set(path[:4]), gen_sets), 4)} for goal, path in GOALS.items()]
    flat = [c for s in generated for c in s]
    return {
        "bundles": bundles, "paths": paths,
        "generated_valid_order": float(np.mean([valid_order(s) for s in generated])),
        "generated_repeats": float(np.mean([len(set(s)) < len(s) for s in generated])),
        "generated_prereq_violations": float(np.mean(
            [not prereqs_met(c, set(s[:i])) for s in generated for i, c in enumerate(s)])),
        "mean_length": {"real": float(np.mean([len(s) for s in real])),
                        "generated": float(np.mean([len(s) for s in generated]))},
        "course_frequency_correlation": float(np.corrcoef(
            [Counter(c for s in real for c in s)[i] / len(real) for i in IDS],
            [Counter(flat)[i] / len(generated) for i in IDS])[0, 1]),
    }


# ----- 2. Embeddings ----------------------------------------------------------------------------------------

def course_embeddings(rec: Recommender) -> np.ndarray:
    """Unit-length embedding of each course, in catalogue order."""
    w = rec.model.tok.weight.detach()[rec.vocab.course_token_ids].numpy()
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def similar_courses(rec: Recommender, course: str, k: int = 5) -> list[tuple[str, float]]:
    """The `k` courses whose embeddings point most nearly the same way (cosine similarity)."""
    emb = course_embeddings(rec)
    i = rec.courses.index(course)
    sims = emb @ emb[i]
    order = [j for j in np.argsort(-sims) if j != i][:k]
    return [(rec.courses[j], float(sims[j])) for j in order]


def embedding_structure(rec: Recommender) -> dict:
    emb = course_embeddings(rec)
    sims = emb @ emb.T
    tracks = np.array([BY_ID[c].track for c in rec.courses])
    off = ~np.eye(len(tracks), dtype=bool)
    same = (tracks[:, None] == tracks[None, :]) & off
    paths = np.zeros_like(sims, dtype=bool)  # pairs that are neighbours on some planted path
    for path in GOALS.values():
        for a, b in zip(path, path[1:]):
            i, j = rec.courses.index(a), rec.courses.index(b)
            paths[i, j] = paths[j, i] = True
    bundles = []
    for a, b, _ in BUNDLES:
        i, j = rec.courses.index(a), rec.courses.index(b)
        rank = 1 + int((sims[i][off[i]] > sims[i, j]).sum())
        bundles.append({"bundle": f"{a} + {b}", "cosine": round(float(sims[i, j]), 3),
                        "rank_among_neighbours_of_first": rank})
    return {
        "same_track_cosine": float(sims[same].mean()), "other_track_cosine": float(sims[~same & off].mean()),
        "path_neighbours_cosine": float(sims[paths].mean()),
        "bundles": bundles,
        "nearest": {c: [n for n, _ in similar_courses(rec, c, 3)] for c in rec.courses},
    }


def plot_embeddings(rec: Recommender, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb = course_embeddings(rec)
    centred = emb - emb.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    xy = centred @ vt[:2].T
    fig, ax = plt.subplots(figsize=(10, 7.5))
    colours = plt.get_cmap("tab10")
    for t, track in enumerate(TRACKS):
        idx = [i for i, c in enumerate(rec.courses) if BY_ID[c].track == track]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=60, color=colours(t), label=track)
        for i in idx:
            ax.annotate(rec.courses[i], xy[i], xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_title("Course embeddings learned by the model (first two principal components)")
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----- 3. Attention -----------------------------------------------------------------------------------------

@torch.no_grad()
def attention_to_history(rec: Recommender, history: list[str]) -> np.ndarray:
    """Attention of the last position to every earlier token, averaged over heads: one row per layer."""
    ids = torch.tensor([rec.vocab.encode(history, eos=False)])
    rec.model(ids, keep_attention=True)
    return np.stack([b.last_attention[0, :, -1].mean(0).numpy() for b in rec.model.blocks])


def plot_attention(rec: Recommender, history: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = attention_to_history(rec, history)
    top = rec.predict_next(history, 1)[0]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.imshow(weights, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(history) + 1), ["<bos>"] + history)
    ax.set_yticks(range(len(weights)), [f"layer {i + 1}" for i in range(len(weights))])
    for (r, c), w in np.ndenumerate(weights):
        ax.text(c, r, f"{w:.2f}", ha="center", va="center", fontsize=8, color="white" if w > 0.5 else "black")
    ax.set_title(f"Where the model looks when predicting the next course ({top[0]}, {top[1]:.0%})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----- Report -----------------------------------------------------------------------------------------------

def report(itemsets: dict, baseline: dict, planted: dict, embeddings: dict) -> str:
    it, b = itemsets, baseline
    lines = [
        "# What the model learned\n",
        f"8,000 histories written by the model vs the 8,000 real training histories. Frequent itemsets of 2-4 "
        f"courses mined from each with the exact Apriori miner (`thilimarketbasket.mine`) at support ≥ "
        f"{it['min_support']:.0%}.\n",
        "## Itemsets\n",
        "The baseline column fills histories of the same lengths with courses drawn by popularity alone, so "
        "it shows what you get from popular courses with no learned structure.\n",
        "| | model | baseline: popularity only |\n|---|---|---|",
        f"| frequent itemsets in the real data | {it['real_itemsets']} | {b['real_itemsets']} |",
        f"| frequent itemsets in the generated histories | {it['generated_itemsets']} | {b['generated_itemsets']} |",
        f"| found in both | {it['in_both']} | {b['in_both']} |",
        f"| recall (real itemsets reproduced) | {it['recall']:.1%} | {b['recall']:.1%} |",
        f"| precision (generated itemsets that are real) | {it['precision']:.1%} | {b['precision']:.1%} |",
        f"| correlation of supports | {it['support_correlation']:.3f} | {b['support_correlation']:.3f} |",
        f"| mean absolute support error | {it['mean_abs_support_error']:.4f} | "
        f"{b['mean_abs_support_error']:.4f} |\n",
        "| size | real | model | both |\n|---|---|---|---|",
        *[f"| {s} | {v['real']} | {v['generated']} | {v['both']} |" for s, v in it["by_size"].items()],
        "\n**Most frequent real itemsets and their support in the model's histories**\n",
        "| itemset | real support | model support |\n|---|---|---|",
        *[f"| {' + '.join(r['itemset'])} | {r['real']:.3f} | {r['generated']:.3f} |" for r in it["top"][:12]],
        "\n## Planted structure\n",
        "| planted bundle (taken right after) | planted rate | real data | model |\n|---|---|---|---|",
        *[f"| {b['bundle']} | {b['planted']:.0%} | {b['real']:.1%} | {b['generated']:.1%} |"
          for b in planted["bundles"]],
        "\n| planted path (first four courses together) | real support | model support |\n|---|---|---|",
        *[f"| {p['path']}: {' '.join(p['first_four'])} | {p['real']:.3f} | {p['generated']:.3f} |"
          for p in planted["paths"]],
        f"\n- Model-written histories in a valid prerequisite order: **{planted['generated_valid_order']:.1%}** "
        f"(courses taken before their prerequisites: {planted['generated_prereq_violations']:.2%}; histories "
        f"repeating a course: {planted['generated_repeats']:.2%}).",
        f"- Mean history length: real {planted['mean_length']['real']:.2f}, model "
        f"{planted['mean_length']['generated']:.2f}. Correlation of per-course frequency: "
        f"{planted['course_frequency_correlation']:.3f}.\n",
        "## Embeddings\n",
        f"- Mean cosine similarity of course embeddings: same track {embeddings['same_track_cosine']:.3f}, "
        f"different track {embeddings['other_track_cosine']:.3f}, neighbours on a planted path "
        f"{embeddings['path_neighbours_cosine']:.3f}.",
        *[f"- Planted bundle {b['bundle']}: cosine {b['cosine']:.3f}, rank {b['rank_among_neighbours_of_first']}"
          f" of 42 among the first course's neighbours." for b in embeddings["bundles"]],
        "\n![embeddings](embeddings.png)\n",
        "**Nearest neighbours (cosine) for a few courses:** " + "; ".join(
            f"{c} → {', '.join(embeddings['nearest'][c])}" for c in ["CS101", "WD101", "DS301", "MA101", "BU201"]),
        "\n## Attention\n",
        "![attention](attention.png)\n",
        "## Reading the results\n",
        "- **The itemsets live in the model's predictions.** Sampling histories from the model and mining them "
        "recovers almost every frequent itemset of the real data with close supports, including both planted "
        "bundles and the opening of every planted path. The extra itemsets the model invents are mostly "
        "4-course sets just under the support threshold in the real data (the model is a little smoother).",
        "- **Embeddings group courses by role, not by basket.** Courses with similar neighbours on either side "
        "(the entry courses CS101, WD101, BU101, MA101; the electives; the advanced courses of each area) point "
        "the same way. Two courses that are taken *together* (DS301 then EL102) are not similar: one is an "
        "advanced ML course, the other an elective taken at any time. So embedding similarity finds "
        "alternatives to a course; co-taken courses are found by asking the model what comes next.",
        "- **Attention follows prerequisites.** In the example above, after MA101, MA102, CS101, CS102 the model "
        "predicts CS201 (Algorithms) and the strongest attention goes to MA102: the course that tells the "
        "theory path (CS102 → CS201) apart from the backend path (CS102 → CS202), and a prerequisite of CS201.",
    ]
    return "\n".join(lines) + "\n"


ATTENTION_EXAMPLE = ["MA101", "MA102", "CS101", "CS102"]

if __name__ == "__main__":
    from app.data import load_split

    rec = Recommender.from_file()
    real = [s["courses"] for s in load_split("train")]
    generated = sample_histories(rec, len(real), seed=0)
    itemsets = compare_itemsets(real, generated)
    baseline = compare_itemsets(real, popularity_histories(real, len(real), seed=0))
    planted = planted_checks(real, generated)
    embeddings = embedding_structure(rec)
    RESULTS.mkdir(exist_ok=True)
    plot_embeddings(rec, RESULTS / "embeddings.png")
    plot_attention(rec, ATTENTION_EXAMPLE, RESULTS / "attention.png")
    (RESULTS / "itemsets.json").write_text(json.dumps(
        {"itemsets": itemsets, "popularity_baseline": baseline, "planted": planted, "embeddings": embeddings},
        indent=2) + "\n")
    text = report(itemsets, baseline, planted, embeddings)
    (RESULTS / "itemsets.md").write_text(text)
    print(text)
