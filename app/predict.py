"""Milestone 3: predict the next course from a partial sequence, and measure it against simple baselines.

`Recommender.predict_next(["CS101", "MA101"], k=5)` runs the history through the model once and reads the
probability of every course as the next token. Courses already completed (and any the student marked
"not interested") are removed and the rest renormalised, so the probabilities shown add up to 1 over the
courses that could actually be recommended.

    python -m app.predict    # evaluates on dataset/test.jsonl -> results/metrics.{json,md}, results/samples.md

The evaluation hides each next course of every test student in turn ("given the first t courses, what is
course t+1?") and records where the true course lands in each method's ranking (hits@K, MRR).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from thilimarketbasket.evaluate import rank_of, summarise

from app.catalog import BY_ID, GOALS, IDS, ROOT, prereqs_met
from app.model import BOS, MODEL_PATH, CourseGPT, Vocab, load

RESULTS = ROOT / "results"
KS = (1, 3, 5)


class Recommender:
    def __init__(self, model: CourseGPT, vocab: Vocab):
        self.model, self.vocab = model.eval(), vocab
        self.courses = vocab.tokens[3:]

    @classmethod
    def from_file(cls, path: Path = MODEL_PATH) -> "Recommender":
        return cls(*load(path))

    def _check(self, completed):
        unknown = [c for c in completed if c not in self.vocab.index]
        if unknown:
            raise ValueError(f"unknown course id(s): {', '.join(unknown)}")

    @torch.no_grad()
    def course_scores(self, completed) -> np.ndarray:
        """Log-probability of each course (catalogue order) being next, before anything is excluded."""
        self._check(completed)
        ids = self.vocab.encode(completed, eos=False)
        ids = [BOS] + ids[1:][-(self.model.cfg.max_len - 1):]  # very long histories keep their latest courses
        logits = self.model(torch.tensor([ids]))[0, -1]
        return torch.log_softmax(logits, -1)[self.vocab.course_token_ids].numpy()

    def next_course_probs(self, completed, exclude=()) -> dict[str, float]:
        """Probability of every course that may still be recommended, renormalised to sum to 1."""
        scores = self.course_scores(completed)
        blocked = set(completed) | set(exclude)
        keep = np.array([c not in blocked for c in self.courses])
        if not keep.any():
            return {}
        p = np.exp(scores - scores[keep].max()) * keep
        p /= p.sum()
        return {c: float(p[i]) for i, c in enumerate(self.courses) if keep[i]}

    def predict_next(self, completed, k: int = 5, exclude=()) -> list[tuple[str, float]]:
        """The `k` most likely next courses as (course id, probability), best first."""
        probs = self.next_course_probs(completed, exclude)
        return sorted(probs.items(), key=lambda kv: -kv[1])[:k]

    def recommend(self, completed, k: int = 5, exclude=(), only_eligible: bool = False) -> list[dict]:
        """Rows for the app: the top `k` courses with title, track, probability and missing prerequisites.
        With `only_eligible`, courses whose prerequisites are not all completed are left out before
        renormalising."""
        exclude = set(exclude)
        if only_eligible:
            exclude |= {c for c in self.courses if not prereqs_met(c, set(completed))}
        rows = []
        for course, p in self.predict_next(completed, k, exclude):
            missing = [q for q in BY_ID[course].prereqs if q not in completed]
            rows.append({"course": course, "title": BY_ID[course].title, "track": BY_ID[course].track,
                         "probability": p, "missing_prereqs": missing})
        return rows

    @torch.no_grad()
    def sequence_scores(self, courses) -> np.ndarray:
        """One forward pass gives the prediction after every prefix: row t scores course t+1 given courses[:t]."""
        ids = self.vocab.encode(courses, eos=False)[: self.model.cfg.max_len]
        logits = self.model(torch.tensor([ids]))[0]
        return torch.log_softmax(logits, -1)[:, self.vocab.course_token_ids].numpy()


# ----- Baselines: counting, no neural network ---------------------------------------------------------------

class Popularity:
    """Recommend the most taken courses, whatever the student has done."""
    name = "popularity"

    def __init__(self, train_sequences):
        counts = Counter(c for seq in train_sequences for c in seq)
        self.scores = np.log(np.array([counts[c] + 1.0 for c in IDS]))

    def course_scores(self, completed):
        return self.scores


class LastCourse:
    """First-order Markov chain: what most often came right after the student's last course."""
    name = "last course (Markov)"

    def __init__(self, train_sequences):
        self.pop = Popularity(train_sequences).scores
        self.after = defaultdict(Counter)
        for seq in train_sequences:
            for a, b in zip(seq, seq[1:]):
                self.after[a][b] += 1

    def course_scores(self, completed):
        if not completed:
            return self.pop
        counts = self.after[completed[-1]]
        return np.log(np.array([counts[c] for c in IDS]) + 1e-3 * np.exp(self.pop) / np.exp(self.pop).sum())


class CoOccurrence:
    """Count, over the whole history, how often each course came after each completed course."""
    name = "co-occurrence (whole history)"

    def __init__(self, train_sequences):
        self.pop = Popularity(train_sequences).scores
        index = {c: i for i, c in enumerate(IDS)}
        self.later = np.zeros((len(IDS), len(IDS)))
        for seq in train_sequences:
            for i, a in enumerate(seq):
                for b in seq[i + 1:]:
                    self.later[index[a], index[b]] += 1
        self.index = index

    def course_scores(self, completed):
        if not completed:
            return self.pop
        total = sum(self.later[self.index[c]] for c in completed)
        return np.log(total + 1e-3 * np.exp(self.pop) / np.exp(self.pop).sum())


# ----- Evaluation -------------------------------------------------------------------------------------------

def shared_last_courses() -> set:
    """Courses that are followed by different courses on different planted paths: the last course alone
    cannot tell which way the student is going."""
    nexts = defaultdict(set)
    for path in GOALS.values():
        for a, b in zip(path, path[1:]):
            nexts[a].add(b)
    return {a for a, followers in nexts.items() if len(followers) > 1}


def cases(sequences):
    """(student number, t, prefix, target) for every next-course prediction in the test set."""
    for n, seq in enumerate(sequences):
        for t in range(1, len(seq)):
            yield n, t, seq[:t], seq[t]


def evaluate(recommender: Recommender, train_sequences, test_sequences) -> dict:
    position = {c: i for i, c in enumerate(IDS)}
    baselines = [Popularity(train_sequences), LastCourse(train_sequences), CoOccurrence(train_sequences)]
    names = [b.name for b in baselines] + ["CourseGPT (this project)"]
    ranks = {name: [] for name in names}
    top1_valid = {name: [] for name in names}
    ambiguous = []
    shared = shared_last_courses()
    model_rows = [recommender.sequence_scores(seq) for seq in test_sequences]
    for n, t, prefix, target in cases(test_sequences):
        exclude = [position[c] for c in prefix]
        ambiguous.append(prefix[-1] in shared)
        for name, scores in zip(names, [b.course_scores(prefix) for b in baselines] + [model_rows[n][t]]):
            ranks[name].append(rank_of(scores, position[target], exclude))
            masked = scores.copy()
            masked[exclude] = -np.inf
            top1_valid[name].append(prereqs_met(IDS[int(np.argmax(masked))], set(prefix)))
    ambiguous = np.array(ambiguous)
    out = {"cases": len(ambiguous), "ambiguous_cases": int(ambiguous.sum()), "methods": {}}
    for name in names:
        r = np.array(ranks[name])
        out["methods"][name] = {
            "all": summarise(r, KS),
            "last_course_is_shared": summarise(r[ambiguous], KS),
            "top1_prereqs_met": float(np.mean(top1_valid[name])),
        }
    return out


def metrics_table(metrics: dict) -> str:
    head = ("| method | hits@1 | hits@3 | hits@5 | MRR | hits@1 when last course is shared | "
            "top-1 has prereqs met |\n|---|---|---|---|---|---|---|\n")
    rows = []
    for name, m in metrics["methods"].items():
        a, s = m["all"], m["last_course_is_shared"]
        bold = "**" if "CourseGPT" in name else ""
        rows.append(f"| {bold}{name}{bold} | {a['hits@1']:.3f} | {a['hits@3']:.3f} | {a['hits@5']:.3f} | "
                    f"{a['mrr']:.3f} | {s['hits@1']:.3f} | {m['top1_prereqs_met']:.1%} |")
    return head + "\n".join(rows) + "\n"


SAMPLE_HISTORIES = [
    ["CS101"],
    ["WD101", "CS101"],
    ["CS101", "CS102"],
    ["MA101", "MA102", "CS101", "CS102"],
    ["CS101", "MA101", "ST101", "DS201"],
    ["CS101", "ST101", "DS201"],
    ["CS101", "MA101", "ST101", "DS201", "MA201", "MA202", "DS301"],
    ["BU101", "EN101", "WD101", "UX201", "BU201"],
    ["EL103", "EL104"],
]


def samples_markdown(recommender: Recommender, k: int = 5) -> str:
    lines = ["# Sample recommendations\n", "Top 5 next courses from the trained model for hand-picked histories. "
             "Probabilities are renormalised over courses not yet completed.\n"]
    for history in SAMPLE_HISTORIES:
        lines.append(f"\n**Completed:** {' → '.join(history)}\n")
        lines.append("| # | course | title | probability |\n|---|---|---|---|")
        for i, (c, p) in enumerate(recommender.predict_next(history, k), 1):
            lines.append(f"| {i} | {c} | {BY_ID[c].title} | {p:.1%} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    from app.data import load_split

    rec = Recommender.from_file()
    train_seq = [s["courses"] for s in load_split("train")]
    test_seq = [s["courses"] for s in load_split("test")]
    metrics = evaluate(rec, train_seq, test_seq)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    table = metrics_table(metrics)
    (RESULTS / "metrics.md").write_text(
        f"# Next-course prediction on {len(test_seq)} held-out students ({metrics['cases']} predictions)\n\n"
        f"Each test student's history is cut after every course; the method ranks all courses not yet taken and "
        f"we record the rank of the course the student really took next. 'Last course is shared' is the "
        f"{metrics['ambiguous_cases']} cases where the last course leads to different next courses on different "
        f"planted paths, so the earlier history matters.\n\n" + table)
    (RESULTS / "samples.md").write_text(samples_markdown(rec))
    print(table)
