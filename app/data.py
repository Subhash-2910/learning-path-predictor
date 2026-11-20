"""Milestone 1: mock student histories, one ordered sequence of course IDs per student.

Like mock retail transactions, but ordered: a "basket" is everything a student completed, in the order they
completed it. Each student mostly follows one planted learning path (catalog.GOALS), stops after a random
number of courses, and adds noise on the way: electives, a swapped pair of courses, a planted bundle, or a
detour into a second path. Prerequisites are always respected.

    python -m app.data      # writes dataset/{train,val,test}.jsonl and dataset/stats.json
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.catalog import BUNDLES, BY_ID, COURSES, GOAL_WEIGHTS, GOALS, prereqs_met

DATASET_DIR = Path("dataset")

P_ELECTIVE = 0.12   # after each course, chance of squeezing in an unrelated course
P_SWAP = 0.15       # chance per student of swapping two neighbouring courses (when prerequisites allow it)
P_DETOUR = 0.2      # chance of taking a few courses from a second path at the end
MIN_LENGTH = 3


def valid_order(sequence) -> bool:
    """True when no course repeats and every course comes after all of its prerequisites."""
    taken = set()
    for course in sequence:
        if course in taken or not prereqs_met(course, taken):
            return False
        taken.add(course)
    return True


def _random_extra(rng, taken) -> str | None:
    """Any course the student may take now; electives are four times as likely as the others."""
    options = [c.id for c in COURSES if c.id not in taken and prereqs_met(c.id, taken)]
    if not options:
        return None
    weights = np.array([4.0 if BY_ID[o].track == "Electives" else 1.0 for o in options])
    return options[rng.choice(len(options), p=weights / weights.sum())]


def generate_student(rng: np.random.Generator) -> dict:
    goals = list(GOALS)
    weights = np.array([GOAL_WEIGHTS[g] for g in goals], dtype=float)
    goal = goals[rng.choice(len(goals), p=weights / weights.sum())]
    path = GOALS[goal]
    length = int(rng.integers(MIN_LENGTH, len(path) + 1))

    sequence, taken = [], set()

    def take(course):
        if course and course not in taken and prereqs_met(course, taken):
            sequence.append(course)
            taken.add(course)

    for course in path[:length]:
        take(course)
        for first, second, p in BUNDLES:
            if course == first and rng.random() < p:
                take(second)
        if rng.random() < P_ELECTIVE:
            take(_random_extra(rng, taken))

    if rng.random() < P_DETOUR:
        other = goals[rng.integers(len(goals))]
        for course in [c for c in GOALS[other] if c not in taken][: int(rng.integers(1, 4))]:
            take(course)

    if rng.random() < P_SWAP and len(sequence) > 2:
        i = int(rng.integers(len(sequence) - 1))
        swapped = sequence[:i] + [sequence[i + 1], sequence[i]] + sequence[i + 2:]
        if valid_order(swapped):
            sequence = swapped

    return {"goal": goal, "courses": sequence}


def generate_dataset(n_students: int = 10_000, seed: int = 0) -> dict:
    """80/10/10 train/validation/test split of `n_students` synthetic students."""
    rng = np.random.default_rng(seed)
    students = [generate_student(rng) for _ in range(n_students)]
    n_train, n_val = int(0.8 * n_students), int(0.1 * n_students)
    return {"train": students[:n_train], "val": students[n_train:n_train + n_val],
            "test": students[n_train + n_val:]}


def save_dataset(data: dict, directory: Path = DATASET_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for split, students in data.items():
        with open(directory / f"{split}.jsonl", "w") as f:
            for s in students:
                f.write(json.dumps(s) + "\n")
    (directory / "stats.json").write_text(json.dumps(describe(data), indent=2) + "\n")


def load_split(split: str, directory: Path = DATASET_DIR) -> list[dict]:
    with open(Path(directory) / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def describe(data: dict) -> dict:
    train = [s["courses"] for s in data["train"]]
    counts = Counter(c for seq in train for c in seq)
    lengths = [len(seq) for seq in train]
    return {
        "students": {split: len(v) for split, v in data.items()},
        "courses_in_catalogue": len(COURSES),
        "mean_length": round(float(np.mean(lengths)), 2),
        "min_length": int(min(lengths)), "max_length": int(max(lengths)),
        "goals": dict(Counter(s["goal"] for s in data["train"]).most_common()),
        "most_taken": dict(counts.most_common(10)),
        "never_taken": sorted(c.id for c in COURSES if counts[c.id] == 0),
    }


if __name__ == "__main__":
    data = generate_dataset()
    save_dataset(data)
    print(json.dumps(describe(data), indent=2))
    for s in data["train"][:5]:
        print(f"{s['goal']:>13}: {' '.join(s['courses'])}")
