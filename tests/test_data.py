"""Milestone 1: the synthetic student histories."""
import numpy as np

from app.catalog import BUNDLES, COURSES, GOALS, IDS
from app.data import describe, generate_dataset, generate_student, load_split, save_dataset, valid_order


def test_every_planted_path_respects_its_own_prerequisites():
    for goal, path in GOALS.items():
        assert valid_order(path), goal


def test_valid_order_rejects_a_course_before_its_prerequisite_and_repeats():
    assert valid_order(["CS101", "CS102"])
    assert not valid_order(["CS102", "CS101"])
    assert not valid_order(["CS101", "CS101"])


def test_students_are_valid_sequences_of_known_course_ids():
    rng = np.random.default_rng(1)
    for _ in range(500):
        student = generate_student(rng)
        assert student["goal"] in GOALS
        assert len(student["courses"]) >= 3
        assert set(student["courses"]) <= set(IDS)
        assert valid_order(student["courses"])


def test_generation_is_reproducible_and_split_80_10_10():
    a, b = generate_dataset(200, seed=3), generate_dataset(200, seed=3)
    assert a == b
    assert [len(a[s]) for s in ("train", "val", "test")] == [160, 20, 20]
    assert generate_dataset(200, seed=4) != a


def test_the_planted_bundles_show_up_in_the_data():
    train = generate_dataset(2000, seed=0)["train"]
    for first, second, p in BUNDLES:
        with_first = [s["courses"] for s in train if first in s["courses"]]
        followed = sum(seq[seq.index(first) + 1:seq.index(first) + 2] == [second] for seq in with_first)
        assert followed / len(with_first) > p - 0.15


def test_save_and_load_round_trip(tmp_path):
    data = generate_dataset(50, seed=0)
    save_dataset(data, tmp_path)
    assert load_split("test", tmp_path) == data["test"]
    stats = describe(data)
    assert stats["courses_in_catalogue"] == len(COURSES) and stats["min_length"] >= 3
