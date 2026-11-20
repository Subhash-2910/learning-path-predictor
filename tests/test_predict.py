"""Milestone 3: predicting the next course."""
import numpy as np
import pytest

from app.catalog import IDS
from app.predict import CoOccurrence, LastCourse, Popularity, Recommender, evaluate, shared_last_courses


@pytest.fixture(scope="module")
def rec(tiny_model):
    model, vocab, _ = tiny_model
    return Recommender(model, vocab)


def test_predict_next_returns_k_courses_best_first_without_completed_ones(rec):
    top = rec.predict_next(["CS101", "MA101"], k=5)
    assert len(top) == 5
    ids, probs = [c for c, _ in top], [p for _, p in top]
    assert not {"CS101", "MA101"} & set(ids)
    assert probs == sorted(probs, reverse=True)
    assert all(0 < p <= 1 for p in probs)


def test_probabilities_cover_every_remaining_course_and_sum_to_one(rec):
    probs = rec.next_course_probs(["CS101"], exclude=["EL101"])
    assert set(probs) == set(IDS) - {"CS101", "EL101"}
    assert sum(probs.values()) == pytest.approx(1.0)


def test_not_interested_courses_are_never_recommended(rec):
    first = rec.predict_next(["CS101"], k=1)[0][0]
    assert first not in [c for c, _ in rec.predict_next(["CS101"], k=10, exclude=[first])]


def test_empty_history_and_unknown_courses(rec):
    assert len(rec.predict_next([], k=3)) == 3
    with pytest.raises(ValueError, match="XX999"):
        rec.predict_next(["CS101", "XX999"])


def test_one_forward_pass_matches_prefix_by_prefix_prediction(rec):
    seq = ["CS101", "MA101", "ST101", "DS201"]
    rows = rec.sequence_scores(seq)
    for t in range(len(seq)):
        assert np.allclose(rows[t], rec.course_scores(seq[:t]), atol=1e-5)


def test_the_model_learns_the_planted_paths(rec):
    """Students who started with WD101 take WD102 next on the full-stack path."""
    assert rec.predict_next(["WD101"], k=1)[0][0] == "WD102"


def test_baselines_on_a_toy_history():
    train = [["CS101", "CS102", "CS201"], ["CS101", "CS102", "CS203"], ["CS101", "MA101"]]
    pos = {c: i for i, c in enumerate(IDS)}
    assert np.argmax(Popularity(train).course_scores([])) == pos["CS101"]
    assert np.argmax(LastCourse(train).course_scores(["CS101"])) == pos["CS102"]
    co = CoOccurrence(train).course_scores(["CS101", "CS102"])
    assert co[pos["CS201"]] == co[pos["CS203"]] > co[pos["MA101"]]


def test_shared_last_courses_include_the_branching_points():
    shared = shared_last_courses()
    assert {"CS101", "DS201"} <= shared
    assert "DS303" not in shared  # the end of a path is followed by nothing


def test_the_model_beats_popularity_on_held_out_students(rec, small_data):
    train = [s["courses"] for s in small_data["train"]]
    test = [s["courses"] for s in small_data["test"]]
    m = evaluate(rec, train, test)["methods"]
    assert m["CourseGPT (this project)"]["all"]["mrr"] > m["popularity"]["all"]["mrr"] + 0.2
