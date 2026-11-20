"""Milestone 5: what the model learned (itemsets, embeddings, attention)."""
import numpy as np
import pytest

from app.analyze import (attention_to_history, compare_itemsets, embedding_structure, plot_attention,
                         plot_embeddings, planted_checks, popularity_histories, sample_histories,
                         similar_courses)
from app.catalog import IDS
from app.predict import Recommender


@pytest.fixture(scope="module")
def rec(tiny_model):
    return Recommender(*tiny_model[:2])


@pytest.fixture(scope="module")
def generated(rec):
    return sample_histories(rec, 400, seed=1)


def test_sampled_histories_are_course_sequences_and_reproducible(rec, generated):
    assert len(generated) == 400
    assert all(set(s) <= set(IDS) for s in generated)
    assert np.mean([len(s) for s in generated]) > 2
    assert sample_histories(rec, 50, seed=1) == sample_histories(rec, 50, seed=1)


def test_identical_data_gives_perfect_itemset_agreement(small_data):
    real = [s["courses"] for s in small_data["train"]]
    same = compare_itemsets(real, real, min_support=0.05)
    assert same["real_itemsets"] > 10
    assert same["recall"] == same["precision"] == 1.0
    assert same["support_correlation"] == pytest.approx(1.0)
    assert same["mean_abs_support_error"] == 0.0


def test_the_model_reproduces_real_itemsets_better_than_popularity(small_data, generated):
    real = [s["courses"] for s in small_data["train"]]
    model = compare_itemsets(real, generated, min_support=0.08, max_size=3)
    baseline = compare_itemsets(real, popularity_histories(real, 400), min_support=0.08, max_size=3)
    assert model["recall"] > 0.5 and model["support_correlation"] > 0.85
    assert model["support_correlation"] > baseline["support_correlation"] + 0.2
    assert model["precision"] > baseline["precision"]


def test_planted_checks_measure_bundles_and_paths(small_data, generated):
    real = [s["courses"] for s in small_data["train"]]
    checks = planted_checks(real, real)
    assert all(b["real"] == b["generated"] > 0.3 for b in checks["bundles"])
    assert checks["generated_valid_order"] == 1.0 and checks["generated_repeats"] == 0.0
    assert planted_checks(real, generated)["course_frequency_correlation"] > 0.8


def test_similar_courses_are_ranked_by_cosine_and_exclude_the_course(rec):
    sims = similar_courses(rec, "CS101", 5)
    assert len(sims) == 5 and "CS101" not in [c for c, _ in sims]
    values = [v for _, v in sims]
    assert values == sorted(values, reverse=True) and all(-1 <= v <= 1 for v in values)


def test_embedding_structure_reports_bundles_and_neighbours(rec):
    out = embedding_structure(rec)
    assert len(out["bundles"]) == 2 and set(out["nearest"]) == set(IDS)
    assert all(1 <= b["rank_among_neighbours_of_first"] <= len(IDS) - 1 for b in out["bundles"])


def test_attention_rows_are_distributions_over_the_history(rec):
    weights = attention_to_history(rec, ["MA101", "MA102", "CS101"])
    assert weights.shape == (rec.model.cfg.n_layers, 4)
    assert np.allclose(weights.sum(1), 1.0, atol=1e-5)


def test_plots_are_written(rec, tmp_path):
    plot_embeddings(rec, tmp_path / "e.png")
    plot_attention(rec, ["CS101", "CS102"], tmp_path / "a.png")
    assert (tmp_path / "e.png").stat().st_size > 10_000 and (tmp_path / "a.png").stat().st_size > 5_000
