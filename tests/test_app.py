"""Milestone 4: the Streamlit interface, driven headlessly with Streamlit's AppTest."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.predict import Recommender

MAIN = Path(__file__).resolve().parent.parent / "app" / "main.py"


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))
    at = AppTest.from_file(str(MAIN), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def shown(at) -> str:
    """The recommendation rows only, e.g. '**1. WD102 · JavaScript**  \nWeb Development'."""
    return "\n".join(m.value for m in at.markdown if re.match(r"\*\*\d+\. ", m.value))


def test_recommends_top_n_for_chosen_completed_courses(app):
    app.multiselect[0].set_value(["WD101"]).run()
    assert not app.exception
    assert "Recommended next" in [s.value for s in app.subheader]
    assert "1. WD102 · JavaScript" in shown(app)
    app.slider[0].set_value(3).run()
    assert len(shown(app).splitlines()) == 2 * 3  # 3 rows of two lines each


def test_not_interested_hides_a_course_and_is_remembered(app):
    app.multiselect[0].set_value(["WD101"]).run()
    app.button(key="hide-WD102").click().run()
    assert "WD102 · JavaScript" not in shown(app)
    again = AppTest.from_file(str(MAIN), default_timeout=30)
    again.run()
    again.multiselect[0].set_value(["WD101"]).run()
    assert "WD102 · JavaScript" not in shown(again)


def test_recommend_rows_flag_missing_prerequisites(tiny_model):
    rec = Recommender(*tiny_model[:2])
    rows = rec.recommend(["CS101"], k=42)
    assert {"course", "title", "track", "probability", "missing_prereqs"} <= set(rows[0])
    assert next(r for r in rows if r["course"] == "DS301")["missing_prereqs"] == ["DS201", "MA201", "MA202"]
    eligible = rec.recommend(["CS101"], k=42, only_eligible=True)
    assert eligible and all(not r["missing_prereqs"] for r in eligible)
    assert sum(r["probability"] for r in eligible) == pytest.approx(1.0)


def test_history_can_be_shared_in_the_url(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "app.json"))
    at = AppTest.from_file(str(MAIN), default_timeout=30)
    at.query_params["completed"] = "WD101,NOPE"
    at.run()
    assert at.multiselect[0].value == ["WD101"]
    assert "1. WD102 · JavaScript" in shown(at)


def test_similar_courses_and_the_analysis_tab_render(app):
    app.multiselect[0].set_value(["CS101"]).run()
    assert not app.exception
    assert "Explore similar courses" in [s.value for s in app.subheader]
    assert app.selectbox[0].options  # the recommended courses
    assert any("similarity" in m.value for m in app.markdown)
    assert [t.label for t in app.tabs] == ["Recommend", "What the model learned"]
    assert any("Itemsets" in m.value for m in app.tabs[1].markdown)
