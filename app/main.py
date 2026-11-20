"""Streamlit app (milestone 4): `streamlit run app/main.py`

Pick the courses you have completed, in the order you took them, and see the model's top-N next courses with
their probabilities. "Not interested" hides a course from all later recommendations (kept in data/app.json).
"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `app.*` imports work under streamlit

from app.analyze import similar_courses  # noqa: E402
from app.catalog import BY_ID, IDS, ROOT, TRACKS, label  # noqa: E402
from app.data import valid_order  # noqa: E402
from app.model import MODEL_PATH  # noqa: E402
from app.predict import Recommender  # noqa: E402
from app.storage import Store  # noqa: E402

st.set_page_config(page_title="Learning Path Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def get_recommender() -> Recommender:
    return Recommender.from_file(MODEL_PATH)


store = Store(os.environ.get("DATA_PATH") or str(ROOT / "data" / "app.json"))
st.title("🎓 Learning Path Predictor")
st.caption("A small GPT trained on 8,000 (synthetic) student histories suggests what to study next.")

if not Path(MODEL_PATH).exists():
    st.error(f"No trained model at `{MODEL_PATH}`. Run `make train` (or `python -m app.model`) first.")
    st.stop()
rec = get_recommender()
recommend_tab, learned_tab = st.tabs(["Recommend", "What the model learned"])

with st.sidebar:
    st.header("Settings")
    k = st.slider("How many recommendations", 1, 10, 5)
    only_eligible = st.toggle("Only courses I can take now", value=False,
                              help="Hide courses whose prerequisites you have not completed yet.")
    hidden = store.get("not_interested", [])
    st.subheader("Not interested")
    if hidden:
        st.write(", ".join(hidden))
        if st.button("Clear list"):
            store.put("not_interested", [])
            st.rerun()
    else:
        st.caption("Nothing hidden yet.")

with recommend_tab:
    # ?completed=CS101,MA101 in the URL pre-fills the history, so a learning path can be shared as a link
    from_url = [c for c in st.query_params.get("completed", "").split(",") if c in BY_ID]
    completed = st.multiselect("Courses you have completed, in the order you took them", IDS, default=from_url,
                               format_func=label, placeholder="Start typing a course, e.g. CS101")

    if completed and not valid_order(completed):
        st.info("Some courses come before their prerequisites in this order. The model still works, but it was "
                "trained on histories that respect prerequisites.")

    st.subheader("Recommended next" if completed else "Where most students start")
    rows = rec.recommend(completed, k=k, exclude=hidden, only_eligible=only_eligible)
    if not rows:
        st.warning("No courses left to recommend: clear the 'not interested' list or untick the filter.")
    for i, row in enumerate(rows, 1):
        c1, c2, c3, c4 = st.columns([5, 3, 2, 2], vertical_alignment="center")
        c1.markdown(f"**{i}. {row['course']} · {row['title']}**  \n{row['track']}")
        c2.progress(row["probability"], text=f"{row['probability']:.1%}")
        if row["missing_prereqs"]:
            c3.markdown(f"⚠️ needs {', '.join(row['missing_prereqs'])}")
        else:
            c3.markdown("✅ prerequisites met")
        if c4.button("Not interested", key=f"hide-{row['course']}"):
            store.put("not_interested", sorted(set(hidden) | {row["course"]}))
            st.rerun()

    if rows:
        st.subheader("Explore similar courses")
        pick = st.selectbox("Courses the model treats most like", [r["course"] for r in rows], format_func=label)
        st.caption("Cosine similarity of the model's course embeddings: courses that play the same role in "
                   "learning paths, i.e. alternatives (see 'What the model learned').")
        for course, sim in similar_courses(rec, pick, 5):
            st.markdown(f"- {label(course)} · {BY_ID[course].track} · similarity {sim:.2f}")

    with st.expander("Course catalogue"):
        for track in TRACKS:
            courses = [c for c in IDS if BY_ID[c].track == track]
            st.markdown(f"**{track}:** " + ", ".join(
                f"{c} {BY_ID[c].title}" + (f" (needs {', '.join(BY_ID[c].prereqs)})" if BY_ID[c].prereqs else "")
                for c in courses))

with learned_tab:
    report = ROOT / "results" / "itemsets.md"
    if report.exists():
        text = report.read_text().replace("![embeddings](embeddings.png)", "").replace(
            "![attention](attention.png)", "")
        st.markdown(text.split("## Embeddings")[0])
        st.image(str(ROOT / "results" / "embeddings.png"))
        st.markdown("## Embeddings" + text.split("## Embeddings")[1].split("## Attention")[0])
        st.image(str(ROOT / "results" / "attention.png"))
        st.markdown("## Attention" + text.split("## Attention")[1])
    else:
        st.info("Run `make analyze` (or `python -m app.analyze`) to produce the analysis.")
