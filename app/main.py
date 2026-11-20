"""Streamlit app (milestone 4): `streamlit run app/main.py`

Pick the courses you have completed, in the order you took them, and see the model's top-N next courses with
their probabilities. "Not interested" hides a course from all later recommendations (kept in data/app.json).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `app.*` imports work under streamlit

from app.catalog import BY_ID, IDS, TRACKS, label  # noqa: E402
from app.data import valid_order  # noqa: E402
from app.model import MODEL_PATH  # noqa: E402
from app.predict import Recommender  # noqa: E402
from app.storage import Store  # noqa: E402

st.set_page_config(page_title="Learning Path Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def get_recommender() -> Recommender:
    return Recommender.from_file(MODEL_PATH)


store = Store()
st.title("🎓 Learning Path Predictor")
st.caption("A small GPT trained on 8,000 (synthetic) student histories suggests what to study next.")

if not Path(MODEL_PATH).exists():
    st.error(f"No trained model at `{MODEL_PATH}`. Run `make train` (or `python -m app.model`) first.")
    st.stop()
rec = get_recommender()

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

with st.expander("Course catalogue"):
    for track in TRACKS:
        courses = [c for c in IDS if BY_ID[c].track == track]
        st.markdown(f"**{track}:** " + ", ".join(
            f"{c} {BY_ID[c].title}" + (f" (needs {', '.join(BY_ID[c].prereqs)})" if BY_ID[c].prereqs else "")
            for c in courses))
