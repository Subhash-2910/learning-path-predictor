"""The course catalogue: 43 courses, their prerequisites, and the learning paths students follow.

Everything here is made up. Because we write the paths ourselves, we know the right answer when we later ask
what the model has learned (milestone 5).
"""
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent  # the project folder, wherever the app is started from


@dataclass(frozen=True)
class Course:
    id: str
    title: str
    track: str
    prereqs: tuple = field(default_factory=tuple)


def _c(id, title, track, *prereqs):
    return Course(id, title, track, tuple(prereqs))


COURSES = [
    # Foundations: no prerequisites
    _c("CS101", "Intro to Programming (Python)", "Foundations"),
    _c("MA101", "Calculus I", "Foundations"),
    _c("MA102", "Discrete Mathematics", "Foundations"),
    _c("ST101", "Intro to Statistics", "Foundations"),
    _c("EN101", "Academic Writing", "Foundations"),
    # Computer science
    _c("CS102", "Data Structures", "Computer Science", "CS101"),
    _c("CS201", "Algorithms", "Computer Science", "CS102", "MA102"),
    _c("CS202", "Databases & SQL", "Computer Science", "CS101"),
    _c("CS203", "Operating Systems", "Computer Science", "CS102"),
    _c("CS301", "Computer Networks", "Computer Science", "CS203"),
    _c("CS302", "Distributed Systems", "Computer Science", "CS301"),
    _c("CS303", "Compilers", "Computer Science", "CS201"),
    # Mathematics and statistics
    _c("MA201", "Linear Algebra", "Mathematics", "MA101"),
    _c("MA202", "Probability", "Mathematics", "MA101"),
    _c("MA203", "Calculus II", "Mathematics", "MA101"),
    _c("MA301", "Optimization", "Mathematics", "MA201", "MA203"),
    _c("ST201", "Regression Modelling", "Mathematics", "ST101", "MA201"),
    _c("ST301", "Bayesian Statistics", "Mathematics", "MA202", "ST201"),
    # Data science and machine learning
    _c("DS201", "Data Analysis with Pandas", "Data Science", "CS101", "ST101"),
    _c("DS202", "Data Visualization", "Data Science", "DS201"),
    _c("DS301", "Machine Learning", "Data Science", "DS201", "MA201", "MA202"),
    _c("DS302", "Deep Learning", "Data Science", "DS301"),
    _c("DS303", "Natural Language Processing", "Data Science", "DS302"),
    _c("DS304", "Computer Vision", "Data Science", "DS302"),
    _c("DS305", "MLOps", "Data Science", "DS301", "CS202"),
    # Web development
    _c("WD101", "HTML & CSS", "Web Development"),
    _c("WD102", "JavaScript", "Web Development", "WD101"),
    _c("WD201", "Frontend Frameworks (React)", "Web Development", "WD102"),
    _c("WD202", "Backend APIs", "Web Development", "CS101", "CS202"),
    _c("WD301", "Full-Stack Project", "Web Development", "WD201", "WD202"),
    _c("WD302", "Cloud Deployment", "Web Development", "WD202"),
    # Security
    _c("SE201", "Intro to Cybersecurity", "Security", "CS101"),
    _c("SE301", "Network Security", "Security", "SE201", "CS301"),
    _c("SE302", "Cryptography", "Security", "SE201", "MA102"),
    # Business and design
    _c("BU101", "Intro to Business", "Business & Design"),
    _c("BU201", "Product Management", "Business & Design", "BU101"),
    _c("BU202", "Business Analytics", "Business & Design", "BU101", "ST101"),
    _c("UX201", "UX Design", "Business & Design", "WD101"),
    _c("UX301", "User Research", "Business & Design", "UX201"),
    # Electives: anyone can take them at any time (they are the noise in the data)
    _c("EL101", "Public Speaking", "Electives"),
    _c("EL102", "Ethics in Technology", "Electives"),
    _c("EL103", "Entrepreneurship", "Electives"),
    _c("EL104", "Intro to Psychology", "Electives"),
]

BY_ID = {c.id: c for c in COURSES}
IDS = [c.id for c in COURSES]
TRACKS = list(dict.fromkeys(c.track for c in COURSES))

# Planted learning paths: each student mostly follows one of these, in this order. Several share their first
# courses and only split later, so the right next course often depends on the whole history, not only the last
# course (CS101 is followed by CS102 on the backend path, but by WD201 for a student who started with WD101).
GOALS = {
    "ml_engineer": ["CS101", "MA101", "ST101", "DS201", "MA201", "MA202", "DS301", "DS302", "DS303"],
    "data_analyst": ["CS101", "ST101", "DS201", "DS202", "BU101", "BU202", "CS202"],
    "backend": ["CS101", "CS102", "CS202", "WD202", "CS203", "CS301", "WD302", "CS302"],
    "full_stack": ["WD101", "WD102", "CS101", "WD201", "CS202", "WD202", "WD301"],
    "security": ["CS101", "CS102", "MA102", "SE201", "CS203", "CS301", "SE301", "SE302"],
    "cs_theory": ["MA101", "MA102", "CS101", "CS102", "CS201", "MA201", "MA203", "CS303", "MA301"],
    "product": ["BU101", "EN101", "WD101", "UX201", "BU201", "ST101", "BU202", "UX301"],
    "statistician": ["MA101", "ST101", "MA202", "MA201", "ST201", "CS101", "DS201", "ST301"],
}
GOAL_WEIGHTS = {"ml_engineer": 3, "data_analyst": 2, "backend": 2, "full_stack": 2,
                "security": 1, "cs_theory": 1, "product": 1, "statistician": 1}

# Planted pairs: when a student takes the first course, they take the second right after it with this
# probability. These are "itemsets" that cut across tracks, so a model can only know them from the data.
BUNDLES = [("DS301", "EL102", 0.6), ("BU201", "EL101", 0.6)]


def prereqs_met(course_id: str, taken) -> bool:
    return all(p in taken for p in BY_ID[course_id].prereqs)


def label(course_id: str) -> str:
    return f"{course_id} · {BY_ID[course_id].title}"
