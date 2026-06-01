"""Project-wide constants, column names, and paths."""

from pathlib import Path

RANDOM_SEED = 10676128          # N-number seed from the capstone (team member)
ALPHA = 0.005                   # significance threshold mandated by the spec
MIN_RATINGS = 10                # reliability threshold for the average rating

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

NUM_COLS = [
    "avg_rating", "avg_difficulty", "num_ratings", "received_pepper",
    "prop_retake", "num_online", "high_conf_male", "high_conf_female",
]
TAG_COLS = [
    "tough_grader", "good_feedback", "respected", "lots_to_read", "participation_matters",
    "dont_skip", "lots_of_homework", "inspirational", "pop_quizzes", "accessible",
    "so_many_papers", "clear_grading", "hilarious", "test_heavy", "graded_by_few",
    "amazing_lectures", "caring", "extra_credit", "group_projects", "lecture_heavy",
]
QUAL_COLS = ["major", "university", "state"]

TAG_LABELS = dict(zip(TAG_COLS, [
    "Tough grader", "Good feedback", "Respected", "Lots to read", "Participation matters",
    "Don't skip class", "Lots of homework", "Inspirational", "Pop quizzes!", "Accessible",
    "So many papers", "Clear grading", "Hilarious", "Test heavy", "Graded by few things",
    "Amazing lectures", "Caring", "Extra credit", "Group projects", "Lecture heavy",
], strict=True))
