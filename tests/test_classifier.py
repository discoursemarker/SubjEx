"""Regression tests for the public SubjEx API."""

from subjex import classify_subjects


def test_simple_subject() -> None:
    row = classify_subjects("Birds migrate every autumn.")

    assert row == "Birds migrate every autumn.\tBirds\t1\tPlural_only"


def test_multiple_subject_types() -> None:
    rows = classify_subjects(
        "There is a problem. It is difficult to please everyone."
    ).splitlines()

    assert rows[0] == "There is a problem.\tThere\t1\tThere"
    assert rows[1] == "It is difficult to please everyone.\tIt\t1\tit_to"


def test_questions_are_ignored() -> None:
    assert classify_subjects("Does this work?") == ""
