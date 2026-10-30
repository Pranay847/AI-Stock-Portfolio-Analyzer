import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.owner_access import MIN_TOKEN_LENGTH, token_problem, verify_owner_token

GOOD = "x" * MIN_TOKEN_LENGTH


@pytest.mark.parametrize("token", [None, "", "x" * (MIN_TOKEN_LENGTH - 1)])
def test_unusable_tokens_are_reported(token):
    assert token_problem(token) is not None


def test_usable_token():
    assert token_problem(GOOD) is None


def test_verify():
    assert verify_owner_token(GOOD, GOOD) is True
    assert verify_owner_token(GOOD + "y", GOOD) is False
    assert verify_owner_token("", GOOD) is False
    assert verify_owner_token(None, GOOD) is False


@pytest.mark.parametrize("configured", [None, "", "short"])
def test_nothing_verifies_against_an_unusable_token(configured):
    # Otherwise an empty configured token would accept an empty submission.
    assert verify_owner_token(configured or "", configured) is False
