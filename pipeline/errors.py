"""Exit codes, the error type and the account-id mask shared by mlops.py, deploy.py, replay.py and feedback.py."""

from __future__ import annotations

import re
from typing import Any

EXIT_OK, EXIT_FAILED, EXIT_UNUSABLE, EXIT_GATE = 0, 1, 2, 3
ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")


class MlopsError(Exception):
    """Something to tell the user, with the exit code to use (1 = failed, 2 = model unusable here, 3 = gate said no)."""

    def __init__(self, message: str, code: int = EXIT_FAILED):
        super().__init__(message)
        self.code = code


def mask(text: Any) -> str:
    """Hide 12-digit account ids before anything is printed on a shared screen or into a log."""
    return ACCOUNT_ID.sub("<account-id>", str(text))


def error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}          # botocore's ClientError carries one
    return response.get("Error", {}).get("Code", "")
