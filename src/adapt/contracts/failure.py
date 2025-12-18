"""Centralized failure policy for contract violations.

Contracts fail fast, loud, and once. All violations raise the same
exception type, allowing caller to handle pipeline bugs uniformly.
"""

from enum import Enum


class FailurePolicy(str, Enum):
    """Failure policy for contract violations.
    
    FAIL_FAST (default): Raise immediately on contract violation
    
    Future options for extensibility:
    - SKIP_FILE: Mark file failed, continue to next file
    - WARN_ONLY: Log and continue (never used in production)
    - QUARANTINE: Mark for manual review
    """
    FAIL_FAST = "fail_fast"


class ContractViolation(RuntimeError):
    """Raised when a pipeline contract is violated.

    This indicates a bug in pipeline logic, not bad user input or recoverable
    science edge cases. It means a pipeline stage did not produce the invariants
    it promised.

    Key distinction:
    - ValueError: User/config error (handled by Pydantic)
    - ContractViolation: Pipeline bug (programmer error)
    - Exception: Recoverable science issues (try/except in algorithms)
    """
    pass
