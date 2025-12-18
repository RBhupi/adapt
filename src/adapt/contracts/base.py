"""Base contract enforcement utilities.

The require() function is the single enforcement mechanism for all contracts.
It is NOT defensive programming — it enforces semantic invariants of the pipeline.
"""

from adapt.contracts.failure import ContractViolation


def require(condition: bool, message: str) -> None:
    """Enforce a pipeline contract.

    This is called at stage boundaries to verify the preceding stage
    produced the guaranteed invariants. It is fail-fast: no recovery,
    no fallback, no silence.

    Parameters
    ----------
    condition : bool
        The invariant that must be true. If False, ContractViolation is raised.

    message : str
        Error message explaining the contract violation (for debugging).

    Raises
    ------
    ContractViolation
        If condition is False. This indicates a bug in pipeline logic.

    Examples
    --------
    >>> require("x" in ds.coords, "Grid contract: missing 'x' coordinate")
    >>> require(df.shape[0] > 0, "Analysis contract: at least one cell expected")
    """
    if not condition:
        raise ContractViolation(message)
