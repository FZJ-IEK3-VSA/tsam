"""Global runtime-tunable options for tsam."""

from __future__ import annotations


class Options:
    """Runtime-tunable options for tsam.

    Access and modify via the module-level ``tsam.options`` instance.

    Examples
    --------
    >>> import tsam
    >>> tsam.options.rescale_max_iterations = 50
    >>> tsam.options.rescale_tolerance = 1e-8
    >>> tsam.options.min_weight = 1e-4
    >>> tsam.options.reset()  # restore defaults
    """

    def __init__(self) -> None:
        self._rescale_max_iterations: int = 20
        self._rescale_tolerance: float = 1e-6
        self._min_weight: float = 1e-6

    @property
    def rescale_max_iterations(self) -> int:
        """Maximum iterations for rescaling convergence (default: 20)."""
        return self._rescale_max_iterations

    @rescale_max_iterations.setter
    def rescale_max_iterations(self, value: int) -> None:
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"rescale_max_iterations must be a positive integer, got {value}"
            )
        self._rescale_max_iterations = value

    @property
    def rescale_tolerance(self) -> float:
        """Convergence tolerance for rescaling (default: 1e-6)."""
        return self._rescale_tolerance

    @rescale_tolerance.setter
    def rescale_tolerance(self, value: float) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"rescale_tolerance must be a positive number, got {value}"
            )
        self._rescale_tolerance = float(value)

    @property
    def min_weight(self) -> float:
        """Minimum allowed column weight (default: 1e-6)."""
        return self._min_weight

    @min_weight.setter
    def min_weight(self, value: float) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"min_weight must be a positive number, got {value}")
        self._min_weight = float(value)

    def reset(self) -> None:
        """Reset all options to defaults."""
        self._rescale_max_iterations = 20
        self._rescale_tolerance = 1e-6
        self._min_weight = 1e-6

    def __repr__(self) -> str:
        return (
            f"Options(\n"
            f"  rescale_max_iterations={self.rescale_max_iterations},\n"
            f"  rescale_tolerance={self.rescale_tolerance},\n"
            f"  min_weight={self.min_weight},\n"
            f")"
        )


options = Options()
