"""Custom exceptions and warnings for tsam."""


class LegacyAPIWarning(DeprecationWarning):
    """Warning for deprecated tsam legacy API usage.

    This warning is raised when using the old class-based API
    (TimeSeriesAggregation) instead of the new functional API (aggregate).

    Users can suppress this warning during migration with:

        import warnings
        from tsam.exceptions import LegacyAPIWarning
        warnings.filterwarnings("ignore", category=LegacyAPIWarning)
    """

    pass
