import sys

if not sys.warnoptions:
    import warnings

    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        append=True,
        message="*The previous implementation of stack is deprecated.*",
    )