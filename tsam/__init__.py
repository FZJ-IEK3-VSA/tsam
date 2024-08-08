import sys

if not sys.warnoptions:
    import warnings

    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        append=True,
        message=r".*The previous implementation of stack is deprecated and will be removed in a future version of pandas.*",
    )