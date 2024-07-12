import sys

if not sys.warnoptions:
    import warnings

    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning,
        append=True,
        #message="The previous implementation of stack is deprecated and will be removed in a future version of pandas. See the What's New notes for pandas 2.1.0 for details. Specify future_stack=True to adopt the new implementation and silence this warning.",
    )