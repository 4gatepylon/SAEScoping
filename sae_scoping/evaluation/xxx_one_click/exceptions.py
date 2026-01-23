from __future__ import annotations

"""We use custom exceptions with try-catch to avoid intersecting errors from outside this library."""


class TooManyRequestsError(Exception):
    pass


class TooManyRequestsErrorLocal(Exception):
    pass  # Local = based on the settings for your method call


class TooManyRequestsErrorGlobal(Exception):
    pass  # Global = based on the settings for your object
