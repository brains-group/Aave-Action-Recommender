"""Optional, study-local fixed follow-up for the shortest-time stress test.

The ordinary agent and evaluator retain their historical variable follow-up.
Callers install this wrapper on their imported function alias before forking.
"""

from functools import wraps


def fixed_followup(original, horizon_seconds):
    if not isinstance(horizon_seconds, int) or horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be a positive integer")

    @wraps(original)
    def limited_profile(recommendation, return_extras=False):
        value = original(recommendation, return_extras=return_extras)
        if not return_extras or value is None:
            return value
        profile, decision, cutoff, user, future, _legacy_horizon = value
        return profile, decision, cutoff, user, future, decision - cutoff + horizon_seconds

    return limited_profile
