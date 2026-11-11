"""Harmonogramy parametrów zależne od numeru epizodu."""


def linear(start: float, end: float, duration: int, t: int) -> float:
    """Liniowo od `start` (t=0) do `end` (t>=duration), potem stała."""
    if duration <= 0:
        return end
    frac = min(max(t / duration, 0.0), 1.0)
    return start + (end - start) * frac
