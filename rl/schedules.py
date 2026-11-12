"""Harmonogramy parametrów zależne od numeru epizodu."""


def linear(start: float, end: float, duration: int, t: int) -> float:
    """Liniowo od `start` (t=0) do `end` (t>=duration), potem stała."""
    if duration <= 0:
        return end
    frac = min(max(t / duration, 0.0), 1.0)
    return start + (end - start) * frac


def piecewise(points, t: int) -> float:
    """Liniowo między punktami [(t0, v0), (t1, v1), ...]; przed pierwszym v0, po ostatnim stała."""
    points = sorted(points)
    if t <= points[0][0]:
        return float(points[0][1])
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t <= t1:
            return float(v0 + (v1 - v0) * (t - t0) / max(t1 - t0, 1))
    return float(points[-1][1])
