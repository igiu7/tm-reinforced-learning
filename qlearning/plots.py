"""Wykresy przebiegu treningu z pliku episodes.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # zapis do pliku, bez okna -> nie blokuje treningu
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import read_log  # noqa: E402

# Paleta kategoryczna (stała kolejność) + kolory tekstu/tła
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
REASONS = ["finish", "off_track", "fell", "stuck", "timeout", "no_progress"]
REASON_LABELS = {"finish": "meta", "off_track": "zjazd z trasy", "fell": "wypadnięcie",
                 "stuck": "utknięcie", "timeout": "limit czasu", "no_progress": "brak postępu (banda)"}


def _moving_avg(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) == 0:
        return y
    w = max(1, min(w, len(y)))
    c = np.cumsum(np.insert(y.astype(float), 0, 0.0))
    out = np.empty(len(y))
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _style(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel("epizod", color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kw)
    return leg


def _no_data(ax, text: str):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK_2, fontsize=9)


def plot_training(csv_path: Path, out_path: Path, window: int = 20) -> Path | None:
    rows = read_log(csv_path)
    train = [r for r in rows if r["mode"] == "train"]
    evals = [r for r in rows if r["mode"] == "eval"]
    if not train:
        print(f"[PLOT] Brak danych w {csv_path}")
        return None

    def col(rs, k):
        return np.array([r[k] for r in rs])

    ep = col(train, "episode")
    ev_ep = col(evals, "episode") if evals else np.array([])

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), facecolor=SURFACE)
    (ax_time, ax_prog), (ax_rew, ax_reason), (ax_speed, ax_expl) = axes

    # --- 1. czasy ukończenia --------------------------------------------------
    _style(ax_time, "Czas przejazdu (ukończone)", "czas [s]")
    fin = [r for r in rows if r["finished"]]
    if fin:
        ftr = [r for r in fin if r["mode"] == "train"]
        fev = [r for r in fin if r["mode"] == "eval"]
        if ftr:
            ax_time.scatter(col(ftr, "episode"), col(ftr, "race_time_ms") / 1000, s=14, color=SERIES[0],
                            alpha=0.55, linewidths=0, label="trening (z eksploracją)")
        if fev:
            ax_time.scatter(col(fev, "episode"), col(fev, "race_time_ms") / 1000, s=40, color=SERIES[1],
                            edgecolors=SURFACE, linewidths=1.5, zorder=3, label="test (bez eksploracji)")
        # najlepszy czas do danego momentu
        fin_sorted = sorted(fin, key=lambda r: (r["episode"], r["mode"]))
        best, bx, by = None, [], []
        for r in fin_sorted:
            t = r["race_time_ms"] / 1000
            best = t if best is None else min(best, t)
            bx.append(r["episode"]); by.append(best)
        bx.append(max(ep.max(), bx[-1])); by.append(best)
        ax_time.step(bx, by, where="post", color=INK, linewidth=2, label="najlepszy dotąd")
        ax_time.annotate(f"{best:.2f} s", (bx[-1], best), xytext=(-4, -12), textcoords="offset points",
                         ha="right", color=INK, fontsize=9, fontweight="bold")
        _legend(ax_time, loc="upper right")
    else:
        _no_data(ax_time, "Jeszcze żaden przejazd nie dojechał do mety")

    # --- 2. postęp trasy -------------------------------------------------------
    _style(ax_prog, "Przejechana część trasy", "postęp [%]")
    prog = col(train, "progress_pct")
    ax_prog.plot(ep, prog, color=SERIES[0], alpha=0.25, linewidth=1)
    ax_prog.plot(ep, _moving_avg(prog, window), color=SERIES[0], linewidth=2, label=f"trening (śr. z {window})")
    if evals:
        ax_prog.plot(ev_ep, col(evals, "progress_pct"), color=SERIES[1], linewidth=2, marker="o",
                     markersize=5, markeredgecolor=SURFACE, label="test")
    ax_prog.set_ylim(0, 105)
    _legend(ax_prog, loc="lower right")

    # --- 3. nagroda ------------------------------------------------------------
    _style(ax_rew, "Suma nagród w epizodzie", "nagroda")
    rew = col(train, "reward")
    ax_rew.plot(ep, rew, color=SERIES[0], alpha=0.25, linewidth=1)
    ax_rew.plot(ep, _moving_avg(rew, window), color=SERIES[0], linewidth=2, label=f"trening (śr. z {window})")
    if evals:
        ax_rew.plot(ev_ep, col(evals, "reward"), color=SERIES[1], linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, label="test")
    _legend(ax_rew, loc="lower right")

    # --- 4. powody zakończenia -------------------------------------------------
    _style(ax_reason, f"Jak kończą się epizody (okno {window})", "udział [%]")
    reasons = np.array([r["reason"] for r in train])
    shares = [_moving_avg((reasons == k).astype(float), window) * 100 for k in REASONS]
    ax_reason.stackplot(ep, shares, colors=SERIES, edgecolor=SURFACE, linewidth=1,
                        labels=[REASON_LABELS[k] for k in REASONS])
    ax_reason.set_ylim(0, 100)
    ax_reason.set_xlim(ep.min(), max(ep.max(), ep.min() + 1))
    _legend(ax_reason, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    # --- 5. prędkość -----------------------------------------------------------
    _style(ax_speed, "Prędkość", "km/h")
    ax_speed.plot(ep, _moving_avg(col(train, "max_speed_kmh"), window), color=SERIES[1], linewidth=2,
                  label="maksymalna")
    ax_speed.plot(ep, _moving_avg(col(train, "avg_speed_kmh"), window), color=SERIES[0], linewidth=2,
                  label="średnia")
    ax_speed.set_ylim(bottom=0)
    _legend(ax_speed, loc="lower right")

    # --- 6. eksploracja --------------------------------------------------------
    _style(ax_expl, "Eksploracja", "%")
    ax_expl.plot(ep, col(train, "epsilon") * 100, color=SERIES[0], linewidth=2, label="epsilon (losowe akcje)")
    ax_expl.plot(ep, col(train, "visited_states_pct"), color=SERIES[2], linewidth=2,
                 label="odwiedzone stany tablicy Q")
    ax_expl.set_ylim(0, 105)
    _legend(ax_expl, loc="center right")

    # --- podsumowanie ----------------------------------------------------------
    n_fin = sum(r["finished"] for r in train)
    best_all = min((r["race_time_ms"] for r in fin), default=None)
    secs = sum(r["wall_time_s"] for r in rows)
    duration = f"{secs / 3600:.1f} h" if secs >= 3600 else f"{secs / 60:.0f} min"
    summary = (f"{len(train)} epizodów treningowych  ·  {n_fin} ukończonych  ·  "
               f"najlepszy czas {best_all / 1000:.2f} s" if best_all else
               f"{len(train)} epizodów treningowych  ·  brak ukończonych")
    fig.suptitle("Q(λ) – przebieg treningu", x=0.01, ha="left", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.01, 0.955, f"{summary}  ·  czas treningu {duration}", ha="left", fontsize=10, color=INK_2)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out_path
