"""Wykresy przebiegu treningu z pliku episodes.csv (wspólne dla wszystkich algorytmów)."""

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
    """Średnia krocząca ignorująca braki danych (nan)."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    w = max(1, min(w, len(y)))
    valid = ~np.isnan(y)
    c = np.cumsum(np.insert(np.where(valid, y, 0.0), 0, 0.0))
    n = np.cumsum(np.insert(valid.astype(float), 0, 0.0))
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        cnt = n[i + 1] - n[lo]
        if cnt > 0:
            out[i] = (c[i + 1] - c[lo]) / cnt
    return out


def _style(ax, title: str, ylabel: str, xlabel: str = "epizod"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def _legend(ax, **kw):
    return ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kw)


def _no_data(ax, text: str):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=INK_2, fontsize=9)


def _raw_and_avg(ax, x, y, window, color, label):
    ax.plot(x, y, color=color, alpha=0.25, linewidth=1)
    ax.plot(x, _moving_avg(y, window), color=color, linewidth=2, label=f"{label} (śr. z {window})")


def plot_training(csv_path: Path, out_path: Path, window: int = 20) -> Path | None:
    rows = read_log(csv_path)
    train = [r for r in rows if r["mode"] == "train"]
    evals = [r for r in rows if r["mode"] == "eval"]
    if not train:
        print(f"[PLOT] Brak danych w {csv_path}")
        return None

    def col(rs, k):
        return np.array([r[k] for r in rs], dtype=float)

    ep = col(train, "episode")
    ev_ep = col(evals, "episode") if evals else np.array([])

    fig, axes = plt.subplots(4, 2, figsize=(15, 16), facecolor=SURFACE)
    (ax_time, ax_prog), (ax_rew, ax_reason), (ax_speed, ax_eps), (ax_loss, ax_q) = axes

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
        best, bx, by = None, [], []
        for r in sorted(fin, key=lambda r: (r["episode"], r["mode"])):
            t = r["race_time_ms"] / 1000
            best = t if best is None else min(best, t)
            bx.append(r["episode"])
            by.append(best)
        bx.append(max(ep.max(), bx[-1]))
        by.append(best)
        ax_time.step(bx, by, where="post", color=INK, linewidth=2, label="najlepszy dotąd")
        ax_time.annotate(f"{best:.2f} s", (bx[-1], best), xytext=(-4, -12), textcoords="offset points",
                         ha="right", color=INK, fontsize=9, fontweight="bold")
        _legend(ax_time, loc="upper right")
    else:
        _no_data(ax_time, "Jeszcze żaden przejazd nie dojechał do mety")

    # --- 2. postęp trasy -------------------------------------------------------
    _style(ax_prog, "Przejechana część trasy", "postęp [%]")
    _raw_and_avg(ax_prog, ep, col(train, "progress_pct"), window, SERIES[0], "trening")
    if evals:
        ax_prog.plot(ev_ep, col(evals, "progress_pct"), color=SERIES[1], linewidth=2, marker="o",
                     markersize=5, markeredgecolor=SURFACE, label="test")
    ax_prog.set_ylim(0, 105)
    _legend(ax_prog, loc="lower right")

    # --- 3. nagroda ------------------------------------------------------------
    _style(ax_rew, "Suma nagród w epizodzie", "nagroda")
    _raw_and_avg(ax_rew, ep, col(train, "reward"), window, SERIES[0], "trening")
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
    _style(ax_eps, "Eksploracja: udział losowych akcji (epsilon)", "epsilon")
    ax_eps.plot(ep, col(train, "epsilon"), color=SERIES[0], linewidth=2)
    ax_eps.set_ylim(0, 1.02)

    # --- 7. strata uczenia -----------------------------------------------------
    loss = col(train, "loss")
    _style(ax_loss, "Strata uczenia (Huber)", "loss")
    if np.isnan(loss).all():
        _no_data(ax_loss, "Brak aktualizacji sieci (bufor jeszcze się zapełnia)")
    else:
        _raw_and_avg(ax_loss, ep, loss, window, SERIES[0], "strata")
        _legend(ax_loss, loc="upper right")

    # --- 8. średnie Q ----------------------------------------------------------
    q = col(train, "q_mean")
    _style(ax_q, "Średnia przewidywana wartość Q", "Q")
    if np.isnan(q).all():
        _no_data(ax_q, "Brak aktualizacji sieci (bufor jeszcze się zapełnia)")
    else:
        _raw_and_avg(ax_q, ep, q, window, SERIES[0], "Q")
        _legend(ax_q, loc="lower right")

    # --- podsumowanie ----------------------------------------------------------
    n_fin = sum(r["finished"] for r in train)
    best_all = min((r["race_time_ms"] for r in fin), default=None)
    secs = sum(r["wall_time_s"] for r in rows)
    duration = f"{secs / 3600:.1f} h" if secs >= 3600 else f"{secs / 60:.0f} min"
    steps = int(train[-1]["total_steps"]) if train else 0
    summary = (f"{len(train)} epizodów treningowych ({steps:,} kroków)  ·  {n_fin} ukończonych  ·  "
               + (f"najlepszy czas {best_all / 1000:.2f} s" if best_all else "brak ukończonych"))
    fig.suptitle("Przebieg treningu", x=0.01, y=0.995, ha="left", va="top", fontsize=15, color=INK,
                 fontweight="bold")
    fig.text(0.01, 0.975, f"{summary}  ·  czas treningu {duration}", ha="left", va="top", fontsize=10,
             color=INK_2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out_path
