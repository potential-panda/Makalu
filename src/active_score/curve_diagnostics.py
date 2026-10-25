"""
Shared diagnostics for any (day_of_week x time-of-day bin) curve produced by
active_score.weekly_curve.aggregate_weekly -- heatmap plotting, per-weekday
summary, active/inactive window derivation, and the cross-period stability
check. Operates purely on the [7][n_bins] `scores` array and the underlying
per-series data, so it's agnostic to which dataset the curve came from
(Volatility/Spread-derived ActivityCurve, LeadLag-derived LeadLagCurve, etc).

Used by active_score/inspect_curve.py (ActivityCurve) and
active_score/inspect_leadlag_curve.py (LeadLagCurve).
"""

import logging
from pathlib import Path

import numpy as np

from active_score.weekly_curve import aggregate_weekly

LOG = logging.getLogger(__name__)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def find_active_window(
    hourly_scores: np.ndarray,
) -> tuple[int, int, float, float] | None:
    """Longest contiguous UTC-hour run above the day's own mean score -- the
    simplification described in doc/vol_spread.md's "Active/Inactive Window
    Simplification" (raw 5-min bins are too noisy to schedule against; the
    active/inactive *ratio* of a coarse contiguous block is what actually
    holds up across independent halves of the training data).

    Rotates to start scanning at the quietest hour so the active run (assumed
    not to wrap past the daily minimum) is captured as one simple slice.
    Returns (start_hour, end_hour, inside_mean, outside_mean), UTC, half-open
    [start, end); None if every hour is on the same side of the threshold.
    """
    n = len(hourly_scores)
    quiet_hour = int(np.argmin(hourly_scores))
    rotated = np.roll(hourly_scores, -quiet_hour)
    threshold = hourly_scores.mean()
    above = rotated > threshold

    best_start, best_len, cur_start, cur_len = None, 0, None, 0
    for i, v in enumerate(above):
        if v:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_start, cur_len = None, 0
    if best_start is None:
        return None

    start = (quiet_hour + best_start) % n
    end = (start + best_len) % n
    idx = [(start + i) % n for i in range(best_len)]
    outside_idx = [h for h in range(n) if h not in idx]
    return (
        start,
        end,
        float(hourly_scores[idx].mean()),
        float(hourly_scores[outside_idx].mean()),
    )


def _fmt_utc_jst(hour: int) -> str:
    return f"UTC {hour:02d}:00 (JST {(hour + 9) % 24:02d}:00)"


def weekday_hourly(scores: np.ndarray, bin_size_sec: int) -> np.ndarray:
    """Mon-Fri rows averaged down to one representative day, at 1h resolution."""
    bins_per_hour = 3600 // bin_size_sec
    weekday_avg = scores[0:5].mean(axis=0)
    n_hours = len(weekday_avg) // bins_per_hour
    return (
        weekday_avg[: n_hours * bins_per_hour]
        .reshape(n_hours, bins_per_hour)
        .mean(axis=1)
    )


def print_active_window(scores: np.ndarray, bin_size_sec: int) -> dict:
    """Weekday-only (Mon-Fri) active/inactive window; weekend is always
    reported as fully inactive -- see doc/vol_spread.md for why (weekend
    activity correlates poorly with the average day, not just around
    exchange maintenance reopen windows)."""
    hourly = weekday_hourly(scores, bin_size_sec)
    window = find_active_window(hourly)
    print("\n── Active/inactive window (weekday, see doc/vol_spread.md) ──")
    if window is None:
        print("  No contiguous active window found (curve too flat).")
        return {}
    start, end, inside_mean, outside_mean = window
    print(
        f"  Active:   {_fmt_utc_jst(start)} -> {_fmt_utc_jst(end)}  (mean={inside_mean:.4f})"
    )
    print(
        f"  Inactive: {_fmt_utc_jst(end)} -> {_fmt_utc_jst(start)}  (mean={outside_mean:.4f})"
    )
    print("  Weekend:  inactive (all)")
    print(f"  active/inactive ratio: {inside_mean / outside_mean:.2f}x")
    print("───────────────────────────────────────────────────────────\n")
    return {
        "weekday_active_utc": [f"{start:02d}:00", f"{end:02d}:00"],
        "weekday_active_jst": [
            f"{(start + 9) % 24:02d}:00",
            f"{(end + 9) % 24:02d}:00",
        ],
        "weekend": "inactive",
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
    }


def print_weekday_summary(scores: np.ndarray) -> None:
    """Per-weekday mean/std/min/max, plus each day's correlation with the
    across-week-average day curve -- a quick numeric eye-check for whether
    the pattern is genuinely weekly or just the same day repeated 7 times
    (correlations near 1.0 for every day => no real day-of-week structure)."""
    avg_day = scores.mean(axis=0)
    print("\n── Per-weekday summary ──────────────────────────────────────")
    print(
        f"  {'day':<5}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}  {'corr_vs_avg_day':>16}"
    )
    print(f"  {'-' * 58}")
    for d in range(7):
        row = scores[d]
        corr = (
            np.corrcoef(row, avg_day)[0, 1]
            if row.std() > 0 and avg_day.std() > 0
            else float("nan")
        )
        print(
            f"  {DAY_NAMES[d]:<5}  {row.mean():>7.4f}  {row.std():>7.4f}  {row.min():>7.4f}  {row.max():>7.4f}  {corr:>16.4f}"
        )
    print("────────────────────────────────────────────────────────────\n")


def _curve_from_subset(series_list: list, bin_size_sec: int) -> np.ndarray | None:
    series_list = [s for s in series_list if len(s) > 100]
    if not series_list:
        return None
    scores, _ = aggregate_weekly(series_list, bin_size_sec)
    return scores


def _compare_curves(a: np.ndarray, b: np.ndarray) -> dict:
    """Overall + per-weekday correlation and mean|Δ| between two independently
    computed curves -- the actual test of "is this curve statistically
    stable", since within-cell std overstates confidence (points inside a
    cell are highly autocorrelated) while agreement between two disjoint
    halves of the data is not."""
    fa, fb = a.flatten(), b.flatten()
    overall_corr = (
        float(np.corrcoef(fa, fb)[0, 1])
        if fa.std() > 0 and fb.std() > 0
        else float("nan")
    )
    per_day = {}
    for d in range(7):
        row_a, row_b = a[d], b[d]
        corr = (
            float(np.corrcoef(row_a, row_b)[0, 1])
            if row_a.std() > 0 and row_b.std() > 0
            else float("nan")
        )
        per_day[DAY_NAMES[d]] = {
            "corr": corr,
            "mean_abs_diff": float(np.mean(np.abs(row_a - row_b))),
        }
    return {
        "overall_corr": overall_corr,
        "overall_mean_abs_diff": float(np.mean(np.abs(fa - fb))),
        "per_day": per_day,
    }


def _apply_window(
    scores: np.ndarray, bin_size_sec: int, start: int, end: int
) -> tuple[float, float]:
    hourly = weekday_hourly(scores, bin_size_sec)
    n = len(hourly)
    idx = (
        list(range(start, end))
        if start < end
        else list(range(start, n)) + list(range(end))
    )
    outside_idx = [h for h in range(n) if h not in idx]
    return float(hourly[idx].mean()), float(hourly[outside_idx].mean())


def run_stability_check(
    series_list: list, bin_size_sec: int, active_window: dict, min_span_days: int = 14
) -> dict:
    """Cross-period check for whether the curve is statistically stable, per
    doc/vol_spread.md's "Active/Inactive Window Simplification": splits the
    already-loaded series two ways and compares the resulting curves.

      chronological (first half vs second half of the date range) -- tests
      whether the pattern still holds later in time (regime drift), the more
      decision-relevant question for "can I trust this going forward".

      interleaved (odd vs even ISO week, same time span) -- tests pure
      sampling stability with the regime held fixed.

    Also re-checks the already-derived active/inactive window against each
    half, since a window that only looks good on the full-sample average is
    not useful for scheduling.
    """
    overall_min = min(s.index.min() for s in series_list)
    overall_max = max(s.index.max() for s in series_list)
    span_days = (overall_max - overall_min).days
    print("\n── Cross-period stability check ──────────────────────────────")
    if span_days < min_span_days:
        print(
            f"  Skipped: date range only spans {span_days}d, need >= {min_span_days}d for a meaningful check."
        )
        print("─────────────────────────────────────────────────────────────\n")
        return {}

    midpoint = overall_min + (overall_max - overall_min) / 2
    h1 = [s[s.index < midpoint] for s in series_list]
    h2 = [s[s.index >= midpoint] for s in series_list]
    odd = [s[s.index.isocalendar().week.to_numpy() % 2 == 1] for s in series_list]
    even = [s[s.index.isocalendar().week.to_numpy() % 2 == 0] for s in series_list]

    result: dict = {}
    splits = {
        "chronological": ("H1 (first half)", "H2 (second half)", h1, h2),
        "interleaved": ("odd ISO week", "even ISO week", odd, even),
    }
    curves = {}
    for key, (label_a, label_b, sa, sb) in splits.items():
        ca, cb = (
            _curve_from_subset(sa, bin_size_sec),
            _curve_from_subset(sb, bin_size_sec),
        )
        curves[key] = (ca, cb)
        if ca is None or cb is None:
            print(f"  {key}: skipped -- one half had no usable data.")
            continue
        cmp = _compare_curves(ca, cb)
        result[key] = cmp
        print(
            f"  {key} ({label_a} vs {label_b}): overall corr={cmp['overall_corr']:.4f}  "
            f"mean|Δ|={cmp['overall_mean_abs_diff']:.4f}"
        )
        for day, d in cmp["per_day"].items():
            print(f"    {day}: corr={d['corr']:.4f}  mean|Δ|={d['mean_abs_diff']:.4f}")

    if active_window:
        start = int(active_window["weekday_active_utc"][0].split(":")[0])
        end = int(active_window["weekday_active_utc"][1].split(":")[0])
        print(
            f"\n  Active window generalization (UTC {start:02d}:00-{end:02d}:00, from full-sample curve):"
        )
        window_result = {}
        for key, (label_a, label_b, _, _) in splits.items():
            ca, cb = curves[key]
            if ca is None or cb is None:
                continue
            for label, c in ((label_a, ca), (label_b, cb)):
                inside, outside = _apply_window(c, bin_size_sec, start, end)
                ratio = inside / outside if outside > 0 else float("nan")
                print(
                    f"    {label}: inside={inside:.4f}  outside={outside:.4f}  ratio={ratio:.2f}x"
                )
                window_result[label] = {
                    "inside_mean": inside,
                    "outside_mean": outside,
                    "ratio": ratio,
                }
        result["active_window_generalization"] = window_result

    print("─────────────────────────────────────────────────────────────\n")
    return result


def plot_heatmap(
    scores: np.ndarray,
    bin_size_sec: int,
    title: str,
    output_path: Path,
    value_label: str = "score (0-1)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bins = scores.shape[1]
    bins_per_hour = 3600 // bin_size_sec

    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(
        scores,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_yticks(range(7))
    ax.set_yticklabels(DAY_NAMES)

    tick_positions = list(range(0, n_bins, 6 * bins_per_hour))
    tick_labels = [f"{(p // bins_per_hour):02d}h" for p in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Time of day (UTC)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value_label, fraction=0.02, pad=0.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOG.info(f"Heatmap saved to {output_path}")
