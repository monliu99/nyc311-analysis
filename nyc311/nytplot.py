"""NYT-style chart helpers; the rcParams live in nyt.mplstyle next to this file.

    import matplotlib.pyplot as plt
    import nytplot as nyt

    nyt.use()
    fig, ax = plt.subplots()
    nyt.area(ax, years, price, nyt.SAGE, label="Gas prices")
    nyt.format_y(ax, prefix="$")
    nyt.title(fig, "Average gas prices per gallon")
    nyt.note(fig, "Source: U.S. Energy Information Administration.")
    nyt.save(fig, "figs/gas.png")
"""
from pathlib import Path
from typing import NamedTuple

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

STYLE = Path(__file__).with_name("nyt.mplstyle")
SANS = "sans-serif"  # resolves to font.sans-serif in the style
NOTE_GRAY = "#767676"


class Hue(NamedTuple):
    line: str  # line, bars, series label
    fill: str  # area under the line
    band: str  # highlighted span under the line


def hue(line):
    """Build a Hue from a line color: fill is a 22% tint, band a 37% tint."""
    rgb = np.array(mcolors.to_rgb(line))
    tint = lambda a: mcolors.to_hex(1 - a * (1 - rgb))
    return Hue(mcolors.to_hex(rgb), tint(0.22), tint(0.37))


# The first two are sampled from the reference graphic; the rest are derived
# companions, in the same order as axes.prop_cycle.
SAGE = Hue("#6b817a", "#dee3e3", "#c5cfcd")
CORAL = Hue("#cf715c", "#f4e0da", "#eecfc6")
HUES = [SAGE, CORAL, hue("#5b7fa3"), hue("#c49a3a"), hue("#8a6d8f"), hue("#8c8c8c")]


def use():
    """Apply nyt.mplstyle to all subsequent figures."""
    plt.style.use(STYLE)


def _left(fig):
    # Constrained layout puts the tick labels w_pad from the figure edge.
    return plt.rcParams["figure.constrained_layout.w_pad"] / fig.get_figwidth()


def title(fig, text, **kw):
    """Bold sans title, flush left with the tick labels."""
    return fig.suptitle(text, x=_left(fig), ha="left", fontfamily=SANS, **kw)


def note(fig, text, **kw):
    """Small gray note/source line under the chart, flush left."""
    return fig.supxlabel(text, x=_left(fig), ha="left", color=NOTE_GRAY, **kw)


def panel_title(ax, text, **kw):
    """Bold sans title over one panel of a small multiple."""
    return ax.set_title(text, fontfamily=SANS, **kw)


def panel_label(ax, text, color, xy=(0.012, 0.05), **kw):
    """Bold series name inside the panel (bottom-left by default), no legend."""
    return ax.text(*xy, text, transform=ax.transAxes, ha="left", va="bottom",
                   color=color, fontfamily=SANS, fontweight="bold",
                   fontsize=12, zorder=3, **kw)


def end_label(ax, x, y, text, color, dy=0, **kw):
    """Label a line at its last point instead of using a legend.

    dy nudges the label vertically (points) when two line ends collide.
    """
    return ax.annotate(text, (np.asarray(x)[-1], np.asarray(y)[-1]),
                       xytext=(5, dy), textcoords="offset points",
                       ha="left", va="center", color=color, fontfamily=SANS,
                       fontweight="bold", annotation_clip=False, **kw)


def area(ax, x, y, color=SAGE, label=None, baseline=0):
    """Line with a same-hue fill down to the baseline; y axis pinned at it."""
    color = color if isinstance(color, Hue) else hue(color)
    fill = ax.fill_between(x, baseline, y, color=color.fill, linewidth=0, zorder=1)
    fill.sticky_edges.y.append(baseline)  # no margin below the baseline
    line, = ax.plot(x, y, color=color.line, zorder=2)
    if label:
        panel_label(ax, label, color.line)
    return line


def _num(v):
    a = np.asarray(v)
    return a.astype(float) if np.issubdtype(a.dtype, np.number) else mdates.date2num(v)


def highlight(ax, x, y, start, end, color=SAGE, baseline=0):
    """Darker shade under the curve from start to end (e.g. an administration)."""
    color = color if isinstance(color, Hue) else hue(color)
    xn, yn = _num(x), np.asarray(y, dtype=float)
    a, b = float(_num(start)), float(_num(end))
    xs = np.r_[a, xn[(xn > a) & (xn < b)], b]
    return ax.fill_between(xs, baseline, np.interp(xs, xn, yn),
                           color=color.band, linewidth=0, zorder=1.1)


class _TopUnitFormatter(mticker.Formatter):
    def __init__(self, prefix, suffix):
        self.prefix, self.suffix = prefix, suffix

    def __call__(self, x, pos=None):
        lo, hi = sorted(self.axis.get_view_interval())
        eps = 1e-9 * (hi - lo)
        shown = [t for t in self.locs if lo - eps <= t <= hi + eps]
        s = f"{x:,.12g}".replace("-", "\N{MINUS SIGN}")
        if x == 0 and lo >= -eps:
            return ""  # zero is the dark baseline, which speaks for itself
        if shown and x == max(shown):
            return f"{self.prefix}{s}{self.suffix}"
        return s


def _unit_ticks(axis, prefix, suffix, step, nbins):
    locator = (mticker.MultipleLocator(step) if step
               else mticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]))
    axis.set_major_locator(locator)
    axis.set_major_formatter(_TopUnitFormatter(prefix, suffix))


def format_y(ax, prefix="", suffix="", step=None, nbins=5):
    """Few round y ticks; unit ("$", "%") only on the top label; no 0 label."""
    _unit_ticks(ax.yaxis, prefix, suffix, step, nbins)


def format_x(ax, prefix="", suffix="", step=None, nbins=5):
    """format_y for the x axis (horizontal bars): unit on the rightmost label."""
    _unit_ticks(ax.xaxis, prefix, suffix, step, nbins)


def save(fig, path, **kw):
    """Save (300-dpi PNG by default; a .pdf/.svg path gives vector) and close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kw)
    plt.close(fig)
    return path
