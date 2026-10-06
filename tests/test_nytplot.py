import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nyc311 import nytplot as nyt  # noqa: E402


def test_top_unit_formatter_draws_without_deprecation_warnings():
    nyt.use()
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [10, 20, 30])
    nyt.format_y(ax, suffix="%")
    with warnings.catch_warnings():
        warnings.simplefilter("error", matplotlib.MatplotlibDeprecationWarning)
        fig.canvas.draw()
    lo, hi = ax.get_ylim()
    visible = [(t.get_position()[1], t.get_text()) for t in ax.get_yticklabels() if lo <= t.get_position()[1] <= hi]
    with_unit = [text for _, text in visible if text.endswith("%")]
    assert with_unit == [max(visible)[1]]
    plt.close(fig)
