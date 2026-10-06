from pathlib import Path

import pandas as pd

TEXT_WIDTH = 90


def _text(v):
    """Cell text for non-numeric columns: SQL NULL / pandas NA shows as NULL, long text is cut."""
    try:
        missing = v is None or bool(pd.isna(v))
    except (TypeError, ValueError):  # lists, arrays
        missing = False
    return "NULL" if missing else str(v)[:TEXT_WIDTH]


class Report:
    """A labeled, length-capped text block: printed for copy-paste and saved to <out>/reports/."""

    def __init__(self, stage, out_dir, max_lines=150):
        self.stage = stage
        self.path = Path(out_dir) / "reports" / f"{stage}.txt"
        self.max_lines = max_lines
        self.lines = []

    def add(self, *lines):
        for line in lines:
            self.lines.extend(str(line).splitlines() or [""])

    def table(self, df, max_rows=25, title=None):
        if title:
            self.add("", f"## {title}")
        if df.empty:
            self.add("(none)")
            return
        shown = df.head(max_rows).copy()
        for c in shown.columns:
            if not pd.api.types.is_numeric_dtype(shown[c]) or pd.api.types.is_bool_dtype(shown[c]):
                text = [_text(v) for v in shown[c]]
                w = max([len(t) for t in text] + [len(str(c))])
                shown[c] = pd.Series([f"{t:<{w}}" for t in text], index=shown.index, dtype=object)
        with pd.option_context("display.width", 250, "display.max_columns", 50,
                               "display.max_colwidth", TEXT_WIDTH + 5, "display.float_format", "{:,.2f}".format):
            self.add(shown.to_string(index=False, justify="left", na_rep="NULL"))
        if len(df) > max_rows:
            self.add(f"({max_rows} of {len(df)} rows shown)")

    def emit(self):
        head, tail = f"===== BEGIN {self.stage} =====", f"===== END {self.stage} ====="
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join([head, *self.lines, tail]))
        body = self.lines
        if len(body) > self.max_lines:
            cut = len(body) - self.max_lines + 1
            body = body[: self.max_lines - 1] + [f"... {cut} more lines in reports/{self.path.name}"]
        text = "\n".join([head, *body, tail])
        print(text)
        return text
