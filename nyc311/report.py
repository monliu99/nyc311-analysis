from pathlib import Path

import pandas as pd


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
        with pd.option_context("display.width", 200, "display.max_columns", 50,
                               "display.max_colwidth", 60, "display.float_format", "{:,.2f}".format):
            self.add(df.head(max_rows).to_string(index=False))
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
