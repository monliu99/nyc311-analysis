"""Final stage: one self-contained HTML report from every question's findings, figures, and the student's prose.

Inputs (all optional; whatever a run produced is included):
  outputs/answers/qNN.md        computed bullet findings written by each question module
  outputs/figs/qNN_*.png        charts (embedded as data URIs; referenced ones in place, the rest after the findings)
  outputs/dashboard/q08_dashboard.html   the Q8 web dashboard (embedded as a live iframe)
  report/prose/qNN.md           the student's own writing, placed before the computed findings; never overwritten
"""
import base64
import html
import re
from datetime import datetime
from pathlib import Path

from nyc311.report import Report

PROSE_DIR = Path(__file__).resolve().parent.parent / "report" / "prose"
DASHBOARD = Path("dashboard") / "q08_dashboard.html"

PARTS = [
    ("Part 1: Data quality assessment", [
        ("q01", "Q1. Missing values",
         "Identify which columns have missing values and calculate the percentage of missing data for each. This is "
         "not trivial as some columns may specify \"missingness\" in different ways. In addition, some entries may be "
         "blank for a reason, e.g., the column is not relevant for certain complaints."),
        ("q02", "Q2. Consistency of data collection over time",
         "a) Investigate daily volume of 311 requests. Identify any unusual patterns, gaps, or anomalies in the data "
         "collection. b) Now, investigate the times in which complaints were closed. Again, identify any unusual "
         "patterns, gaps, or anomalies. c) Create a graphics that the city can use to present information on 311 "
         "request openings and closings over time. The city is particularly interested in day-of-week and "
         "week-of-year effects."),
        ("q03", "Q3. Complaint categories",
         "a) Analyze the 'Complaint Type' field for data consistency issues (e.g., similar categories with different "
         "names, unusual entries, etc.). Provide specific examples. b) Propose a data cleaning strategy that would "
         "standardize these categories while preserving the granularity needed for city operations. Note that while "
         "it's great to handle all the data the city currently has, it would be even better if the strategy could be "
         "easily adapted when more data comes in."),
        ("q04", "Q4. Response times",
         "a) Calculate summary statistics for response times (if available) or time to closure. Include mean, median, "
         "standard deviation, and identify outliers. b) The Mayor wants to promise \"improved response times\" in the "
         "State of the City address. Are there particular neighborhoods or areas in the city which are experiencing "
         "longer than typical response times? c) Are there some agencies that are outperforming others in addressing "
         "complaints quickly? The Mayor wonders if longer response times is due to complaints being harder to handle. "
         "Conduct this analyze to inform the Mayor if some agencies are indeed offering exemplary services."),
    ]),
    ("Part 2: Data visualization and insights", [
        ("q05", "Q5. Top complaint types",
         "Create a visualization showing the top 10 complaint types by volume. Your visualization should be "
         "\"board-ready\" - clear enough for the Mayor to use directly in the presentation. Include appropriate "
         "titles, labels, and formatting."),
        ("q06", "Q6. Weather and complaints",
         "The Office of Emergency Management wants to understand seasonal patterns to better allocate resources. "
         "There is reason to believe that some of these complaints may be due to weather conditions. Give this a try "
         "and create an appropriate graphic(s) that addresses how complaints (their volume and their content) relates "
         "to weather conditions."),
        ("q07", "Q7. Geographic concentration and disparities",
         "Create a geographic visualization (this could be by borough, ZIP code, or community district) showing the "
         "concentration of complaints across NYC and any clear disparities in service requests between areas. What "
         "three key insights about geographic inequality would you highlight to the Mayor? These must be included in "
         "the graphic."),
        ("q08", "Q8. Neighborhood dashboard",
         "The Mayor's Office is particularly interested in developing a dashboard that allows staff to zoom in on an "
         "area and obtain details about complaints in that area. The office would like you to create a web-based "
         "graphic that achieves these goals."),
    ]),
    ("Part 3: Combined analysis and strategic recommendations", [
        ("q09", "Q9. Equity in city services",
         "Combine your geographic analysis with complaint types to identify: which neighborhoods have the highest "
         "volume of unresolved complaints; whether certain types of complaints are concentrated in specific areas; "
         "whether complaints are originating from the city's actions, actions of people in the neighborhood, natural, "
         "or for other reasons; any patterns that suggest inequitable service delivery. Present your findings in a "
         "single, compelling visualization with accompanying insights."),
        ("q10", "Q10. Features for predicting complaint volume",
         "What features (variables) would be most valuable for prediction? Why? Test to see if these variables are "
         "correlated with response times."),
    ]),
]
SUMMARY = ("q11", "Executive summary",
           "You have 5 minutes with the Mayor tomorrow morning. Prepare a concise executive summary that includes: "
           "three most critical findings from your analysis; two immediate action items; one long-term strategic "
           "recommendation. Focus on insights that are actionable and feasible.")

_IMG = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_LINK = re.compile(r"\[([^\]]+)\]\(((?:https?://|mailto:|#)[^)\s]*)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\w*])\*(?![\s*])([^*]+?)(?<!\s)\*(?![\w*])")
_CODE = re.compile(r"`([^`]+)`")
_LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
_SEP = re.compile(r"^\s*\|?\s*:?-{3,}")

CSS = """
body{font-family:Georgia,'Times New Roman',serif;color:#222;max-width:960px;margin:2rem auto;padding:0 1.25rem;line-height:1.55}
h1,h2,h3,h4,h5,nav,figcaption,table,.meta{font-family:-apple-system,'Helvetica Neue',Arial,sans-serif}
h1{font-size:2rem;margin-bottom:.2rem}h2{margin-top:2.5rem;border-top:1px solid #ddd;padding-top:1.2rem}
.meta{color:#767676;font-size:.9rem}.part{font-size:1.4rem;color:#555;margin-top:3rem}
blockquote.brief{color:#555;font-size:.92rem;border-left:3px solid #c5cfcd;margin:0 0 1rem;padding:.2rem .9rem}
figure{margin:1.2rem 0}figure img{max-width:100%;height:auto}figcaption{color:#767676;font-size:.85rem}
table{border-collapse:collapse;font-size:.88rem;margin:1rem 0}th,td{border-bottom:1px solid #e3e3e3;padding:.25rem .6rem;text-align:left}
code{font-size:.88em;background:#f4f4f4;padding:0 .2em}pre{background:#f7f7f7;padding:.8rem;overflow-x:auto;font-size:.8rem}
.missing{color:#cf715c}.prose{margin-bottom:1rem}nav ol{columns:2;font-size:.92rem}
iframe.dashboard{width:100%;height:640px;border:1px solid #ddd}
"""


def _figure(alt, rel, out_dir, embedded):
    path = (Path(out_dir) / html.unescape(rel)).resolve()
    if not path.is_file():
        return f'<p class="missing">Figure not found: {rel}</p>'
    embedded.add(path)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{alt}">'
            f'<figcaption>{alt}</figcaption></figure>')


def _inline(text, out_dir, embedded):
    parts = _CODE.split(text)
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(f"<code>{html.escape(part)}</code>")
            continue
        s = html.escape(part)
        s = _IMG.sub(lambda m: _figure(m.group(1), m.group(2), out_dir, embedded), s)
        s = _LINK.sub(r'<a href="\2">\1</a>', s)
        s = _BOLD.sub(r"<strong>\1</strong>", s)
        s = _ITAL.sub(r"<em>\1</em>", s)
        out.append(s)
    return "".join(out)


def _list_html(items, out_dir, embedded):
    out, stack = [], []
    for indent, tag, text in items:
        if not stack or indent > stack[-1][0]:
            out.append(f"<{tag}>")
            stack.append((indent, tag))
        else:
            while len(stack) > 1 and indent < stack[-1][0]:
                out.append(f"</li></{stack[-1][1]}>")
                stack.pop()
            out.append("</li>")
        out.append(f"<li>{_inline(text, out_dir, embedded)}")
    while stack:
        out.append(f"</li></{stack[-1][1]}>")
        stack.pop()
    return "".join(out)


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_html(text, out_dir, embedded=None):
    """The small markdown subset the question modules write: headings, lists, tables, emphasis, code, links, images."""
    embedded = set() if embedded is None else embedded
    lines, out, para, i = text.splitlines(), [], [], 0

    def flush():
        if para:
            out.append(f"<p>{_inline(' '.join(para), out_dir, embedded)}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
            i += 1
        elif _HEAD.match(line):
            flush()
            hashes, title = _HEAD.match(line).groups()
            level = min(len(hashes) + 2, 6)
            out.append(f"<h{level}>{_inline(title, out_dir, embedded)}</h{level}>")
            i += 1
        elif line.lstrip().startswith("|") and i + 1 < len(lines) and _SEP.match(lines[i + 1]):
            flush()
            head = "".join(f"<th>{_inline(c, out_dir, embedded)}</th>" for c in _cells(line))
            rows, i = [], i + 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append("<tr>" + "".join(f"<td>{_inline(c, out_dir, embedded)}</td>" for c in _cells(lines[i]))
                            + "</tr>")
                i += 1
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")
        elif _LIST.match(line):
            flush()
            items = []
            while i < len(lines) and _LIST.match(lines[i]):
                ws, marker, rest = _LIST.match(lines[i]).groups()
                items.append((len(ws), "ol" if marker[0].isdigit() else "ul", rest))
                i += 1
            out.append(_list_html(items, out_dir, embedded))
        elif _IMG.fullmatch(line.strip()):
            flush()
            out.append(_inline(line.strip(), out_dir, embedded))
            i += 1
        else:
            para.append(line.strip())
            i += 1
    flush()
    return "\n".join(out)


def _strip_title(text):
    lines = text.splitlines()
    return "\n".join(lines[1:]) if lines and re.match(r"^#\s", lines[0]) else text


def _section(qid, title, brief, out_dir, prose_dir, heading="h2"):
    out_dir = Path(out_dir)
    embedded = set()
    body = [f'<blockquote class="brief"><strong>The brief asks:</strong> {html.escape(brief)}</blockquote>']
    prose = Path(prose_dir) / f"{qid}.md"
    if prose.is_file() and prose.read_text().strip():
        body.append(f'<div class="prose">{md_to_html(prose.read_text(), out_dir, embedded)}</div>')
    answers = out_dir / "answers" / f"{qid}.md"
    figs = sorted((out_dir / "figs").glob(f"{qid}_*.png")) if (out_dir / "figs").is_dir() else []
    dash = out_dir / DASHBOARD if qid == "q08" else None
    if answers.is_file():
        body.append(md_to_html(_strip_title(answers.read_text()), out_dir, embedded))
    for fig in figs:
        if fig.resolve() not in embedded:
            body.append(_figure(html.escape(fig.stem.replace("_", " ")), f"figs/{fig.name}", out_dir, embedded))
    if dash is not None and dash.is_file():
        body.append(f'<iframe class="dashboard" title="Neighborhood dashboard" '
                    f'srcdoc="{html.escape(dash.read_text(), quote=True)}"></iframe>'
                    f'<p class="meta">Needs an internet connection to load the map tiles. The same dashboard is '
                    f'also saved on its own as <code>{DASHBOARD.as_posix()}</code>.</p>')
    if not answers.is_file() and not figs and not (dash is not None and dash.is_file()):
        body.append('<p class="missing">This stage did not produce results in this run.</p>')
    return f'<section id="{qid if heading == "h2" else "summary"}"><{heading}>{html.escape(title)}</{heading}>' \
           f'{"".join(body)}</section>'


def build(out_dir, prose_dir=PROSE_DIR, meta=None):
    out_dir, meta = Path(out_dir), meta or {}
    toc = ['<li><a href="#summary">Executive summary</a></li>']
    sections = [_section(*SUMMARY, out_dir, prose_dir, heading="h2").replace('id="q11"', 'id="summary"', 1)]
    for part, questions in PARTS:
        sections.append(f'<h2 class="part">{html.escape(part)}</h2>')
        for qid, title, brief in questions:
            toc.append(f'<li><a href="#{qid}">{html.escape(title)}</a></li>')
            sections.append(_section(qid, title, brief, out_dir, prose_dir))
    appendix = []
    for name, label in [("03_clean", "Cleaning: rows and flags"), ("00_env", "Run environment"),
                        ("01_ingest", "Ingest")]:
        path = out_dir / "reports" / f"{name}.txt"
        if path.is_file():
            appendix.append(f"<h4>{label}</h4><pre>{html.escape(path.read_text())}</pre>")
    toc.append('<li><a href="#appendix">Appendix: data, cleaning, reproducibility</a></li>')
    subtitle = " | ".join(html.escape(str(v)) for v in [meta.get("coverage"), meta.get("title_note")] if v)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>NYC 311 Service Analytics</title>
<style>{CSS}</style></head><body>
<h1>NYC 311 Service Analytics</h1>
<p class="meta">{subtitle}</p>
<p class="meta">Generated {html.escape(meta.get("generated", datetime.now().strftime("%Y-%m-%d %H:%M")))} by
<code>python run_all.py</code>{(" from " + html.escape(meta["data"])) if meta.get("data") else ""}.</p>
<nav><ol>{"".join(toc)}</ol></nav>
{"".join(sections)}
<section id="appendix"><h2>Appendix: data, cleaning, reproducibility</h2>
<p>Every number above is computed by the pipeline in this folder; rerunning <code>python run_all.py</code> on the same
data regenerates this report. Cleaning decisions are documented in the README.</p>{"".join(appendix)}</section>
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(doc)
    return path


def run(cfg):
    from nyc311.clean import clean_glob, clean_path

    meta = {"data": cfg.data_glob}
    if clean_path(cfg).exists():
        con = cfg.connect()
        n, lo, hi = con.sql(f"select count(*), min(created)::date, max(created)::date from {clean_glob(cfg)} "
                            f"where not is_test and not dup_key").fetchone()
        meta["coverage"] = f"{n:,} service requests opened {lo} to {hi}"
    path = build(cfg.out_dir, PROSE_DIR, meta)
    rep = Report("report", cfg.out_dir)
    answered = sorted(p.stem for p in (cfg.out_dir / "answers").glob("q*.md")) if (cfg.out_dir / "answers").is_dir() else []
    rep.add(f"wrote {path} ({path.stat().st_size / 1e6:,.1f} MB)",
            f"sections with computed findings: {', '.join(answered) or 'none'}",
            f"prose files found in {PROSE_DIR}: "
            f"{', '.join(sorted(p.stem for p in PROSE_DIR.glob('q*.md'))) or 'none yet'}")
    rep.emit()
