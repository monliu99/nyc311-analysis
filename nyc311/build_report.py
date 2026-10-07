"""Final stage: one self-contained HTML report from every question's findings, figures, and the student's prose.

Inputs (all optional; whatever a run produced is included):
  outputs/answers/qNN.md        computed bullet findings written by each question module
  outputs/figs/qNN_*.png        charts (embedded as data URIs; referenced ones in place, the rest after the findings)
  outputs/dashboard/q08_dashboard.html   the Q8 web dashboard (embedded as a live iframe)
  report/prose/qNN.md           the student's own writing, placed before the computed findings; never overwritten

The page is styled to match the charts (nyt.mplstyle): Charter-style serif text, bold sans headings, the sage and
coral palette, gray notes. Everything is inline (CSS, a few lines of JS, base64 images); system fonts only, so the
report opens offline. Figures are numbered in reading order; tables right-align numeric columns.
"""
from __future__ import annotations

import base64
import html
import re
import struct
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
FINDINGS_LABEL = "What the data shows"

_IMG = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_LINK = re.compile(r"\[([^\]]+)\]\(((?:https?://|mailto:|#)[^)\s]*)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\w*])\*(?![\s*])([^*]+?)(?<!\s)\*(?![\w*])")
_CODE = re.compile(r"`([^`]+)`")
_LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
_SEP = re.compile(r"^\s*\|?\s*:?-{3,}")
# A table cell that reads as a number: 1,234 / 9.8% / -8% / $37k / 47.6 hours / 1.0x / 1,268,892 (4.5%)
_NUM = re.compile(r"^[+\-−±~≈<>]?\s*[$€£]?\d[\d,]*(?:\.\d+)?\s*"
                  r"(?:%|x|×|pp|points?|k|K|M|B|million|billion|hours?|days?|weeks?|months?|years?|min|minutes?)?"
                  r"(?:\s*\([+\-−]?\d[\d,]*(?:\.\d+)?\s*%?\))?$")
_EMPTY_CELLS = {"", "-", "–", "—", "n/a", "N/A", "NA", "na", "nan", "NaN", "none", "None"}
_FIG = re.compile(r"<figure([^>]*)>(.*?)<figcaption>", re.S)
_SOURCE = re.compile(r"\A<p>(<em>(?:(?!</em>|</p>).)*</em>)</p>", re.S)
_DPI_SCALE = 96 * 1.25  # CSS px per inch, enlarged so a chart's 10.5 pt labels read about as large as the body text

SERIF = "Charter,'Bitstream Charter','Sitka Text',Cambria,Georgia,serif"
SANS = "'Libre Franklin','Franklin Gothic Medium','Franklin Gothic','Helvetica Neue',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"

CSS = """
:root{color-scheme:light;--ink:#111;--text:#222;--muted:#555;--note:#767676;--rule:#e6e6e6;--base:#333;
--sage:#6b817a;--sage-ink:#4f6660;--sage-band:#c5cfcd;--sage-fill:#eef1f0;--coral:#cf715c;--coral-ink:#a8503d;
--serif:SERIF;--sans:SANS;--mono:MONO;--measure:38rem;--measure-data:36rem}
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
body{margin:0;background:#fff;color:var(--text);font:1rem/1.6 var(--serif);font-kerning:normal;
text-rendering:optimizeLegibility}
a{color:var(--sage-ink);text-decoration:underline;text-decoration-color:var(--sage-band);text-underline-offset:.15em}
a:hover{text-decoration-color:currentColor}
a:focus-visible{outline:2px solid var(--sage);outline-offset:2px;border-radius:1px}
code{font-family:var(--mono);font-size:.84em;background:var(--sage-fill);padding:.05em .3em;border-radius:2px}
pre{font:.75rem/1.5 var(--mono);background:#f6f7f6;padding:1rem 1.1rem;overflow-x:auto;margin:.5rem 0 1.5rem}
strong{font-weight:700;color:var(--ink)}
.skip{position:absolute;left:-999px;top:0;background:#fff;padding:.5rem .75rem;font:600 .875rem var(--sans)}
.skip:focus{left:1rem;top:1rem;z-index:10}
.print-only{display:none}
.missing{color:var(--coral-ink);font-family:var(--sans);font-size:.9375rem}

/* page frame: masthead over a contents rail and the report column */
.layout{display:grid;grid-template-columns:13.5rem minmax(0,66rem);grid-template-areas:". head" "toc main";
column-gap:clamp(2rem,4vw,4.5rem);max-width:88rem;margin:0 auto;padding:0 2rem 6rem}
.masthead{grid-area:head;padding:4rem 0 2.5rem;border-bottom:2px solid var(--base)}
.toc{grid-area:toc;position:sticky;top:0;align-self:start;max-height:100vh;overflow-y:auto;
padding:2.75rem .25rem 2rem 0;font:.8125rem/1.35 var(--sans)}
main{grid-area:main;min-width:0}

/* masthead */
h1{font:700 clamp(2.25rem,4.6vw,3.25rem)/1.04 var(--sans);letter-spacing:-.022em;color:var(--ink);margin:0 0 1rem}
.dek{font:1.3125rem/1.4 var(--serif);color:var(--muted);margin:0 0 1.25rem;max-width:var(--measure)}
.meta{font:.8125rem/1.5 var(--sans);color:var(--note);margin:0;max-width:var(--measure)}
.meta code{background:none;padding:0;font-size:.95em}

/* contents rail */
.toc-title{font:700 .8125rem/1 var(--sans);color:var(--ink);margin:0 0 .9rem;padding-left:calc(.7rem + 2px)}
.toc ol{list-style:none;margin:0;padding:0}
.toc a{display:grid;grid-template-columns:2.4rem 1fr;padding:.28rem 0 .28rem .7rem;
border-left:2px solid transparent;color:var(--muted);text-decoration:none}
.toc a:hover{color:var(--ink)}
.toc a[aria-current]{color:var(--ink);font-weight:700;border-left-color:var(--sage)}
.toc-n{color:var(--note);font-variant-numeric:tabular-nums}
.toc a[aria-current] .toc-n{color:var(--sage-ink)}
.toc-part{margin-top:1.1rem}
.toc-part>a,.toc-end>a{display:block;color:var(--ink);font-weight:700}
.toc-part>a{padding-bottom:.35rem}
.toc-end{margin-top:1.1rem}

/* text measure; figures, tables and the map use the full column */
main p,main ul,main ol,main h2,main h4,main h5,main h6,main blockquote{max-width:var(--measure)}
.findings p,.findings ul,.findings ol,.findings h4,.findings h5,.findings h6{max-width:var(--measure-data)}
main li>ul,main li>ol{max-width:none}
ul,ol{padding-left:1.25rem}
li{margin:.4rem 0}
li>ul,li>ol{margin:.35rem 0}
li::marker{color:var(--note)}

/* parts and questions; rules echo the charts: dark baseline, faint gridline */
.part{font:700 clamp(1.75rem,3.2vw,2.25rem)/1.12 var(--sans);letter-spacing:-.015em;color:var(--ink);
margin:6rem 0 0;padding-top:1rem;border-top:2px solid var(--base);scroll-margin-top:1.5rem;max-width:none}
.part-num{display:block;font-size:.875rem;letter-spacing:0;color:var(--sage-ink);margin-bottom:.45rem}
.question{margin-top:4.5rem;padding-top:2.75rem;border-top:1px solid var(--rule);scroll-margin-top:0}
.part+.question{margin-top:2.25rem;padding-top:0;border-top:0}
#summary{margin-top:2.75rem;padding-top:0;border-top:0}
.question>h2{font:700 1.75rem/1.2 var(--sans);letter-spacing:-.012em;color:var(--ink);margin:0 0 1.1rem}
.brief{font:.9375rem/1.55 var(--sans);color:var(--muted);margin:0 0 2rem;padding:.15rem 0 .15rem 1rem;
border-left:3px solid var(--sage-band)}
.brief strong{color:var(--ink);font-weight:600}

/* the student's narrative */
.prose{font-size:1.125rem;line-height:1.62;color:var(--ink)}
#summary .prose{font-size:1.1875rem}
.prose p{margin:0 0 1.05em}
.prose h3{font:700 1.3125rem/1.3 var(--sans);color:var(--ink);margin:2rem 0 .6rem}
.prose h4{font:700 1.125rem/1.3 var(--sans);color:var(--ink);margin:1.75rem 0 .5rem}
.prose li{margin:.55rem 0}

/* computed findings */
.findings{margin-top:2.5rem;border-top:2px solid var(--sage);padding-top:.75rem}
.findings-label{font:700 .875rem/1.2 var(--sans);color:var(--sage-ink);margin:0 0 1.25rem}
.findings p{margin:0 0 1em}
.findings p.source{font:italic .875rem/1.5 var(--serif);color:var(--note);margin-bottom:1.5rem}
.findings p.source em{font-style:italic}
.findings p.source code{background:none;padding:0;color:var(--muted)}
.findings h4{font:700 1.125rem/1.3 var(--sans);color:var(--ink);margin:2.25rem 0 .75rem}
.findings h5,.findings h6,.prose h5,.prose h6{font:700 1rem/1.3 var(--sans);color:var(--ink);margin:1.5rem 0 .5rem}

/* figures: numbered, captioned, wider than the text */
figure{margin:2.25rem 0 2.5rem;break-inside:avoid}
figure img{display:block;max-width:100%;height:auto}
figcaption{font:.875rem/1.45 var(--sans);color:var(--muted);margin-top:.75rem;padding-top:.6rem;
border-top:1px solid var(--rule)}
.fig-num{font-weight:700;color:var(--ink)}
figcaption code{background:none;padding:0}
figure.dashboard iframe{display:block;width:100%;height:min(78vh,720px);min-height:460px;border:1px solid var(--rule)}

/* tables: sans, tabular figures, numbers right-aligned */
.table-wrap{overflow-x:auto;margin:1.5rem 0 2rem;max-width:100%}
table{border-collapse:collapse;font:.875rem/1.4 var(--sans);font-variant-numeric:tabular-nums lining-nums;
color:var(--text)}
th,td{padding:.42rem .65rem;text-align:left;vertical-align:top}
th:first-child,td:first-child{padding-left:0}
th:last-child,td:last-child{padding-right:0}
thead th{font-weight:700;color:var(--ink);vertical-align:bottom;border-bottom:1.5px solid var(--base)}
td{border-bottom:1px solid var(--rule)}
.num{text-align:right;white-space:nowrap}
tbody tr:hover td{background:var(--sage-fill)}
td code,th code{background:none;padding:0;font-size:.92em}

/* appendix */
#appendix>h2{font:700 1.75rem/1.2 var(--sans);color:var(--ink);margin:0 0 1rem}
#appendix h4{font:700 1rem/1.3 var(--sans);color:var(--ink);margin:2rem 0 .5rem}
#appendix{margin-top:6rem;padding-top:1rem;border-top:2px solid var(--base)}

@media (max-width:64rem){
.layout{grid-template-columns:minmax(0,1fr);grid-template-areas:"head" "toc" "main";padding:0 1.25rem 4rem}
.masthead{padding-top:2.5rem}
.toc{position:static;max-height:none;overflow:visible;padding:1.5rem 0 1.75rem;border-bottom:1px solid var(--rule)}
.toc>ol{columns:2 16rem;column-gap:2.5rem}
.toc a{padding-top:.14rem;padding-bottom:.14rem}
.toc-part,.toc-end{break-inside:avoid;margin-top:.75rem}
}
@media (max-width:40rem){
.layout{padding:0 1rem 3rem}
.dek{font-size:1.125rem}
.part{margin-top:4rem}
.question{margin-top:3rem;padding-top:2rem}
.question>h2,#appendix>h2{font-size:1.5rem}
.prose,#summary .prose{font-size:1.0625rem}
ul,ol{padding-left:1.1rem}
figure{margin:1.75rem 0}
figure.dashboard iframe{height:70vh;min-height:380px}
}
@media print{
@page{margin:16mm 15mm}
html{font-size:9.5pt}
body{background:#fff}
.layout{display:block;max-width:none;padding:0}
.toc,.skip{display:none}
.masthead{padding-top:0}
.part{break-before:page;margin-top:0}
#appendix{break-before:page;margin-top:0}
.question{margin-top:2.5rem;padding-top:1.5rem}
h2,h3,h4,h5,.findings-label,.brief,.findings p.source{break-after:avoid}
figure,tr,.brief,pre,blockquote{break-inside:avoid}
figure img{width:auto;height:auto;max-width:100%;max-height:7.4in}
figure.dashboard iframe{display:none}
.print-only{display:block;border:1px dashed var(--sage-band);padding:1rem 1.1rem;font:.9375rem/1.5 var(--sans);
color:var(--muted);max-width:none}
a{color:inherit;text-decoration:none}
thead{display:table-header-group}
tbody tr:hover td{background:none}
pre{white-space:pre-wrap;overflow-wrap:anywhere}
.table-wrap{overflow:visible}
}
""".replace("SERIF", SERIF).replace("SANS", SANS).replace("MONO", MONO)

# Highlights the contents entry for the section under the reading line; without JS the rail still works as links.
TOC_JS = """
(function(){
  if(!('IntersectionObserver' in window))return;
  var links={},cur=null;
  document.querySelectorAll('.toc a[href^="#"]').forEach(function(a){links[a.getAttribute('href').slice(1)]=a;});
  var io=new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      var a=links[e.target.id];
      if(!e.isIntersecting||!a||a===cur)return;
      if(cur)cur.removeAttribute('aria-current');
      a.setAttribute('aria-current','location');cur=a;
    });
  },{rootMargin:'-25% 0px -74% 0px'});
  document.querySelectorAll('main section[id]').forEach(function(s){io.observe(s);});
})();
"""


def _display_size(data):
    """CSS width and height for a PNG saved with its dpi recorded (matplotlib does); None if unknown."""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    pos = 8
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind == b"pHYs" and length == 9:
            per_metre, _, unit = struct.unpack(">IIB", data[pos + 8:pos + 17])
            if unit != 1 or not per_metre:
                return None
            scale = _DPI_SCALE / (per_metre * 0.0254)
            return max(1, round(width * scale)), max(1, round(height * scale))
        if kind in (b"IDAT", b"IEND"):
            return None
        pos += 12 + length
    return None


def _figure(alt, rel, out_dir, embedded):
    path = (Path(out_dir) / html.unescape(rel)).resolve()
    if not path.is_file():
        return f'<p class="missing">Figure not found: {rel}</p>'
    embedded.add(path)
    raw = path.read_bytes()
    data = base64.b64encode(raw).decode("ascii")
    size = _display_size(raw)
    dims = f' width="{size[0]}" height="{size[1]}"' if size else ""
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{alt}"{dims} decoding="async">'
            f'<figcaption>{alt}</figcaption></figure>')


def _inline(text, out_dir, embedded):
    codes = []

    def stash(m):
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    s = html.escape(_CODE.sub(stash, text))
    s = _IMG.sub(lambda m: _figure(m.group(1), m.group(2), out_dir, embedded), s)
    s = _LINK.sub(r'<a href="\2">\1</a>', s)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITAL.sub(r"<em>\1</em>", s)
    return re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], s)


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


def numeric_columns(rows):
    """For each column, True if every filled cell reads as a number (blanks and dashes don't count either way)."""
    flags = []
    for j in range(max((len(r) for r in rows), default=0)):
        values = [re.sub(r"[*`]", "", r[j]).strip() for r in rows if j < len(r)]
        values = [v for v in values if v not in _EMPTY_CELLS]
        flags.append(bool(values) and all(_NUM.match(v) for v in values))
    return flags


def _table_html(header, rows, out_dir, embedded):
    numeric = numeric_columns(rows)

    def cell(tag, text, j):
        cls = ' class="num"' if j < len(numeric) and numeric[j] else ""
        return f"<{tag}{cls}>{_inline(text, out_dir, embedded)}</{tag}>"

    head = "".join(cell("th", c, j) for j, c in enumerate(header))
    body = "".join("<tr>" + "".join(cell("td", c, j) for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _continues_item(line):
    stripped = line.strip()
    return bool(stripped) and not _HEAD.match(line) and not stripped.startswith("|") and not _IMG.fullmatch(stripped)


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
            header, rows, i = _cells(line), [], i + 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            out.append(_table_html(header, rows, out_dir, embedded))
        elif _LIST.match(line):
            flush()
            items = []
            while i < len(lines):
                item = _LIST.match(lines[i])
                if item:
                    ws, marker, rest = item.groups()
                    items.append((len(ws), "ol" if marker[0].isdigit() else "ul", rest))
                elif _continues_item(lines[i]):  # a wrapped line belongs to the item above
                    indent, tag, rest = items[-1]
                    items[-1] = (indent, tag, f"{rest} {lines[i].strip()}")
                else:
                    break
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


def number_figures(body):
    """Number every <figure> in reading order: id="figure-N" and a "Figure N." lead-in on its caption."""
    count = 0

    def number(m):
        nonlocal count
        count += 1
        return (f'<figure id="figure-{count}"{m.group(1)}>{m.group(2)}'
                f'<figcaption><span class="fig-num">Figure {count}.</span> ')

    return _FIG.sub(number, body), count


def _strip_title(text):
    lines = text.splitlines()
    return "\n".join(lines[1:]) if lines and re.match(r"^#\s", lines[0]) else text


def _caption_from_stem(stem):
    """q05_top10_change -> 'Top10 change' (for charts no findings file references)."""
    words = re.sub(r"^q\d+_", "", stem).replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def _dashboard_figure(dash):
    return (f'<figure class="dashboard"><iframe title="Neighborhood dashboard (interactive map)" '
            f'srcdoc="{html.escape(dash.read_text(), quote=True)}"></iframe>'
            f'<p class="print-only">The interactive map does not print. Open '
            f'<code>{DASHBOARD.as_posix()}</code> in a browser to use it.</p>'
            f'<figcaption>Neighborhood dashboard, live. The map loads its mapping library and base map from the '
            f'internet, so it needs a connection; the same dashboard is also saved on its own as '
            f'<code>{DASHBOARD.as_posix()}</code>.</figcaption></figure>')


def _section(qid, title, brief, out_dir, prose_dir, anchor=None):
    out_dir = Path(out_dir)
    embedded = set()
    body = [f"<h2>{html.escape(title)}</h2>",
            f'<blockquote class="brief"><strong>The brief asks:</strong> {html.escape(brief)}</blockquote>']
    prose = Path(prose_dir) / f"{qid}.md"
    if prose.is_file() and prose.read_text().strip():
        body.append(f'<div class="prose">{md_to_html(prose.read_text(), out_dir, embedded)}</div>')
    answers = out_dir / "answers" / f"{qid}.md"
    figs = sorted((out_dir / "figs").glob(f"{qid}_*.png")) if (out_dir / "figs").is_dir() else []
    dash = out_dir / DASHBOARD if qid == "q08" else None
    findings = []
    if answers.is_file():
        # the modules open with an italic provenance line; set it as a source note
        findings.append(_SOURCE.sub(r'<p class="source">\1</p>', md_to_html(_strip_title(answers.read_text()),
                                                                             out_dir, embedded), count=1))
    for fig in figs:
        if fig.resolve() not in embedded:
            findings.append(_figure(html.escape(_caption_from_stem(fig.stem)), f"figs/{fig.name}", out_dir, embedded))
    if dash is not None and dash.is_file():
        findings.append(_dashboard_figure(dash))
    if not findings:
        findings.append('<p class="missing">This stage did not produce results in this run.</p>')
    body.append(f'<div class="findings"><h3 class="findings-label">{FINDINGS_LABEL}</h3>{"".join(findings)}</div>')
    return f'<section id="{anchor or qid}" class="question">{"".join(body)}</section>'


def _toc_link(anchor, number, label):
    return (f'<li><a href="#{anchor}"><span class="toc-n">{html.escape(number)}</span>'
            f'<span class="toc-t">{html.escape(label)}</span></a></li>')


def _split(title, pattern):
    m = re.match(pattern, title)
    return (m.group(1), m.group(2)) if m else ("", title)


def build(out_dir, prose_dir=PROSE_DIR, meta=None):
    out_dir, meta = Path(out_dir), meta or {}
    qid, title, brief = SUMMARY
    toc = [_toc_link("summary", "Q11", title)]
    sections = [_section(qid, title, brief, out_dir, prose_dir, anchor="summary")]
    for n, (part, questions) in enumerate(PARTS, 1):
        label, name = _split(part, r"^(Part \d+):\s*(.*)$")
        sections.append(f'<h2 class="part" id="part-{n}"><span class="part-num">{html.escape(label)}</span>'
                        f'{html.escape(name)}</h2>')
        items = []
        for qid, title, brief in questions:
            items.append(_toc_link(qid, *_split(title, r"^(Q\d+)\.\s*(.*)$")))
            sections.append(_section(qid, title, brief, out_dir, prose_dir))
        toc.append(f'<li class="toc-part"><a href="#part-{n}">{html.escape(part)}</a><ol>{"".join(items)}</ol></li>')
    toc.append('<li class="toc-end"><a href="#appendix">Appendix: data, cleaning, reproducibility</a></li>')
    body, _ = number_figures("".join(sections))
    appendix = []
    for name, label in [("03_clean", "Cleaning: rows and flags"), ("00_env", "Run environment"),
                        ("01_ingest", "Ingest")]:
        path = out_dir / "reports" / f"{name}.txt"
        if path.is_file():
            appendix.append(f"<h4>{label}</h4><pre>{html.escape(path.read_text())}</pre>")
    dek = f'<p class="dek">{html.escape(str(meta["coverage"]))}</p>' if meta.get("coverage") else ""
    note = f'<p class="meta">{html.escape(str(meta["title_note"]))}</p>' if meta.get("title_note") else ""
    generated = html.escape(meta.get("generated", datetime.now().strftime("%Y-%m-%d %H:%M")))
    source = (" from " + html.escape(meta["data"])) if meta.get("data") else ""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>NYC 311 Service Analytics</title>
<style>{CSS}</style></head><body>
<a class="skip" href="#report">Skip to the report</a>
<div class="layout">
<header class="masthead"><h1>NYC 311 Service Analytics</h1>{dek}{note}
<p class="meta">Generated {generated} by <code>python run_all.py</code>{source}.</p></header>
<nav class="toc" aria-label="Contents"><p class="toc-title">Contents</p><ol>{"".join(toc)}</ol></nav>
<main id="report">
{body}
<section id="appendix"><h2>Appendix: data, cleaning, reproducibility</h2>
<p>Every number above is computed by the pipeline in this folder; rerunning <code>python run_all.py</code> on the same
data regenerates this report. Cleaning decisions are documented in the README.</p>{"".join(appendix)}</section>
</main>
</div>
<script>{TOC_JS}</script>
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(doc)
    return path


def _long_date(d):
    """Jan 1, 2010: the date style the findings use."""
    return f"{d:%b} {d.day}, {d.year}" if hasattr(d, "day") else str(d)


def run(cfg):
    from nyc311.clean import clean_glob, clean_path

    meta = {"data": cfg.data_glob}
    if clean_path(cfg).exists():
        con = cfg.connect()
        n, lo, hi = con.sql(f"select count(*), min(created)::date, max(created)::date from {clean_glob(cfg)} "
                            f"where not is_test and not dup_key").fetchone()
        meta["coverage"] = f"{n:,} service requests opened {_long_date(lo)} to {_long_date(hi)}"
    path = build(cfg.out_dir, PROSE_DIR, meta)
    rep = Report("report", cfg.out_dir)
    answered = sorted(p.stem for p in (cfg.out_dir / "answers").glob("q*.md")) if (cfg.out_dir / "answers").is_dir() else []
    rep.add(f"wrote {path} ({path.stat().st_size / 1e6:,.1f} MB)",
            f"sections with computed findings: {', '.join(answered) or 'none'}",
            f"prose files found in {PROSE_DIR}: "
            f"{', '.join(sorted(p.stem for p in PROSE_DIR.glob('q*.md'))) or 'none yet'}")
    rep.emit()
