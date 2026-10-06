import base64

import pytest

from nyc311 import build_report

PNG = base64.b64decode(  # 1x1 transparent PNG
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


def test_report_is_the_last_stage():
    from nyc311.config import STAGE_MODULES, STAGES
    assert STAGES[-1] == "report"
    assert STAGE_MODULES["report"] == "nyc311.build_report"


def test_markdown_blocks_become_html(tmp_path):
    out = build_report.md_to_html(
        "## Findings\n\n- one **bold**\n  - nested *it*\n- two `code`\n\n1. first\n2. second\n\n"
        "A [link](https://example.com) here.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", tmp_path)
    assert "<h4>Findings</h4>" in out
    assert out.count("<ul>") == 2 and "<strong>bold</strong>" in out and "<em>it</em>" in out
    assert "<code>code</code>" in out and "<ol>" in out
    assert '<a href="https://example.com">link</a>' in out
    assert "<table>" in out and "<th>a</th>" in out and "<td>2</td>" in out


def test_markdown_escapes_html_and_keeps_snake_case(tmp_path):
    out = build_report.md_to_html("- <script>x</script> and out_of_file_order and 5 * 3 * 2\n", tmp_path)
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "out_of_file_order" in out
    assert "<em>" not in out


def test_figures_are_embedded_and_missing_ones_noted(tmp_path):
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "q05_top.png").write_bytes(PNG)
    out = build_report.md_to_html("![Top ten](figs/q05_top.png)\n\n![Gone](figs/q05_missing.png)\n", tmp_path)
    assert 'src="data:image/png;base64,' in out and "<figcaption>Top ten</figcaption>" in out
    assert "Figure not found" in out


@pytest.fixture()
def built(tmp_path):
    out, prose = tmp_path / "outputs", tmp_path / "prose"
    for d in (out / "answers", out / "figs", out / "dashboard", out / "reports", prose):
        d.mkdir(parents=True)
    (out / "answers" / "q11.md").write_text("# Q11 Executive summary\n\n- Finding one SUMMARYMARK\n")
    (out / "answers" / "q05.md").write_text("# Q5 Top complaint types\n\n- Heat leads. FINDINGMARK\n\n"
                                            "![Top ten types](figs/q05_top10.png)\n")
    (out / "figs" / "q05_top10.png").write_bytes(PNG)
    (out / "figs" / "q05_extra.png").write_bytes(PNG)
    (out / "dashboard" / "q08_dashboard.html").write_text("<html><body><div id='map'>MAPMARK</div></body></html>")
    (out / "reports" / "03_clean.txt").write_text("===== BEGIN 03_clean =====\nrows: 10\n===== END 03_clean =====")
    (prose / "q05.md").write_text("My own words PROSEMARK.\n")
    path = build_report.build(out, prose, meta={"title_note": "test data"})
    return path.read_text()


def test_summary_comes_first_then_questions_in_order(built):
    assert built.index("SUMMARYMARK") < built.index('id="q01"') < built.index('id="q05"') < built.index('id="q10"')


def test_prose_comes_before_computed_findings(built):
    assert built.index("PROSEMARK") < built.index("FINDINGMARK")


def test_report_is_self_contained(built):
    assert "<img" in built
    assert all(chunk.startswith('"data:image/png;base64,') for chunk in built.split("<img src=")[1:])
    assert built.count("data:image/png;base64,") == 2  # referenced figure plus the unreferenced q05_extra.png


def test_dashboard_is_embedded_as_a_live_map(built):
    assert "<iframe" in built and "srcdoc=" in built and "MAPMARK" in built


def test_missing_stages_say_so_and_brief_questions_appear(built):
    q01 = built[built.index('id="q01"'):built.index('id="q02"')]
    assert "did not produce results" in q01
    assert "top 10 complaint types" in built.lower()
    assert "rows: 10" in built
