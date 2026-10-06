# Your writing for the final report

`python run_all.py` ends with a `report` stage that builds one self-contained `outputs/report.html`:
the executive summary (Q11) first, then one section per question in the brief's order, each with the
brief's question, **your prose from this folder**, then the computed findings and charts. The Q8
dashboard is embedded as a live map.

- One file per question: `q01.md` ... `q11.md` (q11 = the executive summary). Missing files are fine;
  that section then shows only the computed findings.
- Your files are never overwritten by a rerun, so write freely; the numbers in the computed findings
  below your text always come from the latest run.
- Formatting: `#`/`##` headings, `-` bullets, `1.` lists, **bold**, *italic*, `code`, [links](https://...),
  simple pipe tables. You can place a chart inside your text with `![Caption](figs/q05_top10.png)`
  (paths relative to `outputs/`).
- To rebuild only the report after editing your prose: `python run_all.py --stages report` (add the same
  `--data/--work/--out` you used for the run).
