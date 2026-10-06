from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

STAGE_MODULES = {
    "ingest": "nyc311.ingest",
    "discover": "nyc311.discover",
    "clean": "nyc311.clean",
    "q01": "nyc311.q01_missing",
    "q02": "nyc311.q02_timing",
    "q03": "nyc311.q03_complaint_types",
    "q04": "nyc311.q04_response_times",
    "q05": "nyc311.q05_top_types",
    "q06": "nyc311.q06_weather",
    "q07": "nyc311.q07_geography",
    "q08": "nyc311.q08_dashboard",
    "q09": "nyc311.q09_equity",
    "q10": "nyc311.q10_features",
    "q11": "nyc311.q11_summary",
}
STAGES = list(STAGE_MODULES)

MAX_DEFAULT_THREADS = 16
MEMORY_SHARE = 0.7  # of the job's allocation; DuckDB's own default is 80% of the whole node's RAM
CGROUP_LIMIT_FILES = [Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")]


def available_cpus():
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def cgroup_limit_bytes(paths=CGROUP_LIMIT_FILES):
    for path in paths:
        try:
            text = Path(path).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 2**60:
            return int(text)
    return None


def default_memory_limit():
    """A share of the job's memory allocation (SLURM, else cgroup); None if neither is known."""
    mb = os.environ.get("SLURM_MEM_PER_NODE")
    if not mb and os.environ.get("SLURM_MEM_PER_CPU"):
        mb = int(os.environ["SLURM_MEM_PER_CPU"]) * int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    total = int(mb) * 1e6 if mb else cgroup_limit_bytes()
    return f"{max(1, int(total * MEMORY_SHARE / 1e9))}GB" if total else None


@dataclass
class Config:
    data_glob: str
    out_dir: Path
    work_dir: Path
    threads: int
    memory_limit: str | None
    stages: list[str]

    @classmethod
    def from_args(cls, argv=None):
        p = argparse.ArgumentParser(description="NYC 311 lab pipeline")
        p.add_argument("--data", default=os.environ.get("NYC311_DATA", "data/sample/311_batch_*.json"),
                       help="glob for the batch JSON files")
        p.add_argument("--out", default="outputs", help="small outputs: reports, aggregates, figures")
        p.add_argument("--work", default="work", help="large intermediates: manifest, parquet, DuckDB spill")
        p.add_argument("--threads", type=int, default=None)
        p.add_argument("--memory-limit", default=None, help="DuckDB memory_limit, e.g. 48GB")
        p.add_argument("--stages", default=",".join(STAGES), help=f"comma list from {STAGES}")
        a = p.parse_args(argv)
        threads = a.threads or int(os.environ.get("SLURM_CPUS_PER_TASK") or min(available_cpus(), MAX_DEFAULT_THREADS))
        stages = [s.strip() for s in a.stages.split(",") if s.strip()]
        unknown = set(stages) - set(STAGES)
        if unknown:
            p.error(f"unknown stages: {sorted(unknown)}")
        return cls(a.data, Path(a.out), Path(a.work), threads, a.memory_limit or default_memory_limit(), stages)

    def connect(self):
        tmp = self.work_dir / "duckdb_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute(f"SET threads = {int(self.threads)}")
        con.execute(f"SET temp_directory = '{tmp.as_posix()}'")
        con.execute("SET preserve_insertion_order = false")
        if self.memory_limit:
            con.execute(f"SET memory_limit = '{self.memory_limit}'")
        return con
