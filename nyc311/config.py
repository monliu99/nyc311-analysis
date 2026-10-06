import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

STAGES = ["ingest", "discover"]


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
        threads = a.threads or int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
        stages = [s.strip() for s in a.stages.split(",") if s.strip()]
        unknown = set(stages) - set(STAGES)
        if unknown:
            p.error(f"unknown stages: {sorted(unknown)}")
        return cls(a.data, Path(a.out), Path(a.work), threads, a.memory_limit, stages)

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
