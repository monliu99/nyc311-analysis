import os
import platform
import shutil
import sys
from importlib import metadata

from nyc311.report import Report

PACKAGES = ["duckdb", "pyarrow", "pandas", "numpy", "matplotlib", "statsmodels", "folium"]


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "MISSING"


def run(cfg):
    rep = Report("00_env", cfg.out_dir)
    rep.add(f"python {sys.version.split()[0]} on {platform.platform()}")
    rep.add("packages: " + ", ".join(f"{p} {_version(p)}" for p in PACKAGES))
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        mem_gb = float("nan")
    rep.add(f"cpus on node: {os.cpu_count()} | threads used: {cfg.threads} | RAM on node: {mem_gb:,.0f} GB "
            f"| DuckDB memory limit: {cfg.memory_limit or 'default (80% of RAM)'}")
    slurm = {k: v for k, v in os.environ.items() if k in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")}
    if slurm:
        rep.add("slurm: " + ", ".join(f"{k}={v}" for k, v in slurm.items()))
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cfg.work_dir).free / 1e9
    rep.add(f"data: {cfg.data_glob}", f"work dir: {cfg.work_dir.resolve()} ({free:,.0f} GB free)",
            f"out dir: {cfg.out_dir.resolve()}", f"stages: {', '.join(cfg.stages)}")
    return rep.emit()
