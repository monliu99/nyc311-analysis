"""Stage 0: what the run is working with (Python, packages, CPUs, memory, disk, the files DATA matches)."""
import multiprocessing
import os
import platform
import shutil
import sys
from importlib import metadata

from nyc311.report import Report

PACKAGES = ["duckdb", "pyarrow", "pandas", "numpy", "matplotlib", "statsmodels", "folium"]
# WORK holds the raw Parquet copy, the clean Parquet copy and DuckDB's spill files. Measured: the raw copy is
# 4-12% of the JSON size; half the JSON size leaves room for everything.
WORK_SHARE_OF_JSON = 0.5


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "MISSING"


def _cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def run(cfg):
    from nyc311 import ingest

    rep = Report("00_env", cfg.out_dir)
    rep.add(f"python {sys.version.split()[0]} on {platform.platform()}")
    rep.add("packages: " + ", ".join(f"{p} {_version(p)}" for p in PACKAGES))
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        mem_gb = float("nan")
    rep.add(f"cpus on node: {os.cpu_count()} | CPUs this job may use: {_cpus()} | threads used: {cfg.threads} "
            f"| RAM on node: {mem_gb:,.0f} GB | DuckDB memory limit: {cfg.memory_limit or 'default (80% of RAM)'}")
    rep.add(f"process start method: {multiprocessing.get_start_method(allow_none=True) or 'default'} "
            f"(platform default {multiprocessing.get_context().get_start_method()})")
    slurm = {k: v for k, v in os.environ.items()
             if k in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU")}
    if slurm:
        rep.add("slurm: " + ", ".join(f"{k}={v}" for k, v in sorted(slurm.items())))
    if cfg.threads > _cpus():
        rep.add(f"WARNING: {cfg.threads} threads but only {_cpus()} CPUs available; pass --threads {_cpus()}")

    try:
        files = ingest.match_files(cfg.data_glob)
    except ValueError as e:
        files = []
        rep.add(f"WARNING: {e}")
    json_gb = sum(os.path.getsize(f) for f in files) / 1e9
    rep.add(f"data: {cfg.data_glob}")
    if files:
        rep.add(f"files matching DATA: {len(files):,} ({json_gb:,.2f} GB; "
                f"{os.path.basename(files[0])} .. {os.path.basename(files[-1])})")
    else:
        rep.add("WARNING: no files match DATA; check the path (use an absolute path; ~ is expanded)")
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cfg.work_dir).free / 1e9
    rep.add(f"work dir: {cfg.work_dir.resolve()} ({free:,.1f} GB free)", f"out dir: {cfg.out_dir.resolve()}",
            f"stages: {', '.join(cfg.stages)}")
    need = WORK_SHARE_OF_JSON * json_gb
    if free < need:
        rep.add(f"WARNING: WORK has {free:,.1f} GB free; plan for about {need:,.1f} GB (Parquet copies + DuckDB "
                f"spill). Point --work at a folder with more space")
    return rep.emit()
