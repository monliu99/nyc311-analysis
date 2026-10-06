"""Single command for the whole lab: python run_all.py --data "/path/to/311_batch_*.json" """
import importlib
import importlib.util
import sys
import time

from nyc311 import bundle, config, env
from nyc311.config import Config


def run_stage(stage, cfg):
    name = config.STAGE_MODULES[stage]
    if importlib.util.find_spec(name) is None:
        print(f"[{stage}] {name} not built yet; skipping")
        return
    importlib.import_module(name).run(cfg)


def main(argv=None):
    cfg = Config.from_args(argv)
    env.run(cfg)
    for stage in cfg.stages:
        t0 = time.time()
        run_stage(stage, cfg)
        print(f"[{stage}] finished in {time.time() - t0:,.0f} s", flush=True)
    bundle.zip_outputs(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
