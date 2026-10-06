"""Single command for the whole lab: python run_all.py --data "/path/to/311_batch_*.json" """
import sys

from nyc311 import bundle, discover, env, ingest
from nyc311.config import Config

STAGE_FUNCS = {"ingest": ingest.run, "discover": discover.run}


def main(argv=None):
    cfg = Config.from_args(argv)
    env.run(cfg)
    for stage in cfg.stages:
        STAGE_FUNCS[stage](cfg)
    bundle.zip_outputs(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
