import argparse
import json
import logging
import os
import yaml

import pymasq

from pymasq import optimizations as opts
from pymasq import datasets

pymasq.set_seed(123)

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_CFG_FNAME = os.path.join(ROOT_DIR, "core_config.yaml")
TEST_CFG_FNAME = os.path.join(ROOT_DIR, "test_config.yaml")

VERBOSE = True
MAX_ITERS = 1000000000
THRESHOLD = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Integration Test Runner")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="display additional print statements to the console",
    )
    parser.add_argument(
        "--test-config",
        dest="test_cfg",
        default=TEST_CFG_FNAME,
        help="config file for the new functionality under test (Default: %s)"
        % (TEST_CFG_FNAME),
    )
    parser.add_argument(
        "-i",
        "--iters",
        default=MAX_ITERS,
        help="specify the total number of iterations to run (Default: %i)"
        % (MAX_ITERS),
    )

    return parser.parse_args()


def get_configs(test_cfg):
    cfg, new_cfg = None, None

    with open(CORE_CFG_FNAME, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    with open(test_cfg, "r") as f:
        new_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # check new_cfg is not empty
    if new_cfg:
        for key, val in new_cfg.items():
            if key in cfg:
                if isinstance(cfg[key], list):
                    cfg[key].extend(val)
                elif isinstance(cfg[key], dict):
                    cfg[key].update(val)
            else:
                cfg[key] = val

    data_cfg = cfg["datasets"]
    mits_cfg = cfg.get("mitigations", [])
    mets_cfg = cfg.get("metrics", {})
    opts_cfg = cfg.get("optimizations", None)

    if VERBOSE:
        logger.info(
            f"""========== [ Dataset ] ==========\n,
            {json.dumps(data_cfg, indent=4)},
            "\n========== [Mitigations] ==========\n",
            {json.dumps(mits_cfg, indent=4)},
            "\n========== [Metrics] ==========\n",
            {json.dumps(mets_cfg, indent=4)},"""
        )

    return data_cfg, mits_cfg, mets_cfg, opts_cfg


def get_data(data_cfg):
    fname = data_cfg["data"]
    if ".csv" not in fname:
        fname += ".csv"

    df = datasets.load_data(fname)

    if data_cfg.get("dropnans", False):
        df = df.dropna().reset_index(drop=True)
    if data_cfg.get("nrows", False):
        df = df.iloc[0 : data_cfg["nrows"]]
    if data_cfg.get("cols", False):
        cols = data_cfg["cols"]
        df = df.loc[:, cols if isinstance(cols, list) else [cols]]

    if VERBOSE:
        logger.info(df, "\n", df.shape)

    return df


def run(args):
    global VERBOSE
    VERBOSE = args.verbose

    data_cfg, mits_cfg, mets_cfg, opts_cfg = get_configs(args.test_cfg)

    df = get_data(data_cfg)

    if opts_cfg:
        for opt, params in opts_cfg.items():
            func = getattr(opts, opt)
            if "iters" not in params:
                params["iteres"] = args.iters

            algo = func(target=df, mutations=mits_cfg, metrics=mets_cfg, **params)
            mod_df, fit, log = algo.optimize()

            if VERBOSE:
                logger.info("\n============== %s ===============\n" % (opt))
                logger.info(mod_df, "\n", fit, "\n", log)

    else:
        # if no optimizations specified, then simply run ExhaustiveSearch
        algo = opts.ExhaustiveSearch(
            target=df,
            mutations=mits_cfg,
            metrics=mets_cfg,
            iters=args.iters,
            theta=THRESHOLD,
            verbose=1,
        )
        mod_df, fit, log = algo.optimize()

        if VERBOSE:
            logger.info(mod_df, "\n", fit, "\n", log)

    logger.info("[Tests: Complete]")


if __name__ == "__main__":
    args = parse_args()
    run(args)
