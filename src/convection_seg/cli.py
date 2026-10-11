"""`convseg` command line: predict | realtime | evaluate | serve | make-reference."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config


def _predict(args, cfg):
    from PIL import Image

    from .model import Segmenter
    from .realtime import segment_and_save

    seg = Segmenter.from_config(cfg)
    cfg["realtime"]["output_dir"] = args.out
    for p in args.images:
        rec = segment_and_save(seg, Image.open(p).convert("RGB"), Path(p).stem, cfg)
        print(f"{p}: coverage={rec['coverage']:.3f}")


def _realtime(args, cfg):
    from .realtime import run_loop

    run_loop(cfg, once=args.once)


def _evaluate(args, cfg):
    from .evaluate import dump, evaluate, quality_gate

    summary = evaluate(cfg, args.images, args.masks, args.limit, args.out_csv)
    print(dump(summary))
    if args.min_dice is not None and not quality_gate(summary, args.min_dice):
        sys.exit(f"Quality gate failed: mean Dice {summary['mean_dice']:.3f} < {args.min_dice}")


def _serve(args, cfg):
    import uvicorn

    uvicorn.run("convection_seg.api:app", host=args.host, port=args.port)


def _make_reference(args, cfg):
    import json

    from .monitoring import build_reference, read_log

    records = read_log(cfg["monitoring"]["log_file"])
    if len(records) < 10:
        sys.exit(f"Need at least 10 logged predictions, found {len(records)}")
    Path(cfg["monitoring"]["reference_file"]).write_text(json.dumps(build_reference(records), indent=2))
    print(f"Reference written from {len(records)} records")


def main(argv=None):
    p = argparse.ArgumentParser(prog="convseg", description=__doc__)
    p.add_argument("--config", help="path to YAML config")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("predict", help="segment local convection RGB images")
    s.add_argument("images", nargs="+")
    s.add_argument("--out", default="outputs/predict")
    s.set_defaults(fn=_predict)

    s = sub.add_parser("realtime", help="continuously segment the latest SEVIRI scan")
    s.add_argument("--once", action="store_true", help="run a single cycle and exit")
    s.set_defaults(fn=_realtime)

    s = sub.add_parser("evaluate", help="IoU/Dice vs expert masks")
    s.add_argument("--images", default="data/image")
    s.add_argument("--masks", default="data/mask_manual")
    s.add_argument("--limit", type=int)
    s.add_argument("--out-csv")
    s.add_argument("--min-dice", type=float, help="exit non-zero if mean Dice is lower")
    s.set_defaults(fn=_evaluate)

    s = sub.add_parser("serve", help="start the FastAPI service")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=_serve)

    s = sub.add_parser("make-reference", help="freeze drift reference stats from the monitoring log")
    s.set_defaults(fn=_make_reference)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.fn(args, load_config(args.config))


if __name__ == "__main__":
    main()
