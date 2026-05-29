from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run plate detection inference.")
    parser.add_argument("--model", default="runs/train/yolo26-plate/weights/best.pt")
    parser.add_argument("--source", default="dataset/images/test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="yolo26-plate")
    return parser


def resolve_repo_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(resolve_repo_path(args.model))
    predict_args = dict(
        source=resolve_repo_path(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        project=resolve_repo_path(args.project),
        name=args.name,
        save=True,
        exist_ok=True,
    )
    if args.device is not None:
        predict_args["device"] = args.device

    model.predict(**predict_args)


if __name__ == "__main__":
    main()
