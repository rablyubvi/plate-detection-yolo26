from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model.")
    parser.add_argument("--model", default="runs/train/yolo26-plate/weights/best.pt")
    parser.add_argument("--data", default="configs/plate_dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-det", type=int, default=50, help="Maximum detections per image.")
    return parser


def resolve_repo_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(resolve_repo_path(args.model))
    val_args = dict(
        data=resolve_repo_path(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        max_det=args.max_det,
    )
    if args.device is not None:
        val_args["device"] = args.device

    model.val(**val_args)


if __name__ == "__main__":
    main()
