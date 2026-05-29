from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO26 model.")
    parser.add_argument("--model", default="yolo26n.pt", help="Base model or checkpoint path.")
    parser.add_argument("--data", default="configs/plate_dataset.yaml", help="YOLO dataset yaml.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="CUDA device id, 'cpu', or omit for auto selection.")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="yolo26-plate")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the model checkpoint.")
    parser.add_argument("--max-det", type=int, default=50, help="Maximum detections per image during validation.")
    parser.add_argument("--no-val", action="store_true", help="Skip validation during training to save memory.")
    parser.add_argument("--plots", action="store_true", help="Save training plots.")
    return parser


def resolve_repo_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def main() -> None:
    args = build_parser().parse_args()

    model_path = resolve_repo_path(args.model)
    model = YOLO(model_path if Path(model_path).exists() else args.model)
    train_args = dict(
        data=resolve_repo_path(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=resolve_repo_path(args.project),
        name=args.name,
        patience=args.patience,
        pretrained=True,
        plots=args.plots,
        save=True,
        exist_ok=True,
        optimizer="auto",
        amp=True,
        cache=args.cache,
        resume=args.resume,
        val=not args.no_val,
        max_det=args.max_det,
        seed=42,
    )
    if args.device is not None:
        train_args["device"] = args.device

    model.train(**train_args)


if __name__ == "__main__":
    main()
