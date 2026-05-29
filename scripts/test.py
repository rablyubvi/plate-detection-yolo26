from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test the fine-tuned YOLO model on the test split.")
    parser.add_argument("--model", default="runs/train/yolo26-plate/weights/best.pt")
    parser.add_argument("--data", default="configs/plate_dataset.yaml")
    parser.add_argument("--source", default="dataset/images/test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--device", default=None, help="CUDA device id, 'cpu', or omit for auto selection.")
    parser.add_argument("--project", default="runs/test")
    parser.add_argument("--name", default="yolo26-plate")
    parser.add_argument("--save-predictions", action="store_true", help="Save annotated predictions for test images.")
    parser.add_argument("--save-txt", action="store_true", help="Save prediction labels as YOLO txt files.")
    return parser


def resolve_repo_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def collect_image_paths(source_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(source_dir.glob(ext))
    return sorted(image_paths)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def profile_latency(
    model: YOLO,
    source_dir: Path,
    imgsz: int,
    conf: float,
    max_det: int,
    device: str | None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    image_paths = collect_image_paths(source_dir)

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            rows.append({
                "image": image_path.name,
                "num_detections": 0,
                "preprocess_ms": 0.0,
                "inference_ms": 0.0,
                "postprocess_ms": 0.0,
                "wall_time_ms": 0.0,
            })
            continue

        start = time.perf_counter()
        predict_args = dict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            max_det=max_det,
            verbose=False,
        )
        if device is not None:
            predict_args["device"] = device

        results = model.predict(**predict_args)
        wall_time_ms = (time.perf_counter() - start) * 1000.0

        speed = getattr(results[0], "speed", {}) if results else {}
        rows.append({
            "image": image_path.name,
            "num_detections": len(results[0].boxes) if results and results[0].boxes is not None else 0,
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "inference_ms": float(speed.get("inference", 0.0)),
            "postprocess_ms": float(speed.get("postprocess", 0.0)),
            "wall_time_ms": wall_time_ms,
        })

    total = len(rows)
    summary = {
        "images": float(total),
        "avg_preprocess_ms": sum(row["preprocess_ms"] for row in rows) / total if total else 0.0,
        "avg_inference_ms": sum(row["inference_ms"] for row in rows) / total if total else 0.0,
        "avg_postprocess_ms": sum(row["postprocess_ms"] for row in rows) / total if total else 0.0,
        "avg_wall_time_ms": sum(row["wall_time_ms"] for row in rows) / total if total else 0.0,
    }
    return rows, summary


def main() -> None:
    args = build_parser().parse_args()
    model = YOLO(resolve_repo_path(args.model))

    common_args = dict(
        imgsz=args.imgsz,
        device=args.device,
        project=resolve_repo_path(args.project),
        name=args.name,
        exist_ok=True,
    )
    if args.device is None:
        common_args.pop("device")

    metrics = model.val(
        data=resolve_repo_path(args.data),
        split="test",
        batch=args.batch,
        conf=args.conf,
        max_det=args.max_det,
        plots=True,
        **common_args,
    )

    print("\nTest metrics")
    print(f"mAP50:    {metrics.box.map50:.5f}")
    print(f"mAP50-95: {metrics.box.map:.5f}")
    print(f"Precision:{metrics.box.mp:.5f}")
    print(f"Recall:   {metrics.box.mr:.5f}")

    project_dir = Path(resolve_repo_path(args.project)) / args.name
    latency_rows, latency_summary = profile_latency(
        model=model,
        source_dir=Path(resolve_repo_path(args.source)),
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det,
        device=args.device,
    )
    latency_csv = project_dir / "latency.csv"
    write_csv(
        latency_csv,
        [
            "image",
            "num_detections",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "wall_time_ms",
        ],
        latency_rows,
    )

    print("\nLatency")
    print(f"Average preprocess:  {latency_summary['avg_preprocess_ms']:.3f} ms")
    print(f"Average inference:   {latency_summary['avg_inference_ms']:.3f} ms")
    print(f"Average postprocess: {latency_summary['avg_postprocess_ms']:.3f} ms")
    print(f"Average wall time:   {latency_summary['avg_wall_time_ms']:.3f} ms")
    print(f"Latency log saved to: {latency_csv}")

    if args.save_predictions:
        model.predict(
            source=resolve_repo_path(args.source),
            conf=args.conf,
            max_det=args.max_det,
            save=True,
            save_txt=args.save_txt,
            **common_args,
        )


if __name__ == "__main__":
    main()
