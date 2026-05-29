#!/usr/bin/env python3
"""
Demo training script for YOLO26 plate detection with 3 epochs.
Optimized for quick training with reduced batch size and image size.
"""

from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Load model
    model_path = ROOT / "yolo26n.pt"
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))

    # Training configuration for demo
    train_args = dict(
        data=str(ROOT / "configs" / "plate_dataset.yaml"),
        epochs=3,  # Demo: only 3 epochs
        imgsz=416,  # Smaller image size for faster training
        batch=8,   # Smaller batch size for faster training
        workers=4,
        project=str(ROOT / "runs" / "train"),
        name="yolo26-plate-demo",
        patience=10,
        pretrained=True,
        plots=True,
        save=True,
        exist_ok=True,
        optimizer="auto",
        amp=True,
        cache=False,
        resume=False,
        seed=42,
    )

    print("\n" + "="*60)
    print("YOLO26 Plate Detection - Demo Training (3 Epochs)")
    print("="*60)
    print(f"Model: yolo26n.pt")
    print(f"Epochs: {train_args['epochs']}")
    print(f"Image Size: {train_args['imgsz']}")
    print(f"Batch Size: {train_args['batch']}")
    print(f"Output: {train_args['project']}/{train_args['name']}")
    print("="*60 + "\n")

    # Train model
    model.train(**train_args)


if __name__ == "__main__":
    main()
