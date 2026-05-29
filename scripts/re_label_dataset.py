from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT = ROOT / "dataset" / "labels"
SPLITS = ("train", "val", "test")


def relabel_file(label_path: Path) -> tuple[int, int]:
    changed = 0
    invalid = 0
    new_lines: list[str] = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            invalid += 1
            new_lines.append(line)
            print(f"Keep invalid line in {label_path}: {line}")
            continue

        if parts[0] != "0":
            changed += 1
        parts[0] = "0"
        new_lines.append(" ".join(parts))

    label_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    return changed, invalid


def main() -> None:
    total_files = 0
    total_changed = 0
    total_invalid = 0

    for split in SPLITS:
        label_dir = LABEL_ROOT / split
        for label_path in label_dir.glob("*.txt"):
            changed, invalid = relabel_file(label_path)
            total_files += 1
            total_changed += changed
            total_invalid += invalid

    print(f"Processed {total_files} label files")
    print(f"Changed {total_changed} annotations to class 0")
    print(f"Invalid lines kept unchanged: {total_invalid}")


if __name__ == "__main__":
    main()
