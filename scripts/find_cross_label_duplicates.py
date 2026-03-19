#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageRecord:
    path: Path
    label: str
    size_bytes: int
    sha256: str


def iter_images(root: Path):
    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for p in sorted(label_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
                yield label, p


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()





def build_records(root: Path) -> Tuple[List[ImageRecord], List[Tuple[str, str]]]:
    records: List[ImageRecord] = []
    skipped: List[Tuple[str, str]] = []

    for label, path in iter_images(root):
        try:
            size_bytes = path.stat().st_size
            sha = file_sha256(path)
            records.append(
                ImageRecord(
                    path=path,
                    label=label,
                    size_bytes=size_bytes,
                    sha256=sha,
                )
            )
        except (UnidentifiedImageError, OSError, ValueError) as e:
            skipped.append((str(path), str(e)))

    return records, skipped


def group_by(records: List[ImageRecord], key: str) -> Dict[str, List[ImageRecord]]:
    grouped: Dict[str, List[ImageRecord]] = {}
    for rec in records:
        grouped.setdefault(getattr(rec, key), []).append(rec)
    return grouped


def has_cross_label(group: List[ImageRecord]) -> bool:
    labels = {r.label for r in group}
    return len(labels) > 1


def write_reports(
    out_dir: Path,
    root: Path,
    exact_cross_label_groups: List[List[ImageRecord]],
    skipped: List[Tuple[str, str]],
)-> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "dataset_root": str(root),
        "exact_cross_label_group_count": len(exact_cross_label_groups),
        "exact_cross_label_groups": [
            {
                "items": [str(r.path.relative_to(root)) for r in sorted(g, key=lambda x: str(x.path))],
            }
            for g in exact_cross_label_groups
        ],
        "skipped_files": [{"path": p, "error": e} for p, e in skipped],
    }

    json_path = out_dir / "duplicate_report.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    return json_path


def move_or_delete_exact_duplicates(
    root: Path,
    exact_cross_label_groups: List[List[ImageRecord]],
    action: str,
    quarantine_dir: Path,
):
    moved_or_deleted = 0
    for group in exact_cross_label_groups:
        sorted_group = sorted(group, key=lambda x: str(x.path))
        keep = sorted_group[0]

        for rec in sorted_group[1:]:
            if action == "move":
                rel = rec.path.relative_to(root)
                target = quarantine_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(rec.path), str(target))
            elif action == "delete":
                rec.path.unlink(missing_ok=False)
            moved_or_deleted += 1

    return moved_or_deleted


def _parse_delete_list(delete_list_path: Path) -> List[str]:
    lines = delete_list_path.read_text(encoding="utf-8").splitlines()
    items: List[str] = []
    seen = set()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        normalized = line.replace("\\", "/")
        if normalized in seen:
            continue
        seen.add(normalized)
        items.append(normalized)

    return items


def _safe_resolve_under_root(root: Path, rel_path: str) -> Optional[Path]:
    rel = Path(rel_path)
    if rel.is_absolute():
        return None

    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def delete_from_list(root: Path, delete_list_path: Path, dry_run: bool) -> Tuple[int, int, int, int]:
    entries = _parse_delete_list(delete_list_path)
    deleted = 0
    missing = 0
    invalid = 0
    skipped_non_image = 0

    for rel_path in entries:
        resolved = _safe_resolve_under_root(root, rel_path)
        if resolved is None:
            print(f"[INVALID] {rel_path} (path escapes dataset root or absolute path)")
            invalid += 1
            continue

        if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
            print(f"[SKIP] {rel_path} (unsupported extension)")
            skipped_non_image += 1
            continue

        if not resolved.exists():
            print(f"[MISSING] {rel_path}")
            missing += 1
            continue

        if not resolved.is_file():
            print(f"[INVALID] {rel_path} (not a file)")
            invalid += 1
            continue

        if dry_run:
            print(f"[DRY-RUN] delete {rel_path}")
        else:
            resolved.unlink(missing_ok=False)
            print(f"[DELETED] {rel_path}")
        deleted += 1

    return len(entries), deleted, missing, invalid + skipped_non_image


def main():
    parser = argparse.ArgumentParser(
        description="Find duplicated images across different labels inside clothes_dataset."
    )
    parser.add_argument("--root", type=Path, default=Path("clothes_dataset"), help="Dataset root")
    parser.add_argument(
        "--action",
        choices=["report", "move", "delete"],
        default="report",
        help="report: only write reports; move/delete operate on exact cross-label duplicates",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=Path("duplicate_quarantine"),
        help="Where duplicates are moved when --action move",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required for --action delete",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("duplicate_reports"),
        help="Report output directory",
    )
    parser.add_argument(
        "--delete-list",
        type=Path,
        default=None,
        help="Path to a text file containing relative image paths to delete, one per line (e.g. Jaket/075.jpg)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without changing files (for --delete-list mode)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    if args.action == "delete" and not args.yes:
        raise ValueError("Refusing to delete without --yes")

    if args.delete_list is not None:
        delete_list_path = args.delete_list.resolve()
        if not delete_list_path.exists() or not delete_list_path.is_file():
            raise FileNotFoundError(f"Delete list not found: {delete_list_path}")

        if not args.dry_run and not args.yes:
            raise ValueError("Refusing to delete from list without --yes. Use --dry-run first to preview.")

        print(f"Deleting from list: {delete_list_path}")
        total, deleted, missing, invalid = delete_from_list(
            root=root,
            delete_list_path=delete_list_path,
            dry_run=args.dry_run,
        )
        print(f"[SUMMARY] entries={total} deleted={deleted} missing={missing} invalid_or_skipped={invalid}")
        return

    print(f"Scanning images under: {root}")
    records, skipped = build_records(root)
    print(f"Indexed files: {len(records)}")
    if skipped:
        print(f"Skipped files: {len(skipped)}")

    sha_groups = [g for g in group_by(records, "sha256").values() if len(g) > 1]
    exact_cross_label_groups = [g for g in sha_groups if has_cross_label(g)]

    json_path = write_reports(
        out_dir=args.out_dir,
        root=root,
        exact_cross_label_groups=exact_cross_label_groups,
        skipped=skipped,
    )

    print(f"Exact duplicate groups across labels: {len(exact_cross_label_groups)}")
    print(f"Report JSON: {json_path}")

    if args.action in {"move", "delete"}:
        changed = move_or_delete_exact_duplicates(
            root=root,
            exact_cross_label_groups=exact_cross_label_groups,
            action=args.action,
            quarantine_dir=args.quarantine_dir,
        )
        print(f"Action={args.action} applied on {changed} files (exact cross-label duplicates only)")


if __name__ == "__main__":
    main()
