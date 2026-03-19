#!/usr/bin/env bash
set -euo pipefail

# Rename dataset images in each class folder to sequential numbers:
#   001.jpg, 002.jpg, ... (or wider padding if needed)
#
# Default behavior only targets '*.jpg'.
# Use --include-temp to also recover and rename leftover temp files from prior attempts.
#
# Usage examples:
#   bash scripts/rename_dataset_images.sh --dry-run
#   bash scripts/rename_dataset_images.sh
#   bash scripts/rename_dataset_images.sh --include-temp --expected-per-class 500
#   bash scripts/rename_dataset_images.sh --root clothes_dataset --pad 3

ROOT="clothes_dataset"
DRY_RUN=0
INCLUDE_TEMP=0
PAD_WIDTH=0
EXPECTED_PER_CLASS=0
START_INDEX=1

usage() {
  cat <<'EOF'
Rename dataset images in each class folder to sequential JPG names.

Options:
  --root PATH                Dataset root directory (default: clothes_dataset)
  --dry-run                  Print planned actions, do not rename files
  --include-temp             Also include temporary files from interrupted rename attempts:
                             .ren_*.jpg, __ren_*.jpg, .rename_tmp_*.jpg,
                             .rename_stage2_*.jpg, *.renametmp
  --pad N                    Fixed zero-padding width for output names (e.g. 3 -> 001.jpg)
                             Default: auto width, min 3
  --start N                  Start numbering at N (default: 1)
  --expected-per-class N     Fail if any non-empty class folder does not have exactly N files
  -h, --help                 Show this help

Examples:
  bash scripts/rename_dataset_images.sh --dry-run
  bash scripts/rename_dataset_images.sh --include-temp --expected-per-class 500
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --include-temp)
      INCLUDE_TEMP=1
      shift
      ;;
    --pad)
      PAD_WIDTH="${2:-0}"
      shift 2
      ;;
    --start)
      START_INDEX="${2:-1}"
      shift 2
      ;;
    --expected-per-class)
      EXPECTED_PER_CLASS="${2:-0}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $ROOT" >&2
  exit 1
fi

if ! [[ "$PAD_WIDTH" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --pad must be a non-negative integer" >&2
  exit 1
fi

if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]] || [[ "$START_INDEX" -lt 1 ]]; then
  echo "[ERROR] --start must be an integer >= 1" >&2
  exit 1
fi

if ! [[ "$EXPECTED_PER_CLASS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --expected-per-class must be a non-negative integer" >&2
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[INFO] Dry run mode: no files will be renamed"
fi

total_renamed=0
processed_dirs=0

for class_dir in "$ROOT"/*; do
  [[ -d "$class_dir" ]] || continue

  if [[ "$INCLUDE_TEMP" -eq 1 ]]; then
    mapfile -t files < <(
      find "$class_dir" -maxdepth 1 -type f \( \
        -name '*.jpg' -o \
        -name '.ren_*.jpg' -o \
        -name '__ren_*.jpg' -o \
        -name '.rename_tmp_*.jpg' -o \
        -name '.rename_stage2_*.jpg' -o \
        -name '*.renametmp' \
      \) | sort
    )
  else
    mapfile -t files < <(find "$class_dir" -maxdepth 1 -type f -name '*.jpg' | sort)
  fi

  count=${#files[@]}
  if [[ "$count" -eq 0 ]]; then
    echo "[SKIP] $(basename "$class_dir"): 0 files"
    continue
  fi

  if [[ "$EXPECTED_PER_CLASS" -gt 0 ]] && [[ "$count" -ne "$EXPECTED_PER_CLASS" ]]; then
    echo "[ERROR] $(basename "$class_dir"): expected $EXPECTED_PER_CLASS files, found $count" >&2
    exit 1
  fi

  width=${#count}
  if [[ "$width" -lt 3 ]]; then
    width=3
  fi
  if [[ "$PAD_WIDTH" -gt 0 ]]; then
    width="$PAD_WIDTH"
  fi

  prefix=".seqtmp_${RANDOM}_$$_$(date +%s)"
  start="$START_INDEX"

  echo "[INFO] $(basename "$class_dir"): files=$count, width=$width, start=$start"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    first_dst=$(printf "%0*d.jpg" "$width" "$start")
    last_dst=$(printf "%0*d.jpg" "$width" "$((start + count - 1))")
    echo "       plan: $first_dst ... $last_dst"
    processed_dirs=$((processed_dirs + 1))
    total_renamed=$((total_renamed + count))
    continue
  fi

  i=0
  for src in "${files[@]}"; do
    seq=$((start + i))
    tmp=$(printf "%s/%s_%06d.renametmp" "$class_dir" "$prefix" "$seq")
    if [[ -e "$tmp" ]]; then
      echo "[ERROR] Temp collision: $tmp" >&2
      exit 1
    fi
    mv "$src" "$tmp"
    i=$((i + 1))
  done

  mapfile -t temp_files < <(find "$class_dir" -maxdepth 1 -type f -name "${prefix}_*.renametmp" | sort)
  if [[ "${#temp_files[@]}" -ne "$count" ]]; then
    echo "[ERROR] Temp file count mismatch in $(basename "$class_dir"): expected $count, got ${#temp_files[@]}" >&2
    exit 1
  fi

  i=0
  for src in "${temp_files[@]}"; do
    seq=$((start + i))
    dst=$(printf "%s/%0*d.jpg" "$class_dir" "$width" "$seq")
    mv "$src" "$dst"
    i=$((i + 1))
  done

  processed_dirs=$((processed_dirs + 1))
  total_renamed=$((total_renamed + count))
  echo "[DONE] $(basename "$class_dir"): renamed $count files"
done

echo "[SUMMARY] Processed dirs: $processed_dirs"
echo "[SUMMARY] Total files handled: $total_renamed"
