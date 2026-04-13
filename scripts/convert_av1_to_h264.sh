#!/usr/bin/env bash
# Safe batch converter: av1 (or other problematic codecs) -> H.264 (libx264)
# Usage: ./scripts/convert_av1_to_h264.sh [--dry-run] [target_dir]
# Default target_dir: datasets/robofac_sample_real/realworld_data

set -euo pipefail
IFS=$'\n\t'

DRY_RUN=false
TARGET_DIR="datasets/robofac_sample_real/realworld_data"

if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

if [[ $# -ge 1 ]]; then
  TARGET_DIR="$1"
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Target directory not found: $TARGET_DIR" >&2
  exit 2
fi

# Find mp4 files under target
mapfile -t files < <(find "$TARGET_DIR" -type f -name "*.mp4")

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .mp4 files found under $TARGET_DIR"
  exit 0
fi

echo "Found ${#files[@]} .mp4 files. Dry run: $DRY_RUN"

for f in "${files[@]}"; do
  # get codec info
  codec=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$f" || echo "unknown")
  echo "\nFile: $f"
  echo "  codec: $codec"

  # Skip if already h264
  if [[ "$codec" == "h264" ]]; then
    echo "  -> already h264, skipping"
    continue
  fi

  # Build temp output path next to original
  dir=$(dirname "$f")
  base=$(basename "$f")
  tmp="$dir/.tmp_convert_${base}"

  cmd=(ffmpeg -y -i "$f" -c:v libx264 -preset medium -crf 23 -c:a copy -movflags +faststart "$tmp")

  echo "  Planned command: ${cmd[*]}"

  if $DRY_RUN; then
    continue
  fi

  # Convert to tmp file
  if "${cmd[@]}"; then
    # preserve original modification time
    touch -r "$f" "$tmp"
    # atomic replace
    mv -f "$tmp" "$f"
    echo "  -> converted and replaced"
  else
    echo "  -> conversion failed for $f" >&2
    [[ -f "$tmp" ]] && rm -f "$tmp"
  fi

done

echo "Done."
