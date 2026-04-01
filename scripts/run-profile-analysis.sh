#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <profile.sqlite|profile.nsys-rep> [gpu_id]" >&2
  exit 1
fi

caller_pwd="$(pwd)"
profile_input="$1"
gpu_id="${2:-}"

case "$profile_input" in
  /*) profile_path="$profile_input" ;;
  *) profile_path="$caller_pwd/$profile_input" ;;
esac

if [[ ! -f "$profile_path" ]]; then
  echo "Profile not found: $profile_path" >&2
  exit 1
fi

case "$profile_path" in
  *.sqlite|*.sqlite3|*.nsys-rep) ;;
  *)
    echo "Unsupported profile type: $profile_path" >&2
    echo "Expected .sqlite, .sqlite3, or .nsys-rep" >&2
    exit 1
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

base_name="$(basename "$profile_path")"
stem="${base_name%.sqlite}"
stem="${stem%.sqlite3}"
stem="${stem%.nsys-rep}"
gpu_suffix=""
if [[ -n "$gpu_id" ]]; then
  gpu_suffix=".gpu${gpu_id}"
fi

mkdir -p "$repo_root/outputs"
markdown_path="$repo_root/outputs/${stem}${gpu_suffix}.report.md"
findings_path="$repo_root/outputs/${stem}${gpu_suffix}.findings.json"

cmd=(python3 -m sysight analyze "$profile_path" --markdown "$markdown_path" --findings "$findings_path")
if [[ -n "$gpu_id" ]]; then
  cmd+=(--gpu "$gpu_id")
fi

cd "$repo_root"
PYTHONPATH=src "${cmd[@]}"

echo
echo "ANALYSIS_MARKDOWN=$markdown_path"
echo "ANALYSIS_FINDINGS=$findings_path"
