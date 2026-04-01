#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <profile.nsys-rep> [output.sqlite]" >&2
  exit 1
fi

caller_pwd="$(pwd)"
input_arg="$1"
output_arg="${2:-}"

case "$input_arg" in
  /*) input_path="$input_arg" ;;
  *) input_path="$caller_pwd/$input_arg" ;;
esac

if [[ ! -f "$input_path" ]]; then
  echo "Profile not found: $input_path" >&2
  exit 1
fi

case "$input_path" in
  *.nsys-rep) ;;
  *)
    echo "Unsupported profile type: $input_path" >&2
    echo "Expected a .nsys-rep file" >&2
    exit 1
    ;;
esac

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found on PATH" >&2
  echo "Expected to run: nsys export --type=sqlite -o <out.sqlite> <file.nsys-rep>" >&2
  exit 1
fi

if [[ -n "$output_arg" ]]; then
  case "$output_arg" in
    /*) output_path="$output_arg" ;;
    *) output_path="$caller_pwd/$output_arg" ;;
  esac
else
  output_path="${input_path%.nsys-rep}.sqlite"
fi

mkdir -p "$(dirname "$output_path")"

nsys export --type=sqlite -o "$output_path" --force-overwrite=true "$input_path"

if [[ ! -s "$output_path" ]]; then
  echo "nsys export did not produce a usable sqlite file: $output_path" >&2
  exit 1
fi

echo "EXPORTED_SQLITE=$output_path"
