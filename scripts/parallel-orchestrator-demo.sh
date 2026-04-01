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

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
timestamp="$(date +%Y%m%d-%H%M%S)"
stem="$(basename "$profile_path")"
stem="${stem%.sqlite}"
stem="${stem%.sqlite3}"
stem="${stem%.nsys-rep}"
run_dir="$repo_root/outputs/orchestrator/${stem}-${timestamp}"
mkdir -p "$run_dir"

tasks=(
  "top_kernels"
  "nvtx_layer_breakdown"
  "root_cause_matcher"
  "code_location"
)

pids=()
messages=()
logs=()

echo "Main agent workspace: $repo_root"
echo "Profile: $profile_path"
echo "Output dir: $run_dir"
echo

build_prompt() {
  local skill_name="$1"
  local cmd="PYTHONPATH=src python3 -m sysight skill run ${skill_name} '${profile_path}'"
  if [[ -n "$gpu_id" ]]; then
    cmd+=" --gpu ${gpu_id}"
  fi
  cat <<EOF
You are a worker process in a parent-child Codex orchestration demo.

Rules:
- Do not edit any files.
- Do not generate markdown reports.
- Do exactly one analysis task by running this command from the repo root:
  ${cmd}
- After the command finishes, return a concise plain-text summary with:
  1. skill name
  2. 2-4 key findings
  3. no extra planning text
EOF
}

for task in "${tasks[@]}"; do
  message_file="$run_dir/${task}.message.txt"
  log_file="$run_dir/${task}.log"

  echo "Starting worker: $task"
  codex exec \
    --skip-git-repo-check \
    --full-auto \
    --ephemeral \
    -C "$repo_root" \
    -o "$message_file" \
    "$(build_prompt "$task")" \
    >"$log_file" 2>&1 &
  pids+=("$!")
  messages+=("$message_file")
  logs+=("$log_file")
done

echo
echo "Waiting for workers..."
echo

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  pid="${pids[$i]}"
  message_file="${messages[$i]}"
  log_file="${logs[$i]}"
  if wait "$pid"; then
    status="ok"
  else
    status="failed"
  fi
  echo "Worker $task: $status"
  echo "  message: $message_file"
  echo "  log:     $log_file"
  if [[ -s "$message_file" ]]; then
    sed -n '1,12p' "$message_file" | sed 's/^/    /'
  else
    echo "    (no message output)"
  fi
  echo
done

echo "ORCHESTRATOR_OUTPUT_DIR=$run_dir"
