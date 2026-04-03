#!/bin/bash
# Quick start server with default settings

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set default environment variables if not set
export RDS_API_KEY="${RDS_API_KEY:-change-me-to-a-strong-random-key}"
export RDS_HOST="${RDS_HOST:-0.0.0.0}"
export RDS_PORT="${RDS_PORT:-44401}"
export RDS_WORKSPACE_ROOT="${RDS_WORKSPACE_ROOT:-/tmp/rds_workspace}"
export RDS_LOG_ROOT="${RDS_LOG_ROOT:-/tmp/rds_logs}"
export RDS_DB_PATH="${RDS_DB_PATH:-/tmp/rds.db}"

echo "=== Starting Remote Device Server ==="
echo "Host: $RDS_HOST:$RDS_PORT"
echo "API Key: ${RDS_API_KEY:0:10}..."
echo "Workspace: $RDS_WORKSPACE_ROOT"
echo "Logs: $RDS_LOG_ROOT"
echo ""

cd "$PROJECT_ROOT"
python -m server.main
