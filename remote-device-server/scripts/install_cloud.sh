#!/bin/bash
# Install on cloud machine (no internet required if offline packages available)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Remote Device Server Installation ==="
echo ""

# Check if offline packages exist
if [ -d "$PROJECT_ROOT/offline_packages" ]; then
    echo "Found offline packages, installing..."
    pip install --no-index --find-links="$PROJECT_ROOT/offline_packages" \
        fastapi "uvicorn[standard]" pydantic aiosqlite \
        pynvml psutil python-multipart httpx typer rich websockets
else
    echo "No offline packages found."
    echo "Run scripts/prepare_offline.sh on a machine with internet first."
    exit 1
fi

echo ""
echo "Installing remote-device-server..."
cd "$PROJECT_ROOT"
pip install --no-deps -e .

echo ""
echo "Installation complete!"
echo ""
echo "Start the server:"
echo "  export RDS_API_KEY=your-secret-key"
echo "  python -m server.main"
