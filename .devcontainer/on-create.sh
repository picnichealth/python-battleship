#!/usr/bin/env bash
set -euo pipefail

echo "==> Setting up interview environment..."
echo ""

# Install project Python dependencies via uv
if [ -f "pyproject.toml" ]; then
	echo "--- Installing Python dependencies ---"
	uv sync
	echo ""
	# Activate venv by default so python3 has all deps available
	echo 'source /workspaces/python-battleship/.venv/bin/activate' >>~/.bashrc
fi

# Summary
echo "=== Interview Environment Ready ==="
echo ""
echo "Python:      $(python3 --version 2>/dev/null || echo 'not found')"
echo "Node.js:     $(node --version 2>/dev/null || echo 'not found')"
echo "pnpm:        $(pnpm --version 2>/dev/null || echo 'not found')"
echo "uv:          $(uv --version 2>/dev/null || echo 'not found')"
echo "Claude Code: $(claude --version 2>/dev/null || echo 'not found')"
echo ""
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
	echo "Claude Code: API key configured"
else
	echo "Claude Code: ANTHROPIC_API_KEY not set (set in repo Codespaces secrets)"
fi
