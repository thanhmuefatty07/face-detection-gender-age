#!/usr/bin/env bash
# One-Click Biometric Target Locker Launcher (macOS / Linux Terminal)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
exec "$ROOT_DIR/run.command" "$@"
