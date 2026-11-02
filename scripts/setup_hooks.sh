#!/usr/bin/env bash
# Points git at the repo's hooks so the commit-message rule runs on every commit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
chmod +x .githooks/*
git config core.hooksPath .githooks
echo "Hooks enabled from .githooks/"
