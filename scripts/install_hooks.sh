#!/usr/bin/env bash
echo "[install_hooks.sh] Installing Git hooks..."

cp scripts/hooks/post-merge .git/hooks/post-merge
chmod +x .git/hooks/post-merge

cp scripts/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push

echo "[install_hooks.sh] Done installing hooks."
