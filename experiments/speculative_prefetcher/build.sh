#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBURING_PREFIX="${LIBURING_PREFIX:-/data/disk2/ljc/solidattention_deps/liburing-install}"

cmake -S "$ROOT" -B "$ROOT/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLIBURING_PREFIX="$LIBURING_PREFIX"

cmake --build "$ROOT/build" -j"$(nproc)"

mkdir -p "$ROOT/bin"
cp "$ROOT/build/speculative_prefetch_replay" "$ROOT/bin/"

echo "$ROOT/bin"
