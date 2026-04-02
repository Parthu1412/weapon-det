#!/usr/bin/env bash
set -eu

# Repo root
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_DIR"
BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/build}"

cleanup() {
  echo "Stopping weapon-cpp processes..."
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

echo "Starting camera_reader (binds ZMQ_CAMERA_TO_WEAPON_PORT)..."
"${BUILD_DIR}/camera_reader" &
sleep 1

echo "Starting weapon_inference (binds ZMQ_WEAPON_TO_OUTPUT_PORT)..."
"${BUILD_DIR}/weapon_inference" &
sleep 1

echo "Starting msg_gen (connects to weapon output)..."
"${BUILD_DIR}/msg_gen" &
wait
