#!/usr/bin/env bash
set -euo pipefail

# Fetch official C++ learned index baselines (microsoft/ALEX and Jiacheng-WU/lipp)
# into baselines/third_party/ for portable CMake/CXX builds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/baselines/third_party"
mkdir -p "${THIRD_PARTY_DIR}"

if [ ! -d "${THIRD_PARTY_DIR}/ALEX" ]; then
  echo "[fetch] Cloning microsoft/ALEX -> ${THIRD_PARTY_DIR}/ALEX"
  git clone --depth 1 https://github.com/microsoft/ALEX.git "${THIRD_PARTY_DIR}/ALEX"
else
  echo "[fetch] ALEX already present at ${THIRD_PARTY_DIR}/ALEX"
fi

if [ ! -d "${THIRD_PARTY_DIR}/LIPP" ]; then
  echo "[fetch] Cloning Jiacheng-WU/lipp -> ${THIRD_PARTY_DIR}/LIPP"
  git clone --depth 1 https://github.com/Jiacheng-WU/lipp.git "${THIRD_PARTY_DIR}/LIPP"
else
  echo "[fetch] LIPP already present at ${THIRD_PARTY_DIR}/LIPP"
fi

echo "[OK] Third-party baseline headers ready in ${THIRD_PARTY_DIR}"
