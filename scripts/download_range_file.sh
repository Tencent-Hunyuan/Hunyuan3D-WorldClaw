#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 URL OUTPUT EXPECTED_BYTES EXPECTED_SHA256 [CHUNK_BYTES]" >&2
  exit 2
fi
URL="$1"
OUTPUT="$2"
EXPECTED_BYTES="$3"
EXPECTED_SHA256="$4"
CHUNK_BYTES="${5:-268435456}"
mkdir -p "$(dirname "$OUTPUT")"

while true; do
  current=0
  if [[ -f "$OUTPUT" ]]; then
    current="$(stat -c '%s' "$OUTPUT")"
  fi
  if (( current >= EXPECTED_BYTES )); then
    break
  fi
  end=$((current + CHUNK_BYTES - 1))
  if (( end >= EXPECTED_BYTES )); then
    end=$((EXPECTED_BYTES - 1))
  fi
  chunk="${OUTPUT}.chunk"
  rm -f "$chunk"
  curl --fail --silent --show-error --location --retry 5 --retry-delay 2 \
    --connect-timeout 30 --max-time 900 --range "${current}-${end}" \
    "$URL" --output "$chunk"
  got="$(stat -c '%s' "$chunk")"
  want=$((end - current + 1))
  if (( got != want )); then
    echo "range ${current}-${end}: got ${got} bytes, expected ${want}" >&2
    exit 1
  fi
  cat "$chunk" >> "$OUTPUT"
  rm -f "$chunk"
done

actual_size="$(stat -c '%s' "$OUTPUT")"
if (( actual_size != EXPECTED_BYTES )); then
  echo "final size ${actual_size} != ${EXPECTED_BYTES}" >&2
  exit 1
fi
actual_hash="$(sha256sum "$OUTPUT" | awk '{print $1}')"
if [[ "$actual_hash" != "$EXPECTED_SHA256" ]]; then
  echo "sha256 ${actual_hash} != ${EXPECTED_SHA256}" >&2
  exit 1
fi
echo "range download ok: ${OUTPUT} ${actual_size} bytes ${actual_hash}"
