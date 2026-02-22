#!/usr/bin/env sh
# Source this file to initialize SimpleCoder env vars for Dartmouth Chat.
#
# Usage (from pset-1-simplecoder-agent):
#   . scripts/simplecoder_init.sh
# or
#   source scripts/simplecoder_init.sh

# IMPORTANT: do NOT use `set -u` here; it can break zsh prompt helpers.

SIMPLECODER_API_BASE=${SIMPLECODER_API_BASE:-https://chat.dartmouth.edu/api}
SIMPLECODER_MODEL=${SIMPLECODER_MODEL:-anthropic.claude-3-5-haiku-20241022}
export SIMPLECODER_API_BASE
export SIMPLECODER_MODEL

if [ -z "${DARTMOUTH_CHAT_API_KEY:-}" ]; then
  printf "%s\n" "Enter Dartmouth Chat API key (input hidden):" 1>&2
  # shellcheck disable=SC2162
  stty -echo 2>/dev/null
  read DARTMOUTH_CHAT_API_KEY
  stty echo 2>/dev/null
  printf "%s\n" "" 1>&2
  export DARTMOUTH_CHAT_API_KEY
fi

printf "%s\n" "SimpleCoder initialized:" 1>&2
printf "%s\n" "  SIMPLECODER_API_BASE=$SIMPLECODER_API_BASE" 1>&2
printf "%s\n" "  SIMPLECODER_MODEL=$SIMPLECODER_MODEL" 1>&2
printf "%s\n" "" 1>&2
printf "%s\n" "Next:" 1>&2
printf "%s\n" "  python -m simplecoder.main --model \"$SIMPLECODER_MODEL\" --verbose \"What is deep learning?\"" 1>&2
