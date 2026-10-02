#!/usr/bin/env bash
set -euo pipefail
self="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$(dirname "$self")/../.."
export PATH="${PHLAG_ENV:-$HOME/.local/share/mamba/envs/phlag}/bin:$PATH"

if [[ -z ${BATCH_INNER:-} ]]; then
  if [[ -n ${BATCH_SESSION:-} ]]; then
    SESSION=$BATCH_SESSION
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
      echo "batch: tmux session '$SESSION' already exists -- attach with: tmux attach -t $SESSION" >&2
      exit 1
    fi
  else
    SESSION=batch
    n=2
    while tmux has-session -t "=$SESSION" 2>/dev/null; do
      SESSION=batch$n
      n=$((n + 1))
    done
  fi
  export BATCH_LOG_DIR=${BATCH_LOG_DIR:-logs/$SESSION}
  env_args=(-e BATCH_INNER=1)
  for var in PHLAG_ENV BATCH_LOG_DIR CONNECTION_DIR; do
    [[ -n ${!var:-} ]] && env_args+=(-e "$var=${!var}")
  done
  tmux new-session -d -s "$SESSION" -c "$PWD" "${env_args[@]}" \
    "$(printf '%q ' "$self" "$@"); read -rp 'batch finished -- press enter to close '"
  if [[ -n ${TMUX:-} ]]; then
    exec tmux switch-client -t "$SESSION"
  elif [[ -t 0 && -t 1 ]]; then
    exec tmux attach -t "$SESSION"
  fi
  echo "batch: running in tmux session '$SESSION' -- attach with: tmux attach -t $SESSION"
  exit 0
fi

LOG_DIR=$BATCH_LOG_DIR
rm -rf "$LOG_DIR"
exec python -m bench.cli.run_batch "${1:-bench/script.sh}" --log-dir "$LOG_DIR"
