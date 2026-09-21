rm -r logs/*;
mkdir -p logs
run_job() {
  cmd="$1"
  name=$(echo "$cmd" | grep -oE "gaussian/[^ ]+" | tail -1 | sed "s#gaussian/##" | tr "/" "_")
  eval "$cmd" > "logs/${name}.log" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "[error] $name (exit $status): $(tail -5 "logs/${name}.log")"
  else
    echo "[done] $name (exit $status)"
  fi
}
export -f run_job

# outer_jobs (one per script.sh line, all run at once) times each one's own
# inner worker count (PHLAG_BENCH_JOBS, read by bench/benchmark.py's
# inner_worker_cap(), which enforces the actual TOTAL_CORE_BUDGET=50) stays
# <= that budget -- no argument needed.
outer_jobs=$(grep -c '^benchmark' bench/script.sh)
export PHLAG_BENCH_JOBS="$outer_jobs"
grep '^benchmark' bench/script.sh | sed 's/;$//' | \
  xargs -P "$outer_jobs" -I CMD env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c 'run_job "CMD"'
