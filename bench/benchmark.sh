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
grep '^benchmark' bench/script.sh | sed 's/;$//' | \
  xargs -P "$1" -I CMD env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c 'run_job "CMD"'
