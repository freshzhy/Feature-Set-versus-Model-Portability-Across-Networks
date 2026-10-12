#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — single entry point for the whole pipeline.
#
#   ./run.sh setup          create .venv and install requirements.txt
#   ./run.sh preprocess     data/raw/<DS>.csv  ->  data/processed/<DS>_processed.csv
#   ./run.sh rq1            RQ1 signatures, distinctiveness, 30-seed stability
#   ./run.sh rq2            RQ2 cross-dataset similarity
#   ./run.sh rq3            RQ3 diagnostics, transfer experiment (30 seeds), JSD
#   ./run.sh robustness     dedup / train-SHAP / re-selection / scenario ablation
#   ./run.sh figures        all paper figures -> results/figures/
#   ./run.sh all            everything above, in order
#   ./run.sh status         progress of the long-running steps
#
# Any step can be sent to the background (keeps running after the terminal
# closes; on macOS the machine is kept awake):
#   ./run.sh bg rq3         log: results/logs/rq3.log   stop: kill $(cat results/logs/rq3.pid)
# The RQ3 experiments resume from completed tasks when restarted.
#
# Interpreter: $PY if set, else .venv/bin/python, else python3.
# Parallelism: WORKERS / THREADS environment variables (passed to rq3_transfer).
# N_SEEDS overrides the number of RQ3 seeds (for a quick test only).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

DATASETS=(NF-UNSW-NB15-v2 NF-CSE-CIC-IDS2018-v2 NF-ToN-IoT-v2 NF-BoT-IoT-v2)
RAW=data/raw; PROC=data/processed; R=results; LOGS=$R/logs

if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=$(command -v python3); fi
fi
PAR=()
[ -n "${WORKERS:-}" ] && PAR+=(--workers "$WORKERS")
[ -n "${THREADS:-}" ] && PAR+=(--threads "$THREADS")
[ -n "${N_SEEDS:-}" ] && PAR+=(--n-seeds "$N_SEEDS")      # quick test runs only; paper used defaults

log()  { mkdir -p "$LOGS"; echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/pipeline.log"; }
need() { for f in "$@"; do [ -e "$f" ] || { echo "✗ missing $f — run the earlier step first"; exit 1; }; done; }
check_deps() {
  "$PY" -c "import numpy, pandas, sklearn, scipy, xgboost, shap, matplotlib, seaborn" 2>/dev/null \
    || { echo "✗ $PY lacks dependencies — run ./run.sh setup"; exit 1; }
}

# ── steps ────────────────────────────────────────────────────────────────────
step_setup() {
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
  echo "✓ environment ready (.venv)"
}

step_preprocess() {
  mkdir -p "$PROC" "$R/preprocess_logs"
  for ds in "${DATASETS[@]}"; do
    need "$RAW/$ds.csv"
    extra=(); [ "$ds" != NF-UNSW-NB15-v2 ] && extra=(--skip-dup-check)   # large files: duplicate-column check pre-verified
    log "preprocess $ds"
    "$PY" code/preprocess.py --data "$RAW/$ds.csv" --out "$PROC" ${extra[@]+"${extra[@]}"} \
      2>&1 | tee "$R/preprocess_logs/${ds}_preprocess.log"
  done
}

step_rq1() {
  for ds in "${DATASETS[@]}"; do
    need "$PROC/${ds}_processed.csv"
    log "RQ1 signatures $ds"
    "$PY" code/rq1_signatures.py --data "$PROC/${ds}_processed.csv" --out "$R/rq1_signatures"
  done
  log "RQ1 intra-dataset distinctiveness"
  "$PY" code/rq1_distinctiveness.py --rq1-dir "$R/rq1_signatures" --out "$R/rq1_distinctiveness"
  log "RQ1 30-seed signature stability"
  "$PY" code/rq1_stability.py --data-dir "$PROC" --out "$R/rq1_stability" --n-seeds 30
}

step_rq2() {
  log "RQ2 cross-dataset similarity"
  "$PY" code/rq2_similarity.py --rq1-dir "$R/rq1_signatures" --out "$R/rq2_similarity"
}

# rq3_transfer: prep -> run -> aggregate for one mode (extra args select the mode)
transfer() {
  local out=$1; shift
  [ -f "$out/feature_sets.json" ] || "$PY" code/rq3_transfer.py prep "$@"
  "$PY" code/rq3_transfer.py run "$@" ${PAR[@]+"${PAR[@]}"}
  "$PY" code/rq3_transfer.py aggregate "$@" > /dev/null
  log "  -> $out/summary.txt"
}

step_rq3() {
  need "$R/rq1_signatures"
  log "RQ3 diagnostics (split leakage, raw class counts)"
  "$PY" code/rq3_diagnostics.py
  log "RQ3 transfer experiment (30 seeds; several hours)"
  transfer "$R/rq3_main"
  log "RQ3 JSD vs transfer gain"
  "$PY" code/rq3_jsd.py --rq1-dir "$R/rq1_signatures" --data-dir "$PROC" \
    --rq3-dir "$R/rq3_main" --out "$R/rq3_jsd"
}

step_robustness() {
  need "$R/rq3_main/feature_sets.json"
  log "Robustness 1/5: RQ1 signatures on deduplicated data"
  "$PY" code/rq1_dedup_robustness.py run
  "$PY" code/rq1_dedup_robustness.py compare > /dev/null
  log "Robustness 2/5: RQ3 on deduplicated data";              transfer "$R/rq3_dedup"          --dedup
  log "Robustness 3/5: RQ3 with training-split SHAP";           transfer "$R/rq3_trainshap"      --variant trainshap
  log "Robustness 4/5: RQ3 with re-selection on dedup data";    transfer "$R/rq3_dedup_reselect" --variant reselect
  log "Robustness 5/5: RQ3 without scenario-specific features"; transfer "$R/rq3_noscenario"     --variant noscenario
}

step_figures() {
  log "Figures"
  "$PY" code/plot_figures_rq1_rq2.py --rq1-dir "$R/rq1_signatures" --rq2-dir "$R/rq2_similarity"
  "$PY" code/plot_figures_rq3.py
  for e in pdf png; do                      # Fig. 3 is drawn by rq1_distinctiveness.py
    cp "$R/rq1_distinctiveness/summary_figure.$e" "$R/figures/Fig3_intra_distinctiveness.$e"
  done
  log "  -> $R/figures/"
}

step_all() { step_preprocess; step_rq1; step_rq2; step_rq3; step_robustness; step_figures; }

step_status() {
  tail -n 8 "$LOGS/pipeline.log" 2>/dev/null || echo "(nothing started yet)"
  for p in "$LOGS"/*.pid; do
    [ -e "$p" ] || continue
    if kill -0 "$(cat "$p")" 2>/dev/null; then echo "● running: $(basename "$p" .pid) (PID $(cat "$p"))"; fi
  done
  for m in "rq3_main:" "rq3_dedup:--dedup" "rq3_trainshap:--variant trainshap" \
           "rq3_dedup_reselect:--variant reselect" "rq3_noscenario:--variant noscenario"; do
    d=${m%%:*}; a=${m#*:}
    [ -f "$R/$d/run_config.json" ] || continue
    echo "── $d"; "$PY" code/rq3_transfer.py status $a 2>/dev/null | head -6 || true
  done
}

# ── dispatch ─────────────────────────────────────────────────────────────────
STEPS="setup preprocess rq1 rq2 rq3 robustness figures all status"
cmd=${1:-}
if [ "$cmd" = bg ]; then
  s=${2:-}; [[ " $STEPS " == *" $s "* && $s != status && $s != setup ]] || { echo "usage: ./run.sh bg <step>"; exit 1; }
  check_deps; mkdir -p "$LOGS"
  if [ -f "$LOGS/$s.pid" ] && kill -0 "$(cat "$LOGS/$s.pid")" 2>/dev/null; then
    echo "⚠ $s already running (PID $(cat "$LOGS/$s.pid"))"; exit 1; fi
  CAFF=(); command -v caffeinate >/dev/null 2>&1 && CAFF=(caffeinate -i)   # macOS only
  nohup ${CAFF[@]+"${CAFF[@]}"} "$(pwd)/run.sh" "$s" > "$LOGS/$s.log" 2>&1 &
  echo $! > "$LOGS/$s.pid"
  echo "✓ '$s' started in background (PID $!)"
  echo "  progress: ./run.sh status    full log: tail -f $LOGS/$s.log"
  echo "  stop:     kill \$(cat $LOGS/$s.pid)   (re-running resumes)"
  exit 0
fi
[[ -n "$cmd" && " $STEPS " == *" $cmd "* ]] || { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ "$cmd" = setup ] || check_deps
"step_$cmd"
