#!/usr/bin/env bash
# Run a 1+0 blitz stress gauntlet between the current engine and a reference build.
# Requires cutechess-cli in PATH.

set -euo pipefail

ENGINE_NEW=${ENGINE_NEW:-"./program"}
ENGINE_OLD=${ENGINE_OLD:-"./program_old"}
GAMES=${GAMES:-200}
CONCURRENCY=${CONCURRENCY:-4}
TC=${TC:-60} # seconds total (1+0 equivalent)

LOG_DIR=${LOG_DIR:-build/gauntlet}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/blitz_gauntlet.log"
PGN_FILE="$LOG_DIR/blitz_gauntlet.pgn"
SUMMARY_FILE="$LOG_DIR/blitz_gauntlet.summary"
ENG_NEW_STDERR="$LOG_DIR/engine_new.stderr"
ENG_OLD_STDERR="$LOG_DIR/engine_old.stderr"

# Prefer repo-provided cutechess wrapper if available
if [ -x "$(dirname "$0")/cutechess-cli" ]; then
  PATH="$(dirname "$0"):$PATH"
fi

if ! command -v cutechess-cli >/dev/null 2>&1; then
  echo "cutechess-cli not found in PATH" >&2
  exit 1
fi

if [ "${APPEND_LOG:-0}" -eq 1 ]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  LOG_FILE="$LOG_DIR/blitz_gauntlet.$ts.log"
  PGN_FILE="$LOG_DIR/blitz_gauntlet.$ts.pgn"
  SUMMARY_FILE="$LOG_DIR/blitz_gauntlet.$ts.summary"
  : >"$LOG_FILE"
  : >"$PGN_FILE"
  : >"$SUMMARY_FILE"
  echo "Running blitz gauntlet: $ENGINE_NEW vs $ENGINE_OLD" | tee -a "$LOG_FILE"
else
  : >"$LOG_FILE"
  : >"$PGN_FILE"
  : >"$SUMMARY_FILE"
  echo "Running blitz gauntlet: $ENGINE_NEW vs $ENGINE_OLD" | tee -a "$LOG_FILE"
fi

# Validate engine executables
if [ ! -x "$ENGINE_NEW" ]; then
  echo "ERROR: ENGINE_NEW not executable: $ENGINE_NEW" | tee -a "$LOG_FILE"
  exit 1
fi
if [ ! -x "$ENGINE_OLD" ]; then
  echo "ERROR: ENGINE_OLD not executable: $ENGINE_OLD" | tee -a "$LOG_FILE"
  exit 1
fi

# Wrap engines to capture stderr metrics if LOG_METRICS=1
ENGINE_NEW_CMD="$ENGINE_NEW"
ENGINE_OLD_CMD="$ENGINE_OLD"
if [ "${LOG_METRICS:-0}" -eq 1 ]; then
  : >"$ENG_NEW_STDERR"
  : >"$ENG_OLD_STDERR"
  ENGINE_NEW_WRAPPER="$(mktemp)"
  ENGINE_OLD_WRAPPER="$(mktemp)"
  cat >"$ENGINE_NEW_WRAPPER" <<EOF
#!/usr/bin/env bash
exec "$ENGINE_NEW" "\$@" 2>>"$ENG_NEW_STDERR"
EOF
  cat >"$ENGINE_OLD_WRAPPER" <<EOF
#!/usr/bin/env bash
exec "$ENGINE_OLD" "\$@" 2>>"$ENG_OLD_STDERR"
EOF
  chmod +x "$ENGINE_NEW_WRAPPER" "$ENGINE_OLD_WRAPPER"
  ENGINE_NEW_CMD="$ENGINE_NEW_WRAPPER"
  ENGINE_OLD_CMD="$ENGINE_OLD_WRAPPER"
fi

cleanup() {
  [ -n "${ENGINE_NEW_WRAPPER:-}" ] && [ -f "$ENGINE_NEW_WRAPPER" ] && rm -f "$ENGINE_NEW_WRAPPER"
  [ -n "${ENGINE_OLD_WRAPPER:-}" ] && [ -f "$ENGINE_OLD_WRAPPER" ] && rm -f "$ENGINE_OLD_WRAPPER"
  [ -n "${RUN_LOG:-}" ] && [ -f "$RUN_LOG" ] && rm -f "$RUN_LOG"
}
trap cleanup EXIT

RUN_LOG="$(mktemp)"
cutechess-cli \
  -engine cmd="$ENGINE_NEW_CMD" proto=xboard \
  -engine cmd="$ENGINE_OLD_CMD" proto=xboard \
  -each tc=$TC \
  -games $GAMES \
  -concurrency $CONCURRENCY \
  -pgnout "$PGN_FILE" \
  -repeat \
  -recover \
  | tee -a "$LOG_FILE" | tee "$RUN_LOG"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "ERROR: cutechess-cli exited with status $status" | tee -a "$LOG_FILE"
  exit "$status"
fi

echo "--- Summary ---" | tee -a "$LOG_FILE"

# Count time forfeits (flagged losses) only in this run's output from result lines.
FLAG_LOSSES=$(grep -E "^[[:space:]]*Finished game|\\[Result" "$RUN_LOG" | grep -Ei "forfeit|on time" | wc -l | awk '{print $1}')
rm -f "$RUN_LOG"
echo "Flagged losses: $FLAG_LOSSES" | tee -a "$LOG_FILE"

if [ "$FLAG_LOSSES" -gt 0 ]; then
  echo "ERROR: Timeouts detected. Aborting." | tee -a "$LOG_FILE"
  exit 2
fi

# Parse metrics if we captured stderr
AVG_DEPTH="n/a"
AVG_MOVE_MS="n/a"
PV_INSTABLE="n/a"
if [ -f "$ENG_NEW_STDERR" ]; then
  # Average depth across depth_done metrics
  if grep -q "METRIC depth_done" "$ENG_NEW_STDERR"; then
    AVG_DEPTH=$(awk '/METRIC depth_done/ {for(i=1;i<=NF;i++){if($i ~ /depth=/){split($i,a,\"=\"); sum+=a[2]; c++}}} END{if(c>0) printf \"%.2f\", sum/c; else print \"n/a\"}' "$ENG_NEW_STDERR")
    PV_INSTABLE=$(awk '/METRIC depth_done/ {for(i=1;i<=NF;i++){if($i ~ /pv_unstable=/){split($i,a,\"=\"); inst+=a[2]; c++}}} END{if(c>0) printf \"%.2f%%\", (inst/c)*100; else print \"n/a\"}' "$ENG_NEW_STDERR")
  fi
  if grep -q "METRIC move_time" "$ENG_NEW_STDERR"; then
    AVG_MOVE_MS=$(awk '/METRIC move_time/ {for(i=1;i<=NF;i++){if($i ~ /ms=/){split($i,a,\"=\"); sum+=a[2]; c++}}} END{if(c>0) printf \"%.2f\", sum/c; else print \"n/a\"}' "$ENG_NEW_STDERR")
  fi
fi

cat > "$SUMMARY_FILE" <<EOF
Games: $GAMES
Concurrency: $CONCURRENCY
Flagged losses: $FLAG_LOSSES
Avg depth (new): $AVG_DEPTH
Avg move time ms (new): $AVG_MOVE_MS
PV instability rate (new): $PV_INSTABLE
EOF

echo "Summary: $SUMMARY_FILE"
echo "Log: $LOG_FILE"
echo "PGN: $PGN_FILE"

cat <<'NOTE'
Notes:
- This harness focuses on timeouts; cutechess output is appended to the log.
- For per-move depth/time and PV instability metrics, enable engine logging and post-process
  engine logs as needed; cutechess does not provide those directly.
- The script exits non-zero if cutechess-cli is unavailable or the run fails.
NOTE
