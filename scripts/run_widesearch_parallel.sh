#!/usr/bin/env bash
# Run WideSearch in parallel across N lanes, where each lane talks to its
# own compact_server instance on a different pod GPU (via kubectl
# port-forward 18080..18080+N-1 → pod ports 8080..8080+N-1).
#
# Pre-reqs:
#   - Pod side: scripts/run_compact_server_multi_gpu.sh already running with
#     the same NGPU (this script does not start servers — pod needs an
#     interactive kubectl exec session, see kv_compact memory note).
#   - VM side: .venv activated or python3 deps available.
#   - Env: BING_APPID, JINA_API_KEYS exported in this shell.
#
# Usage:
#   NGPU=8 POD_ID=<id>-n0-0-master-0 CTX=prod-lor1-k8s-2 \
#   INSTANCE_PREFIX=ws_en TOTAL=200 \
#   MODEL_CFG=compact-qwen3_5-4b-future-proxy-dual URL_MAP_MODE=url \
#   bash scripts/run_widesearch_parallel.sh
#
# Sharding: TOTAL tasks split into NGPU contiguous slices of size
# ceil(TOTAL/NGPU); last lane takes the remainder.
#
# Each lane runs in the background; the script waits for all to finish then
# exits. Logs at /tmp/widesearch_worker{i}.log, port-forward logs at
# /tmp/pf_widesearch_gpu{i}.log.
set -euo pipefail

# ---- config ----
NGPU="${NGPU:-8}"
POD_ID="${POD_ID:?POD_ID required (e.g. ff6cc80de7c9b477594c-n0-0-master-0)}"
CTX="${CTX:-prod-lor1-k8s-2}"
NAMESPACE="${NAMESPACE:-training-coreai}"
MODEL_CFG="${MODEL_CFG:-compact-qwen3_5-4b-future-proxy-dual}"
URL_MAP_MODE="${URL_MAP_MODE:-url}"
INSTANCE_PREFIX="${INSTANCE_PREFIX:-ws_en}"
TOTAL="${TOTAL:-200}"
TRIAL_NUM="${TRIAL_NUM:-1}"
STAGE="${STAGE:-both}"
LOCAL_PORT_BASE="${LOCAL_PORT_BASE:-18080}"
POD_PORT_BASE="${POD_PORT_BASE:-8080}"

# Sanity: required env vars for tool layer
: "${BING_APPID:?BING_APPID required for Bing search}"
: "${JINA_API_KEYS:?JINA_API_KEYS required for Jina Reader}"

# Ensure PYTHONPATH points at WideSearch root (so `from src...` resolves).
REPO_ROOT="${WIDESEARCH_REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "[parallel] NGPU=${NGPU} TOTAL=${TOTAL} MODEL_CFG=${MODEL_CFG} URL_MAP_MODE=${URL_MAP_MODE}"
echo "[parallel] POD ${CTX}/${NAMESPACE}/${POD_ID}"
echo ""

# ---- 1. start port-forwards (background) ----
PF_PIDS=()
for i in $(seq 0 $((NGPU - 1))); do
    LOCAL_PORT=$((LOCAL_PORT_BASE + i))
    POD_PORT=$((POD_PORT_BASE + i))
    LOG="/tmp/pf_widesearch_gpu${i}.log"
    : > "${LOG}"
    nohup kubectl --context "${CTX}" -n "${NAMESPACE}" port-forward \
        --address 127.0.0.1 "pod/${POD_ID}" "${LOCAL_PORT}:${POD_PORT}" \
        > "${LOG}" 2>&1 &
    PF_PIDS+=($!)
    echo "[parallel] pf lane $i: 127.0.0.1:${LOCAL_PORT} -> pod:${POD_PORT} (pid $!)"
done

# Cleanup hook: kill port-forwards on exit (workers' own subshells inherit
# this; the trap runs in the orchestrator after `wait` returns).
cleanup() {
    echo "[parallel] cleanup: killing port-forwards (${#PF_PIDS[@]})"
    for pid in "${PF_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ---- 2. wait for port-forwards to be healthy ----
echo ""
echo "[parallel] waiting for all ${NGPU} pf+server lanes to respond /health..."
for i in $(seq 0 $((NGPU - 1))); do
    LOCAL_PORT=$((LOCAL_PORT_BASE + i))
    ok=0
    for tries in $(seq 1 30); do
        if curl -s --max-time 3 "http://localhost:${LOCAL_PORT}/health" \
                | grep -q '"status":"ok"'; then
            ok=1
            break
        fi
        sleep 2
    done
    if [ "${ok}" -ne 1 ]; then
        echo "[parallel] ERROR: lane ${i} (port ${LOCAL_PORT}) never came up" >&2
        echo "  Inspect /tmp/pf_widesearch_gpu${i}.log and pod /tmp/compact_server_gpu${i}.log" >&2
        exit 1
    fi
    echo "  lane $i OK"
done

# ---- 3. compute task slices ----
# Contiguous 1-indexed slices of size ceil(TOTAL/NGPU); last lane keeps any
# remainder. instance_id format: ws_en_001 .. ws_en_TOTAL (zero-padded width
# 3, matching WideSearch dataset).
SLICE_SIZE=$(( (TOTAL + NGPU - 1) / NGPU ))
echo ""
echo "[parallel] slice size ~${SLICE_SIZE} task/lane"

WORKER_PIDS=()
for i in $(seq 0 $((NGPU - 1))); do
    START=$((i * SLICE_SIZE + 1))
    END=$(((i + 1) * SLICE_SIZE))
    if [ "${END}" -gt "${TOTAL}" ]; then END="${TOTAL}"; fi
    if [ "${START}" -gt "${TOTAL}" ]; then
        echo "[parallel] lane $i: no tasks (TOTAL=${TOTAL} reached); skipping"
        continue
    fi

    # Build comma-separated instance_id list for this slice.
    IDS=$(seq -f "${INSTANCE_PREFIX}_%03g" "${START}" "${END}" | paste -sd, -)

    LOCAL_PORT=$((LOCAL_PORT_BASE + i))
    LOG="/tmp/widesearch_worker${i}.log"
    : > "${LOG}"

    COMPACT_BASE_URL="http://localhost:${LOCAL_PORT}" \
    URL_MAP_MODE="${URL_MAP_MODE}" \
    BING_APPID="${BING_APPID}" \
    JINA_API_KEYS="${JINA_API_KEYS}" \
    PYTHONPATH="${PYTHONPATH}" \
    nohup python3 "${REPO_ROOT}/scripts/run_infer_and_eval_batching.py" \
        --model_config_name "${MODEL_CFG}" \
        --stage "${STAGE}" \
        --instance_id "${IDS}" \
        --trial_num "${TRIAL_NUM}" \
        --thread_num 1 \
        --url_map_mode "${URL_MAP_MODE}" \
        > "${LOG}" 2>&1 &
    WORKER_PIDS+=($!)
    echo "[parallel] lane $i: tasks ${START}..${END} (${IDS:0:50}...) -> :${LOCAL_PORT} pid $!"
done

# ---- 4. wait for all workers ----
echo ""
echo "[parallel] ${#WORKER_PIDS[@]} workers running. Streaming progress:"
echo "  tail -F /tmp/widesearch_worker{0..$((NGPU-1))}.log"
echo ""

# Wait on each PID, surface non-zero exits but don't abort the whole batch.
FAILED=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "[parallel] worker pid ${pid} exited non-zero" >&2
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "[parallel] all workers done (${FAILED} non-zero exit(s))"
echo "  results: ${REPO_ROOT}/data/output/${MODEL_CFG}_${INSTANCE_PREFIX}_*_response.jsonl"
echo "  evals:   ${REPO_ROOT}/data/output/${MODEL_CFG}_${INSTANCE_PREFIX}_*_eval_result.json"
exit "${FAILED}"
