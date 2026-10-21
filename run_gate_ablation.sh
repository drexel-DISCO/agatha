#!/bin/bash
# ============================================================================
# Train and evaluate Agatha on one task (or a group of tasks) for each seed.
#
# Usage:
#   ./run_gate_ablation.sh <task|group> [gate] [target] [beta] [device]
#
#   gate    initial residual gate value g   (default: 0.1)
#   target  hopfield | ffn | both           (default: both)
#   beta    Hopfield inverse temperature    (default: 50)
#   device  torch device                    (default: cuda:0)
#
# Example:
#   ./run_gate_ablation.sh sst2 0.1 hopfield 50 cuda:0
#
# Run ./run_gate_ablation.sh help for the list of tasks and groups.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python}

# ============================================================================
# Seeds (edit to change which seeds are run)
# ============================================================================

SEEDS=(42 123 456)

# ============================================================================
# Global evaluation settings
# ============================================================================

DEVICE=${5:-cuda:0}
PGD_STEPS=20
EPSILONS="0.05,0.5,1.0,2.0,3.0,5.0"

# AGATHA-specific hyperparameters
BETA=${4:-50}       # inverse temperature
GATE=${2:-0.1}      # gate value (g)
TARGET=${3:-both}   # gate target (hopfield/ffn/both)
MAX_ITERS=100       # Hopfield iterations
TR_EP=30            # training epochs
TR_PAT=20           # training patience

# Output directories (relative to the current working directory)
DIR_AGATHA_MODELS="./models_gate_ablation/${TARGET}"
DIR_AGATHA_OUT="./results_gate_ablation/${TARGET}"

# ============================================================================
# Per-task architecture: hidden size, FFN size, layers, heads.
# MultiRC and WSC use the scaled model; all other tasks use the base model.
# ============================================================================

get_arch() {
    case "$1" in
        multirc|wsc) echo "256 1024 4 4" ;;
        *)           echo "128 512 2 2" ;;
    esac
}

# ============================================================================
# AGATHA
# ============================================================================

run_agatha() {
    local TASK=$1
    local H I L A
    read -r H I L A <<< "$(get_arch "${TASK}")"

    local GATE_MOD="${GATE/./p}"

    for SEED in "${SEEDS[@]}"; do
        local MODEL_DIR="${DIR_AGATHA_MODELS}/${TASK}/seed${SEED}/b${BETA}_g${GATE_MOD}"
        local OUT_DIR="${DIR_AGATHA_OUT}/hidden${H}_layer${L}_head${A}/${TASK}/seed${SEED}"
        mkdir -p "${MODEL_DIR}" "${OUT_DIR}"

        echo ""
        echo "======================================================================"
        echo "[AGATHA] TASK: ${TASK} | GATE: ${GATE} | TARGET: ${TARGET} | SEED: ${SEED}"
        echo "ARCH: ${H}/${L}/${A} | EPOCHS: ${TR_EP} | PATIENCE: ${TR_PAT}"
        echo "======================================================================"

        "${PYTHON}" "${SCRIPT_DIR}/agatha.py" \
            --hidden_size "${H}" \
            --num_heads "${A}" \
            --num_layers "${L}" \
            --intermediate_size "${I}" \
            --task "${TASK}" \
            --beta "${BETA}" \
            --max_iterations "${MAX_ITERS}" \
            --train True \
            --pretrained True \
            --epochs "${TR_EP}" \
            --patience "${TR_PAT}" \
            --gate_mode learned \
            --gate_init "${GATE}" \
            --gate_target "${TARGET}" \
            --pgd_steps "${PGD_STEPS}" \
            --pgd_norm l2 \
            --eval_epsilons "${EPSILONS}" \
            --seed "${SEED}" \
            --model_dir "${MODEL_DIR}" \
            --out_dir "${OUT_DIR}" \
            --device "${DEVICE}" \
            --track_basin_ridge False
    done
}

# ============================================================================
# Run AGATHA
# ============================================================================

run_task_all() {
    local TASK=$1

    echo ""
    echo "##################################################################"
    echo "  TASK: ${TASK}  |  DEVICE: ${DEVICE}  |  SEEDS: ${SEEDS[*]}"
    echo "  Running: AGATHA"
    echo "##################################################################"

    run_agatha "${TASK}"

    echo ""
    echo "##################################################################"
    echo "  TASK ${TASK} COMPLETE"
    echo "##################################################################"
}

# ============================================================================
# Grouped runners
# ============================================================================

run_glue_standard() {
    for t in sst2 cola mnli qnli qqp; do run_task_all "$t"; done
}

run_glue_small() {
    for t in wnli mrpc rte; do run_task_all "$t"; done
}

run_glue_all() {
    run_glue_small
    run_glue_standard
}

run_superglue() {
    for t in boolq cb multirc wsc; do run_task_all "$t"; done
}

run_all() {
    run_glue_all
    run_superglue
    echo ""
    echo "=============================================="
    echo "ALL EXPERIMENTS COMPLETE"
    echo "=============================================="
}

# ============================================================================
# Usage
# ============================================================================

usage() {
    echo "Agatha experiment script"
    echo ""
    echo "Usage: $0 <task|group> [gate] [target] [beta] [device]"
    echo ""
    echo "Individual tasks:"
    echo "  sst2, cola, mnli, qnli, qqp, mrpc, rte, wnli   (GLUE)"
    echo "  boolq, cb, multirc, wsc                        (SuperGLUE)"
    echo ""
    echo "Grouped runs:"
    echo "  glue_standard    SST-2, CoLA, MNLI, QNLI, QQP"
    echo "  glue_small       WNLI, MRPC, RTE"
    echo "  glue_all         All GLUE tasks"
    echo "  superglue        BoolQ, CB, MultiRC, WSC"
    echo "  all              Everything"
    echo ""
    echo "Seeds (edit SEEDS at the top of the script): ${SEEDS[*]}"
}

# ============================================================================
# Main
# ============================================================================

case "${1:-help}" in
    sst2|cola|mnli|qnli|qqp|mrpc|rte|wnli|boolq|cb|multirc|wsc)
        run_task_all "$1" ;;

    glue_standard)  run_glue_standard ;;
    glue_small)     run_glue_small ;;
    glue_all)       run_glue_all ;;
    superglue)      run_superglue ;;
    all)            run_all ;;

    help|--help|-h) usage ;;
    *)              echo "Unknown command: $1"; usage; exit 1 ;;
esac
