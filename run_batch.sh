#!/bin/bash
# ============================================================================
# Sweep Agatha over beta, gate target and gate value for one task. Each
# configuration is run for every seed by run_gate_ablation.sh.
#
# Usage:
#   ./run_batch.sh <task> [device]
#
# Example:
#   ./run_batch.sh sst2 cuda:0
# ============================================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <task> [device]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task=${1}
device=${2:-cuda:0}

for beta in 1 5 10 15 50 100; do
    for target in hopfield both; do
        for gate in 0 0.01 0.1 0.3 0.5 0.7 0.9 1.0; do
            bash "${SCRIPT_DIR}/run_gate_ablation.sh" "${task}" "${gate}" "${target}" "${beta}" "${device}"
        done
    done
done
