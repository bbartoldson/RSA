#!/bin/bash
# Experiment commands to run
# Runs 10 incremental sampling calls (0:1, 1:2, ..., 9:10),
# pushing the rollouts file after each call.

ROLLOUTS_FILE="eval/frontierscience/claude-sonnet-4-6_thinking-adaptive/rollouts_k_4_N_16.jsonl"

for i in $(seq 0 9); do
    START=$i
    END=$((i + 1))
    echo "=== Running --n-samples ${START}:${END} ==="
    python eval_frontierscience.py --model claude-sonnet-4-6 --domain bio --adaptive-thinking --n-samples ${START}:${END}

    echo "=== Pushing rollouts after sample ${START}:${END} ==="
    git add "$ROLLOUTS_FILE"
    git commit -m "Add rollouts for sample ${START}:${END}"
    git push
done
