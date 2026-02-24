#!/bin/bash
# Experiment commands to run

# Old-style thinking (default)
python eval_frontierscience.py --model claude-sonnet-4-6 --domain bio --adaptive-thinking

# Adaptive thinking (new style)
#python eval_frontierscience.py --model claude-sonnet-4-6 --domain bio --loops 1 --n-samples 7:8 --adaptive-thinking