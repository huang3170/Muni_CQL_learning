#!/usr/bin/env bash
set -euo pipefail

python prepare_hybrid_replay_data.py make-demo \
  --output-dir demo_train \
  --episodes 40 \
  --seed 2026

python prepare_hybrid_replay_data.py make-demo \
  --output-dir demo_valid \
  --episodes 12 \
  --seed 3026

python simulator_online_train.py \
  --train-replay-dir demo_train \
  --valid-replay-dir demo_valid \
  --output-dir demo_output \
  --train-episodes 50 \
  --replay-warmup 500 \
  --batch-size 256 \
  --evaluation-interval-episodes 10 \
  --evaluation-max-episodes 12 \
  --device auto
