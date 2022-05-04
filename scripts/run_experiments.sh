#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

python train_sac.py --env-name Hopper-v2 --seed 123456 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1