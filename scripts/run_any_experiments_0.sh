#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

# DDPG AntTruncated-v2
python train_ddpg.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000
python train_ddpg.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000
python train_ddpg.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000

python train_ddpg.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --n-batches 2
python train_ddpg.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --n-batches 2
python train_ddpg.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --n-batches 2

python train_ddpg.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --model-based
python train_ddpg.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --model-based
python train_ddpg.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --model-based

# SAC AntTruncated-v2
python train_sac.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4
python train_sac.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4
python train_sac.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4

python train_sac.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2
python train_sac.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2
python train_sac.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2

python train_sac.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
python train_sac.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
python train_sac.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
