#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

# SAC Hopper-v2
python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1
python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1
python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1
python train_sac.py --env-name Hopper-v2 --seed 123456 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1
python train_sac.py --env-name Hopper-v2 --seed 1234567 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1

python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based
python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based
python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based
python train_sac.py --env-name Hopper-v2 --seed 123456 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based
python train_sac.py --env-name Hopper-v2 --seed 1234567 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based

python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic
python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic
python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic
python train_sac.py --env-name Hopper-v2 --seed 123456 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic
python train_sac.py --env-name Hopper-v2 --seed 1234567 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic