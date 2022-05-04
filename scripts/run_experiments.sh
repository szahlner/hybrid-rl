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

# DDPG Hopper-v2
python train_ddpg.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000
python train_ddpg.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000
python train_ddpg.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000
python train_ddpg.py --env-name Hopper-v2 --seed 123456 --max-timesteps 125000
python train_ddpg.py --env-name Hopper-v2 --seed 1234567 --max-timesteps 125000

python train_ddpg.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --model-based
python train_ddpg.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --model-based
python train_ddpg.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --model-based
python train_ddpg.py --env-name Hopper-v2 --seed 123456 --max-timesteps 125000 --model-based
python train_ddpg.py --env-name Hopper-v2 --seed 1234567 --max-timesteps 125000 --model-based

python train_ddpg.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --model-based --model-type stochastic
python train_ddpg.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --model-based --model-type stochastic
python train_ddpg.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --model-based --model-type stochastic
python train_ddpg.py --env-name Hopper-v2 --seed 123456 --max-timesteps 125000 --model-based --model-type stochastic
python train_ddpg.py --env-name Hopper-v2 --seed 1234567 --max-timesteps 125000 --model-based --model-type stochastic

# SAC Walker2d-v2
python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name Walker2d-v2 --seed 123456 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name Walker2d-v2 --seed 1234567 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3

python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based
python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based
python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based
python train_sac.py --env-name Walker2d-v2 --seed 123456 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based
python train_sac.py --env-name Walker2d-v2 --seed 1234567 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based

python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic
python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic
python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic
python train_sac.py --env-name Walker2d-v2 --seed 123456 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic
python train_sac.py --env-name Walker2d-v2 --seed 1234567 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic

# DDPG Walker2d-v2
python train_ddpg.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000
python train_ddpg.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000
python train_ddpg.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000
python train_ddpg.py --env-name Walker2d-v2 --seed 123456 --max-timesteps 300000
python train_ddpg.py --env-name Walker2d-v2 --seed 1234567 --max-timesteps 300000

python train_ddpg.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --model-based
python train_ddpg.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --model-based
python train_ddpg.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --model-based
python train_ddpg.py --env-name Walker2d-v2 --seed 123456 --max-timesteps 300000 --model-based
python train_ddpg.py --env-name Walker2d-v2 --seed 1234567 --max-timesteps 300000 --model-based

python train_ddpg.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --model-based --model-type stochastic
python train_ddpg.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --model-based --model-type stochastic
python train_ddpg.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --model-based --model-type stochastic
python train_ddpg.py --env-name Walker2d-v2 --seed 123456 --max-timesteps 300000 --model-based --model-type stochastic
python train_ddpg.py --env-name Walker2d-v2 --seed 1234567 --max-timesteps 300000 --model-based --model-type stochastic

# SAC HalfCheetah-v2
python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name HalfCheetah-v2 --seed 123456 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name HalfCheetah-v2 --seed 1234567 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3

python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based
python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based
python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based
python train_sac.py --env-name HalfCheetah-v2 --seed 123456 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based
python train_sac.py --env-name HalfCheetah-v2 --seed 1234567 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based

python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic
python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic
python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic
python train_sac.py --env-name HalfCheetah-v2 --seed 123456 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic
python train_sac.py --env-name HalfCheetah-v2 --seed 1234567 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic

# DDPG HalfCheetah-v2
python train_ddpg.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000
python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000
python train_ddpg.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000
python train_ddpg.py --env-name HalfCheetah-v2 --seed 123456 --max-timesteps 400000
python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234567 --max-timesteps 400000

python train_ddpg.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --model-based
python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --model-based
python train_ddpg.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --model-based
python train_ddpg.py --env-name HalfCheetah-v2 --seed 123456 --max-timesteps 400000 --model-based
python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234567 --max-timesteps 400000 --model-based

python train_ddpg.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --model-based --model-type stochastic
python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --model-based --model-type stochastic
python train_ddpg.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --model-based --model-type stochastic
python train_ddpg.py --env-name HalfCheetah-v2 --seed 123456 --max-timesteps 400000 --model-based --model-type stochastic
python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234567 --max-timesteps 400000 --model-based --model-type stochastic