#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

# DDPG Hopper-v2
# python train_ddpg.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000
# python train_ddpg.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000
# python train_ddpg.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000

# python train_ddpg.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --model-based
# python train_ddpg.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --model-based
# python train_ddpg.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --model-based

python train_ddpg.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --model-based --model-type stochastic --model-stochastic-percentage 0.90
python train_ddpg.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --model-based --model-type stochastic --model-stochastic-percentage 0.90
python train_ddpg.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --model-based --model-type stochastic --model-stochastic-percentage 0.90

# DDPG Walker2d-v2
# python train_ddpg.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000
# python train_ddpg.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000
# python train_ddpg.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000

# python train_ddpg.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --model-based
# python train_ddpg.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --model-based
# python train_ddpg.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --model-based

python train_ddpg.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --model-based --model-type stochastic --model-stochastic-percentage 0.60
python train_ddpg.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --model-based --model-type stochastic --model-stochastic-percentage 0.60
python train_ddpg.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --model-based --model-type stochastic --model-stochastic-percentage 0.60

# DDPG HalfCheetah-v2
# python train_ddpg.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000
# python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000
# python train_ddpg.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000

# python train_ddpg.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --model-based
# python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --model-based
# python train_ddpg.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --model-based

python train_ddpg.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --model-based --model-type stochastic --model-stochastic-percentage 0.83
python train_ddpg.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --model-based --model-type stochastic --model-stochastic-percentage 0.83
python train_ddpg.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --model-based --model-type stochastic --model-stochastic-percentage 0.83