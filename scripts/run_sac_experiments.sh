#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

# SAC Hopper-v2
# python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1
# python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1
# python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1

# python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based
# python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based
# python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based

python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic --model-stochastic-percentage 0.95
python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic --model-stochastic-percentage 0.95
python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic --model-stochastic-percentage 0.95

# SAC Walker2d-v2
# python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3
# python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3
# python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3

# python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based
# python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based
# python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based

python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.60
python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.60
python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.60

# SAC HalfCheetah-v2
# python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3
# python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3
# python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3

# python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based
# python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based
# python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based

python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.95
python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.95
python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.95

# SAC Ant-v2
python train_sac.py --env-name Ant-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4
python train_sac.py --env-name Ant-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4
python train_sac.py --env-name Ant-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4

python train_sac.py --env-name Ant-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
python train_sac.py --env-name Ant-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
python train_sac.py --env-name Ant-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based

# python train_sac.py --env-name Ant-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.83
# python train_sac.py --env-name Ant-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.83
# python train_sac.py --env-name Ant-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.83

# SAC Humanoid-v2
python train_sac.py --env-name Humanoid-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2
python train_sac.py --env-name Humanoid-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2
python train_sac.py --env-name Humanoid-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2

python train_sac.py --env-name Humanoid-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based
python train_sac.py --env-name Humanoid-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based
python train_sac.py --env-name Humanoid-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based

# python train_sac.py --env-name Humanoid-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based --model-type stochastic --model-stochastic-percentage 0.83
# python train_sac.py --env-name Humanoid-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based --model-type stochastic --model-stochastic-percentage 0.83
# python train_sac.py --env-name Humanoid-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based --model-type stochastic --model-stochastic-percentage 0.83