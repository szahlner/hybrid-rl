#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

# SAC Hopper-v2
python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1
python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1
python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1

# python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --n-batches 2
# python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --n-batches 2
# python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --n-batches 2

# python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based
# python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based
# python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based

# python train_sac.py --env-name Hopper-v2 --seed 123 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic --model-stochastic-percentage 0.95
# python train_sac.py --env-name Hopper-v2 --seed 1234 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic --model-stochastic-percentage 0.95
# python train_sac.py --env-name Hopper-v2 --seed 12345 --max-timesteps 125000 --automatic-entropy-tuning --target-entropy -1 --model-based --model-type stochastic --model-stochastic-percentage 0.95

# SAC Walker2d-v2
python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3

# python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --n-batches 2
# python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --n-batches 2
# python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --n-batches 2

# python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based
# python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based
# python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based

# python train_sac.py --env-name Walker2d-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.60
# python train_sac.py --env-name Walker2d-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.60
# python train_sac.py --env-name Walker2d-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.60

# SAC HalfCheetah-v2
python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3
python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3

# python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --n-batches 2
# python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --n-batches 2
# python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --n-batches 2

# python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based
# python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based
# python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based

# python train_sac.py --env-name HalfCheetah-v2 --seed 123 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.95
# python train_sac.py --env-name HalfCheetah-v2 --seed 1234 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.95
# python train_sac.py --env-name HalfCheetah-v2 --seed 12345 --max-timesteps 400000 --automatic-entropy-tuning --target-entropy -3 --model-based --model-type stochastic --model-stochastic-percentage 0.95

# SAC Ant-v2
python train_sac.py --env-name Ant-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4
python train_sac.py --env-name Ant-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4
python train_sac.py --env-name Ant-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4

# python train_sac.py --env-name Ant-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2
# python train_sac.py --env-name Ant-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2
# python train_sac.py --env-name Ant-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2

# python train_sac.py --env-name Ant-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
# python train_sac.py --env-name Ant-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
# python train_sac.py --env-name Ant-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based

# python train_sac.py --env-name Ant-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.10
# python train_sac.py --env-name Ant-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.10
# python train_sac.py --env-name Ant-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.10

# SAC AntTruncated-v2
python train_sac.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4
python train_sac.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4
python train_sac.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4

# python train_sac.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2
# python train_sac.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2
# python train_sac.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2

# python train_sac.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
# python train_sac.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based
# python train_sac.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based

# python train_sac.py --env-name AntTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.10
# python train_sac.py --env-name AntTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.10
# python train_sac.py --env-name AntTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --model-based --model-type stochastic --model-stochastic-percentage 0.10

# SAC InvertedPendulum-v2
python train_sac.py --env-name InvertedPendulum-v2 --seed 123 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05
python train_sac.py --env-name InvertedPendulum-v2 --seed 1234 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05
python train_sac.py --env-name InvertedPendulum-v2 --seed 12345 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05

# python train_sac.py --env-name InvertedPendulum-v2 --seed 123 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --n-batches 2
# python train_sac.py --env-name InvertedPendulum-v2 --seed 1234 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --n-batches 2
# python train_sac.py --env-name InvertedPendulum-v2 --seed 12345 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --n-batches 2

# python train_sac.py --env-name InvertedPendulum-v2 --seed 123 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --model-based --model-training-freq 500
# python train_sac.py --env-name InvertedPendulum-v2 --seed 1234 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --model-based --model-training-freq 500
# python train_sac.py --env-name InvertedPendulum-v2 --seed 12345 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --model-based --model-training-freq 500

# python train_sac.py --env-name InvertedPendulum-v2 --seed 123 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --model-based --model-training-freq 500 --model-type stochastic --model-stochastic-percentage 1.00
# python train_sac.py --env-name InvertedPendulum-v2 --seed 1234 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --model-based --model-training-freq 500 --model-type stochastic --model-stochastic-percentage 1.00
# python train_sac.py --env-name InvertedPendulum-v2 --seed 12345 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.05 --model-based --model-training-freq 500 --model-type stochastic --model-stochastic-percentage 1.00

# SAC InvertedDoublePendulum-v2
python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 123 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5
python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 1234 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5
python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 12345 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5

# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 123 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --n-batches 2
# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 1234 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --n-batches 2
# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 12345 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --n-batches 2

# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 123 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --model-based --model-training-freq 500
# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 1234 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --model-based --model-training-freq 500
# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 12345 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --model-based --model-training-freq 500

# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 123 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --model-based --model-training-freq 500 --model-type stochastic --model-stochastic-percentage 0.80
# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 1234 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --model-based --model-training-freq 500 --model-type stochastic --model-stochastic-percentage 0.80
# python train_sac.py --env-name InvertedDoublePendulum-v2 --seed 12345 --max-timesteps 50000 --automatic-entropy-tuning --target-entropy -0.5 --model-based --model-training-freq 500 --model-type stochastic --model-stochastic-percentage 0.80

# SAC HumanoidTruncated-v2
python train_sac.py --env-name HumanoidTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2
python train_sac.py --env-name HumanoidTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2
python train_sac.py --env-name HumanoidTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2

# python train_sac.py --env-name HumanoidTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2
# python train_sac.py --env-name HumanoidTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2
# python train_sac.py --env-name HumanoidTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -4 --n-batches 2

# python train_sac.py --env-name HumanoidTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based
# python train_sac.py --env-name HumanoidTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based
# python train_sac.py --env-name HumanoidTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based

# python train_sac.py --env-name HumanoidTruncated-v2 --seed 123 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based --model-type stochastic --model-stochastic-percentage 0.83
# python train_sac.py --env-name HumanoidTruncated-v2 --seed 1234 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based --model-type stochastic --model-stochastic-percentage 0.83
# python train_sac.py --env-name HumanoidTruncated-v2 --seed 12345 --max-timesteps 300000 --automatic-entropy-tuning --target-entropy -2 --model-based --model-type stochastic --model-stochastic-percentage 0.83