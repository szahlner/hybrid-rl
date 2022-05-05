#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

# FetchPush-v1
python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40 --model-dim-chunk 19
python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40 --model-dim-chunk 19
python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40 --model-dim-chunk 19

python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 19
python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 19
python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 19

# python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 19
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 19
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 19

# ShadowHandBlock-v1
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 3750 --n-cycles 40
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 3750 --n-cycles 40
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 3750 --n-cycles 40

python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 3750 --n-cycles 40 --model-based
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 3750 --n-cycles 40 --model-based
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 3750 --n-cycles 40 --model-based

# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 3750 --n-cycles 40 --model-based --model-type stochastic
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 3750 --n-cycles 40 --model-based --model-type stochastic
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 3750 --n-cycles 40 --model-based --model-type stochastic