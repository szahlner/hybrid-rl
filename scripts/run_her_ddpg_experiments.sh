#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

# FetchPush-v1 DONE
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40

# python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 19 --model-training-freq 1000
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 19 --model-training-freq 1000
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 19 --model-training-freq 1000

# python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 30
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 30
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 30

# ShadowHandBlock-v1
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40
python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000

# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000

# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40 --model-based --model-type stochastic
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40 --model-based --model-type stochastic
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40 --model-based --model-type stochastic