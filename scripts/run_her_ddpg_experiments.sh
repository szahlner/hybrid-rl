#!/bin/bash
cd "/home/hybrid-rl/hybrid-rl"

# FetchPush-v1 DONE
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40

# python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 15 --model-training-freq 1000
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 15 --model-training-freq 1000
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40 --model-based --model-dim-chunk 15 --model-training-freq 1000

# python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 15
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 15
# python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 400 --n-cycles 40 --model-based --model-type stochastic --model-dim-chunk 15

# ShadowHandBlock-v1
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40

# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40 --model-based --model-training-freq 1000

# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40 --model-based --model-type stochastic
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40 --model-based --model-type stochastic
# python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40 --model-based --model-type stochastic

# FetchPush-v1 - MPI version
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 50 --n-cycles 40
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 50 --n-cycles 40 --model-based --model-dim-chunk 15
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 123 --n-epochs 50 --n-cycles 40 --model-based --model-dim-chunk 15 --model-type stochastic --model-stochastic-percentage 0.90

mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 50 --n-cycles 40
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 50 --n-cycles 40 --model-based --model-dim-chunk 15
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 1234 --n-epochs 50 --n-cycles 40 --model-based --model-dim-chunk 15 --model-type stochastic --model-stochastic-percentage 0.90

mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 50 --n-cycles 40
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 50 --n-cycles 40 --model-based --model-dim-chunk 15
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name FetchPush-v1 --seed 12345 --n-epochs 50 --n-cycles 40 --model-based --model-dim-chunk 15 --model-type stochastic --model-stochastic-percentage 0.90

# ShadowHandBlock-v1 - MPI version
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40 --alpah 0.01
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40 --alpah 0.01 --model-based --model-training-freq 200
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 123 --n-epochs 1000 --n-cycles 40 --alpah 0.01 --model-based --model-training-freq 200 --model-type stochastic --model-stochastic-percentage 0.80

mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40 --alpah 0.01
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40 --alpah 0.01 --model-based --model-training-freq 200
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 1234 --n-epochs 1000 --n-cycles 40 --alpah 0.01 --model-based --model-training-freq 200 --model-type stochastic --model-stochastic-percentage 0.80

mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40 --alpah 0.01
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40 --alpah 0.01 --model-based --model-training-freq 200
mpirun -np 8 --allow-run-as-root python train_her_ddpg.py --env-name ShadowHandBlock-v1 --seed 12345 --n-epochs 1000 --n-cycles 40 --alpah 0.01 --model-based --model-training-freq 200 --model-type stochastic --model-stochastic-percentage 0.80