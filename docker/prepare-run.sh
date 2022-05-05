#!/bin/bash
# Enter directory and copy training files
cd "/home/hybrid-rl/hybrid-rl"
cp "./experiments/train_her_ddpg.py" "./train_her_ddpg.py"
cp "./experiments/train_her_sac.py" "./train_her_sac.py"
cp "./experiments/train_ddpg_her.py" "./train_ddpg_her.py"
cp "./experiments/train_sac_her.py" "./train_sac_her.py"
cp "./experiments/train_ddpg.py" "./train_ddpg.py"
cp "./experiments/train_sac.py" "./train_sac.py"
cp "./experiments/train_redq_sac.py" "./train_redq_sac.py"

# Copy script files
cp "./scripts/run_ddpg_experiments.sh" "./run_ddpg_experiments.sh"
cp "./scripts/run_sac_experiments.sh" "./run_sac_experiments.sh"
cp "./scripts/run_her_ddpg_experiments.sh" "./run_her_ddpg_experiments.sh"
cp "./scripts/run_her_sac_experiments.sh" "./run_her_sac_experiments.sh"

echo "######################################################"
echo "ALL DONE, you are now prepared to run the experiments!"
echo "######################################################"