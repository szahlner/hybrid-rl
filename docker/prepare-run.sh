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
cp "./scripts/run_experiments.sh" "./run_experiments.sh"

echo "######################################################"
echo "ALL DONE, you are now prepared to run the experiments!"
echo "######################################################"