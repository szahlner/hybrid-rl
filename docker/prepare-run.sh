#!/bin/bash

echo "######################################################"
echo "Start preparing run_scripts"
echo "copy training files"
# Enter directory and copy training files
cd "/home/hybrid-rl/hybrid-rl"
cp "./experiments/train_her_ddpg.py" "./train_her_ddpg.py"
cp "./experiments/train_her_sac.py" "./train_her_sac.py"
cp "./experiments/train_ddpg_her.py" "./train_ddpg_her.py"
cp "./experiments/train_sac_her.py" "./train_sac_her.py"
cp "./experiments/train_ddpg.py" "./train_ddpg.py"
cp "./experiments/train_sac.py" "./train_sac.py"
cp "./experiments/train_redq_sac.py" "./train_redq_sac.py"

echo "copy script files"
# Copy script files
cp "./scripts/run_ddpg_experiments.sh" "./run_ddpg_experiments.sh"
cp "./scripts/run_sac_experiments.sh" "./run_sac_experiments.sh"
cp "./scripts/run_her_ddpg_experiments.sh" "./run_her_ddpg_experiments.sh"
cp "./scripts/run_her_sac_experiments.sh" "./run_her_sac_experiments.sh"
cp "./scripts/run_any_experiments_0.sh" "./run_any_experiments_0.sh"
cp "./scripts/run_any_experiments_1.sh" "./run_any_experiments_1.sh"
cp "./scripts/run_any_experiments_2.sh" "./run_any_experiments_2.sh"
cp "./scripts/run_any_experiments_3.sh" "./run_any_experiments_3.sh"

echo "set permissions"
# Set permissions
chmod +x "./run_ddpg_experiments.sh"
chmod +x "./run_sac_experiments.sh"
chmod +x "./run_her_ddpg_experiments.sh"
chmod +x "./run_her_sac_experiments.sh"
chmod +x "./run_any_experiments_0.sh"
chmod +x "./run_any_experiments_1.sh"
chmod +x "./run_any_experiments_2.sh"
chmod +x "./run_any_experiments_3.sh"

echo "ALL DONE, you are now prepared to run the experiments!"
echo "######################################################"