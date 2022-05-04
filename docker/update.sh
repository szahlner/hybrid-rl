#!/bin/bash
# Select right directory
cd "/home/hybrid-rl/"

# Remove existing
rm -r ./hybrid-rl
rm -r ./shadowhand-gym

# Clone new
git clone https://github.com/szahlner/hybrid-rl.git
git clone https://github.com/szahlner/shadowhand-gym.git
pip3 install -e shadowhand-gym