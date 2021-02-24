#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=alexnet-baseline-ce
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

#SBATCH -o run-logs/baseline-ce_%j.txt
#SBATCH -e run-logs/baseline-ce_%j.err
#SBATCH --ntasks=4 --nodes=1
#SBATCH --mem-per-cpu=10G

#SBATCH --mail-user=taesoo.d.lee@yale.edu


# Running CE early-stopping 
python3 grace-main.py  --gpu 0 --run_name baseline-ce /home/tdl29/project/

