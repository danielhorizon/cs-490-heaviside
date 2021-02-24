#!/bin/bash
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=48:00:00

#SBATCH -o run-logs/af1-0.1-%j.txt
#SBATCH -e run-logs/af1-0.1-%j.err
#SBATCH --job-name=af1-0.1

#SBATCH --mail-user=taesoo.d.lee@yale.edu


#SBATCH --ntasks=4 --nodes=1
#SBATCH --mem-per-cpu=10G


# Running Thresholding 
python3 grace-main-thresh.py --thresh 0.1 --run_name grace-af1-0.1 --gpu 0 /home/tdl29/project/


# currently running 
# python3 grace-main-thresh.py  --thresh 0.1 --run_name grace-af1-0.1 --gpu 0 /home/tdl29/project/
# python3 grace-main-thresh.py  --thresh 0.2 --run_name grace-af1-0.2 --gpu 0 /home/tdl29/project/
# 

# python3 grace-main-thresh.py  --thresh 0.4 --run_name grace-af1-0.4 --gpu 0 /home/tdl29/project/
# python3 grace-main-thresh.py  --thresh 0.5 --run_name grace-af1-0.5 --gpu 0 /home/tdl29/project/

# currently funning: 0.1, 0.3, 0.4, 0.5