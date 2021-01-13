!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=alexnet-heaviside-af1
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH -o heaviside_%j.txt
#SBATCH -e heaviside_%j.err

#SBATCH --mail-user=taesoo.d.lee@yale.edu


#SBATCH --ntasks=4 --nodes=1
#SBATCH --mem-per-cpu=10G


# Running A-F1 early stopping 
python3 grace-main-af1.py --gpu 0 /home/tdl29/project/

# Running Thresholding 
python3 grace-main-af1.py --thresh 0.1 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.125 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.2 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.3 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.4 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.5 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.6 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.7 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.8 --gpu 0 /home/tdl29/project/
python3 grace-main-af1.py --thresh 0.9 --gpu 0 /home/tdl29/project/