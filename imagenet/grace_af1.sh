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
