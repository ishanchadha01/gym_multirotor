#!/bin/bash
#SBATCH -J acmpc_nowind                                                 # Job name
#SBATCH --account=gts-ychen3538                                 # Tracking account
#SBATCH -N1 --gres=gpu:A100:1                                   # Number of nodes and cores per node required
#SBATCH -t720                                                   # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                               # QOS Name
#SBATCH -o acmpc_nowind.out                                   # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                              # Mail preferences
#SBATCH --mail-user=ichadha3@gatech.edu                         # E-mail address for notifications
cd /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/rlr/gym_multirotor/examples         # Change to working directory
module load cuda
conda activate rlr
python play_multirotor.py