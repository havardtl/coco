#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-mh-ikom
#SBATCH --time=02:0:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB
#SBATCH --job-name="boco"
#SBATCH --output=slurm_log.txt
#SBATCH --mail-user=havard.t.lindholm@ntnu.no
#SBATCH --mail-type=END,FAIL

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate coco

boco_segment_initial.py --cores 1 &> last_run.txt

