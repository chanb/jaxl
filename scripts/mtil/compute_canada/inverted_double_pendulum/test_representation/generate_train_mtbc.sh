#!/bin/bash
#SBATCH --account=def-schuurma
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-14
#SBATCH --output=/home/chanb/scratch/jaxl/run_reports/inverted_double_pendulum/%j.out

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate
`sed -n "${SLURM_ARRAY_TASK_ID}p" < export-generate_train_mtbc-gravity-representation_analysis.dat`
echo ${SLURM_ARRAY_TASK_ID}

echo "Current working directory is `pwd`"
echo "Running on hostname `hostname`"

echo "Starting run at: `date`"
python3 /home/chanb/scratch/jaxl/jaxl/main.py \
  --config_path=${config_path} \
  --run_seed=${run_seed}
echo "Program test finished with exit code $? at: `date`"
