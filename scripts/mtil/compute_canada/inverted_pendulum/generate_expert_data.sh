#!/bin/bash
#SBATCH --account=def-schuurma
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --array=1-100
#SBATCH --output=/home/chanb/scratch/jaxl/run_reports/%j.out

module load python/3.9
module load mujoco
source ~/jaxl_env/bin/activate
`sed -n "${SLURM_ARRAY_TASK_ID}p" < export-generate_expert_data-inverted_pendulum.dat`
echo ${SLURM_ARRAY_TASK_ID}

echo "Current working directory is `pwd`"
echo "Running on hostname `hostname`"

echo "Starting run at: `date`"
python3 /home/chanb/scratch/jaxl/jaxl/evaluate_rl_agents.py \
  --save_path=${save_path} \
  --run_path=${run_path} \
  --run_seed=${run_seed} \
  --env_seed=${env_seed} \
  --num_episodes=${num_episodes} \
  --buffer_size=${buffer_size}

echo "Program test finished with exit code $? at: `date`"