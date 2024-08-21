import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cc_utils.configs import EXPERIMENTS
from cc_utils.constants import (
    CONFIG_DIR,
    LOG_DIR,
    RUN_REPORT_DIR,
    REPO_PATH,
    CC_ACCOUNT,
    PLOT_DIR,
)

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.join(RUN_REPORT_DIR, "plot_results"), exist_ok=True)

num_runs = 0
dat_content = ""
for runs_dir in os.listdir(LOG_DIR):
    result_path = os.path.join(LOG_DIR, runs_dir)
    save_path = os.path.join(PLOT_DIR, runs_dir)
    dat_content += (
        "export results_dir={} save_path={} key=accuracies context=none \n".format(
            result_path,
            save_path,
        )
    )
    num_runs += 1

    dat_content += (
        "export results_dir={} save_path={} key=accuracies context=last \n".format(
            result_path,
            save_path,
        )
    )
    num_runs += 1

    dat_content += (
        "export results_dir={} save_path={} key=losses context=none \n".format(
            result_path,
            save_path,
        )
    )
    num_runs += 1

    dat_content += (
        "export results_dir={} save_path={} key=losses context=last \n".format(
            result_path,
            save_path,
        )
    )
    num_runs += 1

with open(os.path.join(CONFIG_DIR, "plot_results.dat"), "w+") as f:
    f.writelines(dat_content)

sbatch_content = ""
sbatch_content += "#!/bin/bash\n"
sbatch_content += "#SBATCH --account={}\n".format(CC_ACCOUNT)
sbatch_content += "#SBATCH --time=00:10:00\n"
sbatch_content += "#SBATCH --cpus-per-task=1\n"
sbatch_content += "#SBATCH --mem=12G\n"
sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
sbatch_content += "#SBATCH --output={}/%j.out\n".format(
    os.path.join(RUN_REPORT_DIR, "plot_results")
)
sbatch_content += "module load python/3.10\n"
sbatch_content += "module load mujoco\n"
sbatch_content += "source ~/icl_env/bin/activate\n"
sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
sbatch_content += " < {}`\n".format(os.path.join(CONFIG_DIR, "plot_results.dat"))
sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
sbatch_content += 'echo "Current working directory is `pwd`"\n'
sbatch_content += 'echo "Running on hostname `hostname`"\n'
sbatch_content += "echo ${learner_path}\n"
sbatch_content += 'echo "Starting run at: `date`"\n'
sbatch_content += "python3 {}/experiments/cc_utils/plot_results.py \\\n".format(
    REPO_PATH
)
sbatch_content += "  --results_dir=${results_dir} \\\n"
sbatch_content += "  --save_path=${save_path} \\\n"
sbatch_content += "  --key=${key} \\\n"
sbatch_content += "  --context=${context} \n"
sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

with open(
    "./sbatch_plot_results.sh",
    "w+",
) as f:
    f.writelines(sbatch_content)
