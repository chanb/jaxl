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
)

sbatch_dir = "./sbatch_scripts"
os.makedirs(sbatch_dir, exist_ok=True)

run_all_content = "#!/bin/bash\n"
for exp_name, exp_config in EXPERIMENTS.items():
    os.makedirs(os.path.join(RUN_REPORT_DIR, "eval"), exist_ok=True)
    result_dir = os.path.join(LOG_DIR, exp_name)
    num_runs = 0
    dat_content = ""

    for variant in os.listdir(result_dir):
        learner_path = os.path.join(result_dir, variant)

        num_runs += 1
        dat_content += "export learner_path={} \n".format(
            learner_path,
        )

    with open(os.path.join(CONFIG_DIR, "eval-{}.dat".format(exp_name)), "w+") as f:
        f.writelines(dat_content)

    sbatch_content = ""
    sbatch_content += "#!/bin/bash\n"
    sbatch_content += "#SBATCH --account={}\n".format(CC_ACCOUNT)
    sbatch_content += "#SBATCH --time={}\n".format(exp_config["run_time"])
    sbatch_content += "#SBATCH --cpus-per-task=1\n"
    sbatch_content += "#SBATCH --mem=3G\n"
    sbatch_content += "#SBATCH --array=1-{}\n".format(num_runs)
    sbatch_content += "#SBATCH --output={}/%j.out\n".format(
        os.path.join(RUN_REPORT_DIR, "eval", exp_name)
    )
    sbatch_content += "module load python/3.10\n"
    sbatch_content += "module load mujoco\n"
    sbatch_content += "source ~/icl_env/bin/activate\n"
    sbatch_content += '`sed -n "${SLURM_ARRAY_TASK_ID}p"'
    sbatch_content += " < {}`\n".format(
        os.path.join(CONFIG_DIR, "eval-{}.dat".format(exp_name))
    )
    sbatch_content += "echo ${SLURM_ARRAY_TASK_ID}\n"
    sbatch_content += 'echo "Current working directory is `pwd`"\n'
    sbatch_content += 'echo "Running on hostname `hostname`"\n'
    sbatch_content += "echo ${learner_path}\n"
    sbatch_content += 'echo "Starting run at: `date`"\n'
    sbatch_content += "python3 {}/experiments/cc_utils/evaluation.py \\\n".format(
        REPO_PATH
    )
    sbatch_content += "  --learner_path=${learner_path} \n"
    sbatch_content += 'echo "Program test finished with exit code $? at: `date`"\n'

    script_path = os.path.join(sbatch_dir, f"run_all-eval-{exp_name}.sh")
    with open(
        script_path,
        "w+",
    ) as f:
        f.writelines(sbatch_content)

    run_all_content += "sbatch {}\n".format(script_path)

with open(
    "./sbatch_all_eval.sh",
    "w+",
) as f:
    f.writelines(run_all_content)
